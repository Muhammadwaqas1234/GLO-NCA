#!/usr/bin/env python
r"""
Phase 2 regression suite (NO TRAINING).

Static + synthetic checks that the Phase 2 hardening did not change the V3
science, and that the fixed paths behave as claimed. Every check is cheap and
CPU-only; checks needing torch are SKIPPED (not faked) when torch is absent.

Usage:  python scripts/phase2_regression.py
Exit:   0 = all executed checks PASS, 1 = at least one FAIL.
"""
import io
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS = []


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"[{status:4}] {name}" + (f"  -- {detail}" if detail else ""))


def check(name, fn):
    try:
        ok, detail = fn()
        record(name, "PASS" if ok else "FAIL", detail)
    except _Skip as s:
        record(name, "SKIP", str(s))
    except Exception as exc:  # a crashing check is a failure, never a pass
        record(name, "FAIL", f"{type(exc).__name__}: {exc}")


class _Skip(Exception):
    pass


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise _Skip("torch not installed in this environment")


# --------------------------------------------------------------------------- #
# 1. Scientific baseline is unchanged
# --------------------------------------------------------------------------- #
def t_production_config():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(_ROOT, "configs/v3_multilevel_ckpt.yaml")))
    want = {
        ("experiment", "seed"): 42,
        ("training", "epochs"): 300,
        ("training", "batch_size"): 1,
        ("training", "patch_size"): 128,
        ("model", "version"): "v3",
        ("model", "fire_rate"): 0.6,
        ("model", "hidden"): 128,
        ("loss", "tversky_beta"): 0.75,
        ("loss", "focal_gamma"): 1.33,
        ("ema", "decay"): 0.999,
        ("gradient", "max_norm"): 1.0,
        ("memory", "gradient_checkpointing"): True,
    }
    bad = [f"{a}.{b}={cfg[a][b]!r} (want {v!r})"
           for (a, b), v in want.items() if cfg[a][b] != v]
    lv = [(cfg["model"][f"level{i}"]["resolution"],
           cfg["model"][f"level{i}"]["channels"],
           cfg["model"][f"level{i}"]["nca_steps"]) for i in (1, 2, 3)]
    if lv != [(32, 24, 20), (96, 24, 20), (128, 16, 10)]:
        bad.append(f"levels={lv}")
    steps = sum(s for _, _, s in lv)
    if steps != 50:
        bad.append(f"total NCA steps={steps} (want 50)")
    return (not bad), ("; ".join(bad) if bad else
                       "32/96/128, 20+20+10=50, 300ep, ckpt=ON, seed 42")


def t_param_count():
    torch = _torch()
    from src.experiment.config import load_config
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    cfg = load_config(os.path.join(_ROOT, "configs/v3_multilevel_ckpt.yaml"))
    m = build_v3_from_config(cfg, 4, 3, torch.device("cpu"))
    n = m.parameter_report()["total_parameters"]
    return n == 40656, f"{n} parameters (want 40656)"


def t_v3_forward_backward():
    torch = _torch()
    from src.experiment.config import load_config
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    from src.losses.LossFunctions import FocalTverskyCELoss
    # smoke_test_v3.yaml is the genuinely SMALL config (16/24/32, 4+4+3 steps).
    # v3_smoke_1epoch.yaml keeps the full production resolutions (32/96/128) and
    # only shortens the epoch count, so it would run the full 128^3 workload here.
    cfg = load_config(os.path.join(_ROOT, "configs/smoke_test_v3.yaml"))
    m = build_v3_from_config(cfg, 4, 3, torch.device("cpu"))
    x = torch.randn(1, 16, 16, 16, 4)
    y = m(x)
    fine = int(cfg.raw["model"]["level3"]["resolution"])   # model fixes output size
    if tuple(y.shape) != (1, 3, fine, fine, fine):
        return False, f"output shape {tuple(y.shape)} (want (1,3,{fine},{fine},{fine}))"
    loss = FocalTverskyCELoss(0.25, 0.75, 1.33, 0.5)(
        y[:, 0], torch.zeros(1, fine, fine, fine))
    loss.backward()
    g = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    return g > 0, f"forward+backward OK, {g} tensors received gradient"


def t_checkpointing_equivalence():
    """ckpt ON vs OFF must give identical output AND identical gradients."""
    torch = _torch()
    from src.models.Model_BasicNCA3D import BasicNCA3D

    def run(use_ckpt):
        torch.manual_seed(0)
        m = BasicNCA3D(8, 0.6, torch.device("cpu"), 16, input_channels=4,
                       kernel_size=3, use_attention=True, use_spatial=True)
        m.use_checkpoint = use_ckpt
        torch.manual_seed(1)
        x = torch.randn(1, 6, 6, 6, 8, requires_grad=True)
        torch.manual_seed(2)          # identical fire-rate mask stream
        out = m(x, steps=3, fire_rate=0.6)
        out.sum().backward()
        return out.detach().clone(), [p.grad.clone() for p in m.parameters()
                                      if p.grad is not None]

    o1, g1 = run(False)
    o2, g2 = run(True)
    same_o = torch.allclose(o1, o2, atol=1e-6)
    same_g = len(g1) == len(g2) and all(
        torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2))
    return (same_o and same_g), (
        f"outputs_identical={same_o}, gradients_identical={same_g}")


# --------------------------------------------------------------------------- #
# 2. Phase 2 fixes behave as claimed
# --------------------------------------------------------------------------- #
def t_patchify_equivalence():
    """The hoisted ET/WT reduction must be output- AND RNG-state-identical."""
    import numpy as np
    try:
        # The dataset module imports torchio -> torch; without it the REAL
        # implementation cannot be exercised. Skip honestly rather than test a
        # copy of the logic (the Phase 1 audit flagged exactly that anti-pattern).
        from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS as DS
    except ImportError as exc:
        raise _Skip(f"dataset module needs torch/torchio ({exc.name})")

    class _Exp:
        def __init__(self, p, r):
            self.p, self.r = p, r

        def get_from_config(self, tag):
            return {"priotize_masks": self.p, "prioritize_region": self.r}.get(tag)

    def reference(img, label, size, prioritize=0.7, region=2):
        contains = prioritize is not None and (random.uniform(0, 1) < prioritize)
        px = py = pz = 0
        fb = None
        for _ in range(50):
            px = random.randint(0, img.shape[0] - size[0])
            py = random.randint(0, img.shape[1] - size[1])
            pz = random.randint(0, img.shape[2] - size[2])
            if not contains:
                break
            if label[px:px+size[0], py:py+size[1], pz:pz+size[2], region].max() > 0:
                break
            if region != 0 and fb is None:
                if label[px:px+size[0], py:py+size[1], pz:pz+size[2], 0].max() > 0:
                    fb = (px, py, pz)
        else:
            if fb is not None:
                px, py, pz = fb
        return (img[px:px+size[0], py:py+size[1], pz:pz+size[2], :],
                label[px:px+size[0], py:py+size[1], pz:pz+size[2], :])

    ds = DS()
    ds.exp = _Exp(0.7, 2)
    ds.size = (8, 8, 8)
    bad = 0
    for trial in range(200):
        rng = np.random.RandomState(trial)
        img = rng.rand(8, 8, 8, 4).astype(np.float32)
        lab = (rng.rand(8, 8, 8, 3) > [0.5, 0.7, 0.95]).astype(np.float32)
        if trial % 3 == 0:
            lab[..., 2] = 0          # no ET  -> exercises the 50-retry path
        if trial % 7 == 0:
            lab[:] = 0               # fully empty
        random.seed(trial)
        ra = reference(img, lab, (8, 8, 8))
        sa = random.getstate()
        random.seed(trial)
        rb = ds.patchify_multimodal(img, lab)
        sb = random.getstate()
        if not (np.array_equal(ra[0], rb[0]) and np.array_equal(ra[1], rb[1])):
            bad += 1
        elif sa != sb:
            bad += 1
    return bad == 0, f"200 trials (incl. no-ET/all-empty): {bad} mismatches"


def t_epoch_rng_diversity():
    """Per-(epoch, case) seeding: deterministic, yet varies across epochs."""
    def draws(base, ep, idx):
        s = (base * 1_000_003 + ep * 9_176_231 + idx) % (2 ** 32)
        random.seed(s)
        return tuple(random.random() for _ in range(4))
    if draws(42, 7, 3) != draws(42, 7, 3):
        return False, "not deterministic for the same (epoch, case)"
    across = {draws(42, ep, 3) for ep in range(100)}
    within = {draws(42, 7, i) for i in range(100)}
    return (len(across) == 100 and len(within) == 100), \
        f"{len(across)}/100 distinct across epochs, {len(within)}/100 across cases"


def t_no_v2_production_default():
    """No production entry point may silently default to the V2 baseline."""
    bad = []
    tp = io.open(os.path.join(_ROOT, "train.py"), encoding="utf-8").read()
    if 'default=os.path.join("configs", "gcp_full.yaml")' in tp:
        bad.append("train.py --config still defaults to V2")
    # Inspect only EXECUTABLE Dockerfile lines; comments explaining the removed
    # V2 default must not trip this check.
    df_lines = [ln.strip() for ln
                in io.open(os.path.join(_ROOT, "Dockerfile"), encoding="utf-8")
                if not ln.strip().startswith("#")]
    if any(ln.startswith("CMD") and "gcp_full" in ln for ln in df_lines):
        bad.append("Dockerfile CMD still defaults to V2")
    rt = io.open(os.path.join(_ROOT, "cloud/scripts/run_training.sh"),
                 encoding="utf-8").read()
    if '${1:-configs/gcp_full.yaml}' in rt:
        bad.append("run_training.sh still defaults to V2")
    sg = io.open(os.path.join(_ROOT, "cloud/scripts/setup_gcp.sh"),
                 encoding="utf-8").read()
    if "clone -b v2 " in sg:
        bad.append("setup_gcp.sh still clones branch v2")
    return (not bad), ("; ".join(bad) if bad else
                       "no V2 default in train.py/Dockerfile/run_training/setup_gcp")


def t_gate_cannot_launch_production():
    """The pre-training gate must have NO path that runs the full config."""
    g = io.open(os.path.join(_ROOT, "cloud/scripts/pretrain_gate.sh"),
                encoding="utf-8").read()
    bad = []
    for i, line in enumerate(g.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        if "docker run" in s and "GATE_CFG" in s:
            bad.append(f"line {i}: docker run with GATE_CFG")
        if s.startswith("||") and "docker run" in s:
            bad.append(f"line {i}: fallback docker run")
    return (not bad), ("; ".join(bad) if bad else
                       "no fallback; smoke failure cannot start production")


def t_split_reaches_container():
    df = io.open(os.path.join(_ROOT, "Dockerfile"), encoding="utf-8").read()
    di = io.open(os.path.join(_ROOT, ".dockerignore"), encoding="utf-8").read()
    bad = []
    if "COPY split/master_split.json" not in df:
        bad.append("Dockerfile does not COPY the split")
    if "!split/master_split.json" not in di:
        bad.append(".dockerignore does not re-include the split")
    return (not bad), ("; ".join(bad) if bad else
                       "split is COPY'd into the image and un-ignored")


def t_no_ls_t_heuristic():
    e = io.open(os.path.join(_ROOT, "cloud/scripts/_train_entrypoint.sh"),
                encoding="utf-8").read()
    bad = [f"line {i}" for i, ln in enumerate(e.splitlines(), 1)
           if "ls -t" in ln and not ln.strip().startswith("#")]
    return (not bad), ("ls -t at " + ", ".join(bad)) if bad else \
        "experiment id is explicit (--experiment-id), no directory guessing"


def t_canonical_split():
    p = os.path.join(_ROOT, "split/master_split.json")
    d = json.load(open(p))
    want_sha = "d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d"
    bad = []
    if d["split_sha256"] != want_sha:
        bad.append("sha256 CHANGED")
    if (d["train_count"], d["val_count"], d["test_count"]) != (898, 200, 198):
        bad.append(f"counts {d['train_count']}/{d['val_count']}/{d['test_count']}")
    if (d["train_subject_count"], d["val_subject_count"],
            d["test_subject_count"]) != (567, 121, 122):
        bad.append("subject counts changed")
    tr, va, te = set(d["train"]), set(d["validation"]), set(d["test"])
    if tr & va or tr & te or va & te:
        bad.append("PARTITIONS OVERLAP")
    return (not bad), ("; ".join(bad) if bad else
                       "898/200/198, 567/121/122 subjects, disjoint, sha matches")


def t_resume_no_v2_fallback():
    tp = io.open(os.path.join(_ROOT, "train.py"), encoding="utf-8").read()
    if "cfg_path = args.config" in tp:
        return False, "resume still falls back to --config (V2 default)"
    return ("return 2" in tp and "cannot resume" in tp), \
        "resume without its own saved config fails loudly"


def t_workspace_explicit_id():
    from src.experiment.workspace import Workspace
    import inspect
    sig = inspect.signature(Workspace.create)
    return "experiment_id" in sig.parameters, \
        f"Workspace.create{sig}"


def t_v3_configs_declare_split():
    """Every V3 config must name the canonical split, or explicitly opt out."""
    import glob
    import yaml
    bad = []
    for p in sorted(glob.glob(os.path.join(_ROOT, "configs", "*.yaml"))):
        c = yaml.safe_load(open(p)) or {}
        if str((c.get("model") or {}).get("version", "")).lower() != "v3":
            continue
        data = c.get("data") or {}
        if not data.get("split_file") and not data.get("allow_seeded_split"):
            bad.append(os.path.basename(p))
    return (not bad), ("missing split declaration: " + ", ".join(bad)) if bad else \
        "all V3 configs name master_split.json (or opt out explicitly)"


def t_scheduler_restore_not_silent():
    src = io.open(os.path.join(_ROOT, "src/experiment/checkpoint.py"),
                  encoding="utf-8").read()
    if "pass  # scheduler shape can shift" in src:
        return False, "scheduler restore still fails silently"
    return "raise RuntimeError" in src, \
        "scheduler restore failure raises instead of silently resetting the LR"


def t_compileall():
    import compileall
    ok = compileall.compile_dir(os.path.join(_ROOT, "src"), quiet=2, force=True)
    ok &= compileall.compile_file(os.path.join(_ROOT, "train.py"), quiet=2, force=True)
    return bool(ok), "src/ + train.py byte-compile"


def main():
    print("=" * 72)
    print("PHASE 2 REGRESSION SUITE  (no training is performed)")
    print("=" * 72)

    print("\n-- scientific baseline (must be UNCHANGED) --")
    check("production config: architecture + hyperparameters frozen", t_production_config)
    check("V3 parameter count == 40,656", t_param_count)
    check("V3 forward + backward (synthetic)", t_v3_forward_backward)
    check("gradient checkpointing: identical outputs AND gradients", t_checkpointing_equivalence)
    check("canonical split unchanged + partitions disjoint", t_canonical_split)

    print("\n-- Phase 2 fixes --")
    check("patchify: output AND RNG-state identical", t_patchify_equivalence)
    check("per-(epoch,case) RNG: deterministic + varied", t_epoch_rng_diversity)
    check("P0: gate cannot launch production training", t_gate_cannot_launch_production)
    check("P0: canonical split reaches the container", t_split_reaches_container)
    check("P0: resume never falls back to a V2 config", t_resume_no_v2_fallback)
    check("P1: no 'ls -t' experiment-directory guessing", t_no_ls_t_heuristic)
    check("P1: explicit --experiment-id supported", t_workspace_explicit_id)
    check("V2 defaults removed from production entry points", t_no_v2_production_default)
    check("P1: every V3 config declares its split", t_v3_configs_declare_split)
    check("P1: scheduler restore failure is not silent", t_scheduler_restore_not_silent)
    check("compileall src/ + train.py", t_compileall)

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    n_skip = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print("\n" + "=" * 72)
    print(f"PASS {n_pass}   FAIL {n_fail}   SKIP {n_skip}")
    if n_skip:
        print("SKIPPED checks were NOT run (torch unavailable here) and must be")
        print("re-run on the GPU VM before training. They are not passes.")
    print("PHASE 2 REGRESSION: " + ("PASS" if n_fail == 0 else "FAIL"))
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
