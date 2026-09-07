#!/usr/bin/env python
r"""Phase 3 readiness audit: verifies the repo is scientifically frozen and
GCP-ready. Static checks only (no training, no GPU needed). Prints PASS / FAIL /
MANUAL CHECK REQUIRED per item and a final verdict.

Usage:  python scripts/verify_phase3_ready.py
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

REGIONS = ["WT", "TC", "ET"]
# Scientific control variables that MUST be identical across A0-A3.
CONTROL_KEYS = [
    ("experiment", "seed"), ("training", "epochs"), ("training", "patch_size"),
    ("training", "batch_size"), ("training", "augmentation"),
    ("model", "channel_n"), ("model", "hidden"), ("model", "steps"),
    ("model", "fire_rate"), ("optimizer", "learning_rate"),
    ("optimizer", "weight_decay"), ("loss", "tversky_beta"),
    ("loss", "focal_gamma"), ("ema", "enabled"), ("ema", "decay"),
    ("gradient", "clipping_enabled"), ("gradient", "max_norm"),
]


def _load(name):
    from src.experiment.config import load_config
    return load_config(os.path.join(_REPO, "configs", f"{name}.yaml"))


def check_ablation_matrix(repo):
    """A0=F/F, A1=T/F, A2=F/T, A3=T/T, and identical control variables."""
    want = {"ablation_baseline": (False, False), "ablation_se": (True, False),
            "ablation_spatial": (False, True), "ablation_full": (True, True)}
    cfgs = {n: _load(n) for n in want}
    for n, (a, s) in want.items():
        got = (cfgs[n].get("model", "use_attention"), cfgs[n].get("model", "use_spatial"))
        if got != (a, s):
            return False, f"{n} switches {got} != expected {(a, s)}"
    # control variables identical across all four
    ref = cfgs["ablation_baseline"]
    for n, cfg in cfgs.items():
        for sec, key in CONTROL_KEYS:
            if cfg.get(sec, key) != ref.get(sec, key):
                return False, f"{n} differs in {sec}.{key} " \
                              f"({cfg.get(sec,key)} != {ref.get(sec,key)})"
        if cfg.get("data", "split_file") != "split/master_split.json":
            return False, f"{n} does not reference the master split"
    return True, "A0=F/F A1=T/F A2=F/T A3=T/T; control vars identical; master split shared"


def check_final_config(repo):
    cfg = _load("gcp_full")
    checks = {
        "name": (cfg.name, "GLO-NCA-V2-final"),
        "epochs": (cfg.get("training", "epochs"), 200),
        "patch_size": (cfg.get("training", "patch_size"), 96),
        "batch_size": (cfg.get("training", "batch_size"), 1),
        "seed": (cfg.seed, 42),
        "augmentation": (cfg.get("training", "augmentation"), "light"),
        "use_attention": (cfg.get("model", "use_attention"), True),
        "use_spatial": (cfg.get("model", "use_spatial"), True),
    }
    bad = [f"{k}={v[0]}!={v[1]}" for k, v in checks.items() if v[0] != v[1]]
    if bad:
        return False, "; ".join(bad)
    if cfg.get("data", "split_file") != "split/master_split.json":
        return False, "final config does not reference the master split"
    return True, "name/epochs/patch/batch/seed/aug/SE/spatial + master split OK"


def main() -> int:
    results = []
    def add(name, status, detail=""):
        results.append((name, status))
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))

    # branch
    try:
        br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                     cwd=_REPO, text=True).strip()
        add("branch v2", "PASS" if br == "v2" else "FAIL", br)
    except Exception:
        add("branch v2", "MANUAL CHECK REQUIRED", "git unavailable")

    # required configs present
    need = ["ablation_baseline", "ablation_se", "ablation_spatial",
            "ablation_full", "gcp_full", "smoke_test"]
    missing = [n for n in need if not os.path.exists(os.path.join(_REPO, "configs", f"{n}.yaml"))]
    add("required configs", "PASS" if not missing else "FAIL", f"missing {missing}" if missing else "")

    ok, msg = check_ablation_matrix(_REPO); add("A0/A1/A2/A3 + control vars", "PASS" if ok else "FAIL", msg)
    ok, msg = check_final_config(_REPO); add("final config integrity", "PASS" if ok else "FAIL", msg)

    # master split support in datasource
    from src.experiment import datasource
    add("master split support",
        "PASS" if hasattr(datasource, "load_master_split") else "FAIL")

    # frozen-test / threshold discipline (static string audit of the runner)
    runner = open(os.path.join(_REPO, "src", "experiment", "runner.py"), encoding="utf-8").read()
    add("threshold tuning validation-only",
        "PASS" if "tune_thresholds(val_pairs)" in runner else "MANUAL CHECK REQUIRED")
    add("test uses frozen thresholds",
        "PASS" if "ME.score(test_pairs, thresholds)" in runner else "MANUAL CHECK REQUIRED")
    add("best on smoothed validation",
        "PASS" if "vm_smooth" in runner and "best" in runner else "MANUAL CHECK REQUIRED")

    # HD95 voxels, no TTA/ensemble
    me = open(os.path.join(_REPO, "src", "experiment", "metrics_eval.py"), encoding="utf-8").read()
    add("HD95 in voxels (not mm)",
        "PASS" if "hd95" in me and "mm" not in me.lower() else "MANUAL CHECK REQUIRED")
    joined = runner + me
    add("no TTA / ensemble",
        "PASS" if ("tta" not in joined.lower() and "ensemble" not in joined.lower()) else "FAIL")

    # methodology files unchanged since Phase 2 commit
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", "9a2bd62", "HEAD", "--",
             "src/models", "src/agents", "src/losses", "src/datasets"],
            cwd=_REPO, text=True).strip()
        add("research methodology unchanged", "PASS" if not diff else "FAIL",
            diff or "no changes under src/{models,agents,losses,datasets}")
    except Exception:
        add("research methodology unchanged", "MANUAL CHECK REQUIRED", "git unavailable")

    # thesis generators present
    gens = ["make_ablation_table.py", "make_thesis_tables.py", "make_figures.py"]
    gm = [g for g in gens if not os.path.exists(os.path.join(_HERE, g))]
    add("thesis generators", "PASS" if not gm else "FAIL", f"missing {gm}" if gm else "")

    # cloud scripts present
    cs = os.path.join(_REPO, "cloud", "scripts")
    add("cloud scripts", "PASS" if os.path.isdir(cs) and
        os.path.exists(os.path.join(cs, "run_training.sh")) else "FAIL")

    # no credentials tracked
    try:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=_REPO, text=True)
        leak = [l for l in tracked.splitlines()
                if l.endswith("gcp.env") or l.endswith(".json") and "service" in l.lower()]
        add("no credentials tracked", "PASS" if not leak else "FAIL",
            f"tracked: {leak}" if leak else "")
    except Exception:
        add("no credentials tracked", "MANUAL CHECK REQUIRED")

    fails = [n for n, s in results if s == "FAIL"]
    print("\n" + "=" * 50)
    if fails:
        print("PHASE 3 READINESS: NOT READY")
        print("blockers: " + ", ".join(fails))
        return 1
    print("PHASE 3 READINESS: READY (static checks). GPU/GCP items verified by "
          "preflight_gcp.py on the VM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
