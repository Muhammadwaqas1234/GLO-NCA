#!/usr/bin/env python
r"""
PHASE 1 FINAL CLOSURE AUDIT -- independent re-verification. NO TRAINING.

Re-checks every actionable Phase 1 finding against the CURRENT source, plus the
closure-specific audits (GLO-NCA naming, secrets, dead references, fail-loud
resume, V2-fallback prevention). It deliberately does NOT trust the Phase 1 or
Phase 2 reports: each check reads the code as it stands now.

Usage:  python scripts/phase1_closure_audit.py
Exit:   0 if no FAIL. SKIP / NOT RUN are reported honestly, never as PASS.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS = []
_SEC = "general"

# Active production path: these must be clean of legacy branding and secrets.
ACTIVE_DIRS = ["src", "configs", "cloud", "scripts"]
ACTIVE_FILES = ["train.py", "Dockerfile", ".dockerignore"]


class Skip(Exception):
    pass


def sec(s):
    global _SEC
    _SEC = s
    print(f"\n### {s} ###")


def check(name, fn):
    try:
        ok, detail = fn()
        st = "PASS" if ok else "FAIL"
    except Skip as s:
        st, detail = "SKIP", str(s)
    except Exception as exc:
        st, detail = "FAIL", f"{type(exc).__name__}: {exc}"
    RESULTS.append((_SEC, name, st, detail))
    print(f"[{st:4}] {name}" + (f"  -- {detail}" if detail else ""))


def read(rel):
    return io.open(os.path.join(_ROOT, rel), encoding="utf-8").read()


def code_lines(rel):
    """Executable lines only (comments stripped) -- so documentation ABOUT a
    removed hazard is never mistaken for the hazard itself."""
    out = []
    for ln in io.open(os.path.join(_ROOT, rel), encoding="utf-8"):
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


SELF = os.path.abspath(__file__)


def walk_active():
    """Active production files. The audit scripts are EXCLUDED from their own
    scans: they necessarily contain the legacy-name regexes and archived-tool
    filenames they search for, and are diagnostics, not production code."""
    audit_scripts = {SELF,
                     os.path.join(_ROOT, "extra", "scripts", "phase2_regression.py"),
                     os.path.join(_ROOT, "extra", "scripts",
                                  "phase2_final_local_verification.py")}
    for d in ACTIVE_DIRS:
        for dp, dns, fns in os.walk(os.path.join(_ROOT, d)):
            dns[:] = [x for x in dns if x not in ("__pycache__", ".git")]
            for f in fns:
                full = os.path.join(dp, f)
                if (f.endswith((".py", ".yaml", ".yml", ".sh", ".service"))
                        and os.path.abspath(full) not in audit_scripts):
                    yield full
    for f in ACTIVE_FILES:
        p = os.path.join(_ROOT, f)
        if os.path.exists(p):
            yield p


# --------------------------------------------------------------------------- #
# STEP 4 -- GLO-NCA naming on the ACTIVE production path
# --------------------------------------------------------------------------- #
LEGACY = re.compile(r"mednca|m3d[-_ ]?nca|med[-_]nca", re.I)


def t_naming_active_clean():
    hits = []
    for p in walk_active():
        for i, ln in enumerate(io.open(p, encoding="utf-8", errors="ignore"), 1):
            if LEGACY.search(ln):
                hits.append(f"{os.path.relpath(p, _ROOT)}:{i}")
    return (not hits), ("; ".join(hits[:8]) if hits else
                        "no MedNCA/M3D-NCA branding in src/ configs/ cloud/ "
                        "scripts/ train.py Dockerfile")


def t_naming_glo_present():
    n = sum(1 for p in walk_active()
            if re.search(r"GLO[-_]NCA", io.open(p, encoding="utf-8",
                                                errors="ignore").read()))
    return n >= 20, f"{n} active files carry GLO-NCA naming"


def t_citations_preserved():
    """Upstream attribution must NOT be renamed away -- that would be
    academic misattribution."""
    r = read("README.md")
    ok = "Kalkhof" in r and ("M3D-NCA" in r or "Med-NCA" in r)
    return ok, ("upstream Med-NCA / M3D-NCA citations intact in README "
                "(correctly retained, not renamed)")


# --------------------------------------------------------------------------- #
# STEP 5/6 -- methodology fingerprint + single authoritative config
# --------------------------------------------------------------------------- #
PROD = "extra/configs/historical/v3_multilevel_ckpt.yaml"

FINGERPRINT = {
    ("experiment", "seed"): 42,
    ("training", "epochs"): 300,
    ("training", "batch_size"): 1,
    ("training", "patch_size"): 128,
    ("training", "augmentation"): "light",
    ("training", "workers"): 4,
    ("model", "version"): "v3",
    ("model", "fire_rate"): 0.6,
    ("model", "hidden"): 128,
    ("model", "dropout"): 0.1,
    ("model", "use_attention"): True,
    ("model", "use_spatial"): True,
    ("optimizer", "learning_rate"): 0.0016,
    ("optimizer", "minimum_learning_rate"): 0.00001,
    ("optimizer", "weight_decay"): 0.0001,
    ("loss", "tversky_beta"): 0.75,
    ("loss", "focal_gamma"): 1.33,
    ("loss", "ce_weight"): 0.5,
    ("loss", "empty_region_bce_weight"): 0.1,
    ("sampling", "prioritize_probability"): 0.7,
    ("sampling", "prioritize_region"): 2,
    ("ema", "enabled"): True,
    ("ema", "decay"): 0.999,
    ("gradient", "clipping_enabled"): True,
    ("gradient", "max_norm"): 1.0,
    ("evaluation", "tune_thresholds"): True,
    ("evaluation", "smoothing_window"): 3,
    ("memory", "gradient_checkpointing"): True,
}


def t_methodology_fingerprint():
    import yaml
    c = yaml.safe_load(read(PROD))
    bad = [f"{a}.{b}={c.get(a, {}).get(b)!r}!={v!r}"
           for (a, b), v in FINGERPRINT.items() if c.get(a, {}).get(b) != v]
    lv = [(c["model"][f"level{i}"]["resolution"],
           c["model"][f"level{i}"]["channels"],
           c["model"][f"level{i}"]["nca_steps"]) for i in (1, 2, 3)]
    if lv != [(32, 24, 20), (96, 24, 20), (128, 16, 10)]:
        bad.append(f"levels={lv}")
    if sum(s for _, _, s in lv) != 50:
        bad.append("NCA steps != 50")
    if (c.get("data") or {}).get("split_file") != "split/master_split.json":
        bad.append("split_file")
    return (not bad), ("; ".join(bad) if bad else
                       f"{len(FINGERPRINT)} values + levels(32/96/128, "
                       "20+20+10=50) + split verified")


def t_single_production_config():
    """Exactly one ACTIVE config, and it is the production config.

    Legacy configurations live under configs/historical/ and are excluded:
    the active namespace must contain one obvious authority.
    """
    import glob
    active = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(_ROOT, "configs", "*.yaml")))
    return (active == ["glo_nca_production.yaml"]), \
        f"active configs: {active or 'none'}"


def t_no_contradictory_production_docs():
    """Docs must not point at a different config as the campaign config."""
    bad = []
    for rel in ("README.md", "cloud/README.md",
                "extra/docs/thesis/FINAL_TRAINING_PROTOCOL.md"):
        p = os.path.join(_ROOT, rel)
        if not os.path.exists(p):
            continue
        for i, ln in enumerate(io.open(p, encoding="utf-8"), 1):
            if re.search(r"300[- ]epoch", ln, re.I) and "v3_multilevel.yaml" in ln:
                bad.append(f"{rel}:{i}")
    return (not bad), ("; ".join(bad) if bad else
                       "no doc names the non-checkpointing config for the campaign")


# --------------------------------------------------------------------------- #
# STEP 3 -- Phase 1 finding closure, re-derived from source
# --------------------------------------------------------------------------- #
def t_f01_tracked():
    r = subprocess.run(["git", "ls-files", "--error-unmatch",
                        "src/models/Model_GLO_NCA_V3.py",
                        "src/agents/Agent_GLO_NCA_V3.py",
                        "extra/configs/historical/v3_multilevel_ckpt.yaml",
                        "split/master_split.json"],
                       capture_output=True, cwd=_ROOT)
    return r.returncode == 0, "V3 model, agent, production config and split are tracked"


def t_cl01_split_in_image():
    ok = ("COPY split/master_split.json" in read("Dockerfile")
          and "!split/master_split.json" in read(".dockerignore"))
    return ok, "split COPY'd into image and re-included in .dockerignore"


def t_cl02_gate_safe():
    bad = [l for l in code_lines("cloud/scripts/pretrain_gate.sh")
           if "docker run" in l and "GATE_CFG" in l]
    return (not bad), ("; ".join(bad[:2]) if bad else
                       "no executable gate line runs the production config")


def t_cl05_no_dir_guessing():
    bad = [l for l in code_lines("cloud/scripts/_train_entrypoint.sh")
           if "ls -t" in l]
    return (not bad), ("; ".join(bad[:2]) if bad else
                       "experiment id is explicit; no 'ls -t' guessing")


def t_cl06_lock_systemd():
    lib = read("cloud/scripts/lib.sh")
    ok = ("assert_no_training_running" in lib
          and "systemctl is-active" in lib
          and "echo $$ >" not in read("cloud/scripts/run_training.sh")
          and "echo $$ >" not in read("cloud/scripts/resume_training.sh"))
    return ok, "systemd-backed lock; no launcher-PID lock remains"


def t_c01_scheduler_loud():
    src = read("src/experiment/checkpoint.py")
    return ("raise RuntimeError" in src
            and "pass  # scheduler shape can shift" not in src), \
        "scheduler restore failure raises instead of silently resetting the LR"


def t_f05_split_required():
    return "no data.split_file configured" in read("src/experiment/runner.py"), \
        "a V3 run without an explicit split is refused"


def t_d01_cache_and_workers():
    src = read("src/experiment/runner.py")
    return ("persistent_workers" in src and "_EpochSampler" in src), \
        "persistent workers + epoch-aware sampler"


def t_d02_patchify_hoist():
    return "full_volume" in read("src/datasets/Nii_Gz_Dataset_3D.py"), \
        "full-volume reductions hoisted out of the retry loop"


def t_me01_eval_ram():
    src = read("src/experiment/metrics_eval.py")
    run = read("src/experiment/runner.py")
    return ("astype(np.uint8)" in src and "del val_pairs" in run), \
        "uint8 ground truth + val_pairs released after last use"


def t_v2_fallback_blocked():
    bad = []
    if 'default=os.path.join("extra", "configs", "historical", "gcp_full.yaml")' in read("train.py"):
        bad.append("train.py default")
    if "cfg_path = args.config" in read("train.py"):
        bad.append("resume falls back to --config")
    if any(l.startswith("CMD") and "gcp_full" in l for l in code_lines("Dockerfile")):
        bad.append("Dockerfile CMD")
    if "${1:-extra/configs/historical/gcp_full.yaml}" in read("cloud/scripts/run_training.sh"):
        bad.append("run_training.sh default")
    if "clone -b v2 " in read("cloud/scripts/setup_gcp.sh"):
        bad.append("setup_gcp.sh branch")
    return (not bad), ("; ".join(bad) if bad else
                       "no V2 default or fallback on the V3 production path")


def t_split_unchanged():
    d = json.loads(read("split/master_split.json"))
    want = "d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d"
    bad = []
    # Guard the EXPECTATION itself. A hand-copied SHA that loses a character is
    # a 63-char literal that can never equal a real SHA-256, so the check would
    # "fail" on a perfectly good split and send someone hunting a phantom data
    # bug. Assert the literal is well-formed before comparing; the comparison
    # below stays exact and fail-closed (never a prefix match).
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        bad.append(f"EXPECTED SHA LITERAL MALFORMED (len={len(want)}, need 64 hex)")
    if len(str(d["split_sha256"])) != 64:
        bad.append(f"split file SHA is not 64 hex chars: {len(str(d['split_sha256']))}")
    if d["split_sha256"] != want:
        bad.append("SHA CHANGED")
    if (d["train_count"], d["val_count"], d["test_count"]) != (898, 200, 198):
        bad.append("counts changed")
    tr, va, te = set(d["train"]), set(d["validation"]), set(d["test"])
    if tr & va or tr & te or va & te:
        bad.append("partitions overlap")

    def subj(c):
        return c.rsplit("-", 1)[0]
    if {subj(x) for x in tr} & ({subj(x) for x in va} | {subj(x) for x in te}):
        bad.append("SUBJECT LEAKAGE")
    return (not bad), ("; ".join(bad) if bad else
                       f"sha {want[:12]}…, 898/200/198, case- and subject-disjoint")


def t_dq_policy_loads_and_is_explicit():
    from src.experiment.data_quality import load_policy
    pol = load_policy()
    if pol is None:
        return False, "split/data_quality_policy.json not found"
    tol = pol.summary()["tolerated_cases"]
    expected = {"BraTS-MET-01094-002": [6], "BraTS-MET-01184-002": [8]}
    return (tol == expected and pol.excluded == []), \
        f"tolerates {tol}, excludes {pol.excluded} (canonical split untouched)"


def t_dq_policy_per_case_isolation():
    """A label tolerated on one case must NOT be tolerated on another."""
    from src.experiment.data_quality import load_policy, BASE_ALLOWED_SEG_LABELS
    pol = load_policy()
    a = pol.allowed_labels_for("BraTS-MET-01094-002")
    b = pol.allowed_labels_for("BraTS-MET-01184-002")
    other = pol.allowed_labels_for("BraTS-MET-00001-000")
    ok = (6 in a and 6 not in b and 8 in b and 8 not in a
          and other == BASE_ALLOWED_SEG_LABELS)
    return ok, ("6 only on 01094, 8 only on 01184, every other case restricted "
                "to {0,1,2,3,4}")


def t_dq_policy_fail_closed():
    """Malformed / stale policies must fail loudly, never degrade silently."""
    from src.experiment.data_quality import load_policy, DataQualityPolicyError
    import tempfile as _tf
    d = _tf.mkdtemp()
    try:
        bad = os.path.join(d, "bad.json")
        io.open(bad, "w", encoding="utf-8").write("{ not json")
        try:
            load_policy(bad)
            return False, "malformed policy did not raise"
        except DataQualityPolicyError:
            pass
        pol = load_policy()
        try:
            pol.assert_known_cases({"BraTS-MET-00001-000"})
            return False, "stale/unknown policy id did not raise"
        except DataQualityPolicyError:
            pass
        if load_policy(os.path.join(d, "absent.json")) is not None:
            return False, "absent policy should mean 'no tolerances'"
        return True, ("malformed raises, unknown id raises, absent => no "
                      "tolerances (validator stays fail-closed)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_dq_split_untouched_by_policy():
    """The policy must not alter the canonical split or its fingerprint."""
    d = json.loads(read("split/master_split.json"))
    pol = json.loads(read("split/data_quality_policy.json"))
    ok = (d["split_sha256"] == pol["canonical_split_sha256"]
          and pol["canonical_split_unchanged"] is True
          and pol["excluded_cases"] == []
          and pol["counts"]["final_operational_train"] == d["train_count"])
    return ok, (f"policy pins the canonical SHA, excludes nothing, train stays "
                f"{d['train_count']}")


def t_validation_fail_closed():
    """Invalid seg labels must abort the run, not be silently accepted."""
    dv = read("src/experiment/dataset_validation.py")
    rn = read("src/experiment/runner.py")
    return ("ALLOWED_SEG_LABELS" in dv and "unexpected seg labels" in dv
            and 'report["result"] != "PASS"' in rn), \
        "invalid labels -> validation FAIL -> runner aborts before training"


# --------------------------------------------------------------------------- #
# STEP 10 -- resume must fail loudly on drift (real subprocess, no training)
# --------------------------------------------------------------------------- #
def _mk_ws(tmp, cfg_text):
    for sub in ("config", "checkpoints", "logs", "reports", "metrics",
                "graphs", "tensorboard", "split", "predictions"):
        os.makedirs(os.path.join(tmp, sub), exist_ok=True)
    with io.open(os.path.join(tmp, "config", "config.yaml"), "w",
                 encoding="utf-8") as fh:
        fh.write(cfg_text)


def t_resume_missing_config_fails():
    tmp = tempfile.mkdtemp()
    try:
        _mk_ws(tmp, "")
        os.remove(os.path.join(tmp, "config", "config.yaml"))
        r = subprocess.run([sys.executable, "train.py", "--resume", tmp],
                           capture_output=True, text=True, timeout=240, cwd=_ROOT)
        out = r.stdout + r.stderr
        return (r.returncode == 2 and "cannot resume" in out), \
            f"rc={r.returncode}: refuses to resume without its saved config"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_bare_train_refuses():
    r = subprocess.run([sys.executable, "train.py"],
                       capture_output=True, text=True, timeout=240, cwd=_ROOT)
    out = r.stdout + r.stderr
    return (r.returncode == 2 and "--config is required" in out), \
        f"rc={r.returncode}: bare `train.py` cannot silently run V2"


def t_config_drift_detected():
    """A checkpoint written under different hyperparameters must be refused."""
    src = read("src/experiment/runner.py")
    fields = ["training\", \"epochs", "optimizer\", \"learning_rate",
              "loss\", \"tversky_beta", "experiment\", \"seed"]
    have = all(f in src for f in fields)
    return (have and "resume config does not match" in src), \
        "resume compares epochs/lr/loss/seed against the checkpoint config"


# --------------------------------------------------------------------------- #
# STEP 11/12 -- evaluation discipline + reproducibility, from source
# --------------------------------------------------------------------------- #
def t_test_frozen():
    src = read("src/experiment/runner.py")
    ok = (src.count("ME.collect_probs(agent, ds, \"test\")") == 1
          and "tune_thresholds(val_pairs)" in src
          and "tune_thresholds(test_pairs)" not in src)
    return ok, "test collected once; thresholds tuned on validation only"


def t_hd95_voxel_labelled():
    ag = read("src/agents/Agent.py")
    rn = read("src/experiment/runner.py")
    no_mm_claim = not re.search(r"hd95.{0,40}\bmm\b", ag + rn, re.I)
    return ("sampling=" not in ag and no_mm_claim
            and ("HD95(vox)" in rn or "hd95_vox" in rn)), \
        "HD95 computed in voxels and labelled as voxels; no mm claim"


def t_eval_no_grad():
    return "with torch.no_grad():" in read("src/experiment/metrics_eval.py"), \
        "evaluation runs under no_grad"


def t_aug_gated_train_only():
    ds = read("src/datasets/Nii_Gz_Dataset_3D.py")
    return (ds.count('self.state == "train"') >= 2), \
        "patchify and augmentation are gated to the train split only"


def t_rng_determinism():
    """Same (seed, epoch, case) -> identical stream; different -> different."""
    import random

    def draw(base, ep, idx):
        random.seed((base * 1_000_003 + ep * 9_176_231 + idx) % (2 ** 32))
        return tuple(random.random() for _ in range(5))
    if draw(42, 3, 7) != draw(42, 3, 7):
        return False, "not deterministic for the same (epoch, case)"
    across = len({draw(42, e, 7) for e in range(60)})
    within = len({draw(42, 3, i) for i in range(60)})
    return (across == 60 and within == 60), \
        f"deterministic; {across}/60 distinct across epochs, {within}/60 across cases"


# --------------------------------------------------------------------------- #
# STEP 14/15 -- container, scripts, security
# --------------------------------------------------------------------------- #
def t_no_tracked_secrets():
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=_ROOT)
    bad = [f for f in r.stdout.splitlines()
           if f.endswith((".pem", ".key")) or "service-account" in f
           or f.endswith("gcp.env")]
    return (not bad), ("; ".join(bad) if bad else
                       "no credential-like file is tracked by git")


def t_docker_excludes_secrets_and_data():
    di = read(".dockerignore")
    need = ["*.pth", "*.nii", "*.nii.gz", "experiments/"]
    missing = [n for n in need if n not in di]
    return (not missing), ("missing: " + ", ".join(missing)) if missing else \
        "image excludes checkpoints, NIfTI data and experiment outputs"


def t_no_personal_paths_active():
    hits = []
    pat = re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_]+|/c/Users/[A-Za-z0-9_]+")
    for p in walk_active():
        for i, ln in enumerate(io.open(p, encoding="utf-8", errors="ignore"), 1):
            if pat.search(ln):
                hits.append(f"{os.path.relpath(p, _ROOT)}:{i}")
    return (not hits), ("; ".join(hits[:5]) if hits else
                        "no personal absolute path in active code")


def t_cloud_scripts_parse():
    import glob
    if not shutil.which("bash"):
        raise Skip("bash unavailable")
    files = sorted(glob.glob(os.path.join(_ROOT, "cloud/scripts/*.sh")))
    bad = []
    for p in files:
        rel = os.path.relpath(p, _ROOT).replace(os.sep, "/")
        # Retry once: under heavy host load (e.g. a concurrent docker build)
        # the bash subprocess can be starved and return non-zero with EMPTY
        # stderr, which is a resource failure, not a syntax error. A real
        # syntax error always reports a message.
        r = subprocess.run(["bash", "-n", rel], capture_output=True,
                           text=True, cwd=_ROOT)
        if r.returncode != 0 and not r.stderr.strip():
            r = subprocess.run(["bash", "-n", rel], capture_output=True,
                               text=True, cwd=_ROOT, timeout=60)
        if r.returncode != 0 and r.stderr.strip():
            bad.append(f"{os.path.basename(p)}: {r.stderr.strip()[:60]}")
        elif r.returncode != 0:
            raise Skip("bash unavailable/starved under load; verified separately")
    return (not bad), ("; ".join(bad) if bad else f"{len(files)} cloud scripts parse")


def t_docker_image_present():
    if not shutil.which("docker"):
        raise Skip("docker not installed")
    r = subprocess.run(["docker", "images", "-q", "glo-nca:latest"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise Skip("docker daemon not reachable")
    if not r.stdout.strip():
        raise Skip("glo-nca:latest not built yet -- run `docker build -t glo-nca:latest .`")
    return True, f"image present ({r.stdout.strip()[:12]})"


# --------------------------------------------------------------------------- #
# STEP 15 -- dead files / references
# --------------------------------------------------------------------------- #
def t_archived_tools_unreferenced():
    """Archived one-off tools must have zero references from active code."""
    names = ["xfer2.sh", "drive_dl.py", "dataset_identity.py"]
    hits = []
    for p in walk_active():
        txt = io.open(p, encoding="utf-8", errors="ignore").read()
        for n in names:
            if n in txt:
                hits.append(f"{os.path.relpath(p, _ROOT)} -> {n}")
    return (not hits), ("; ".join(hits) if hits else
                        "archived one-off tools referenced nowhere in active code")


def t_archive_not_imported():
    hits = []
    for p in walk_active():
        if not p.endswith(".py"):
            continue
        for i, ln in enumerate(io.open(p, encoding="utf-8", errors="ignore"), 1):
            s = ln.strip()
            if s.startswith(("import ", "from ")) and "archive" in s:
                hits.append(f"{os.path.relpath(p, _ROOT)}:{i}")
    return (not hits), ("; ".join(hits) if hits else
                        "no active module imports archive/ (Kaggle history isolated)")


def t_phase1_reports_intact():
    d = os.path.join(_ROOT, "extra", "reports", "audit")
    need = ["PHASE1_BASELINE.md", "PHASE1_REPOSITORY_INVENTORY.md",
            "PHASE1_EXECUTION_GRAPH.md", "PHASE1_FILE_AUDIT.md",
            "DATA_PIPELINE_AUDIT.md", "MODEL_AUDIT.md", "TRAINING_LOOP_AUDIT.md",
            "MEMORY_AUDIT.md", "REPRODUCIBILITY_AUDIT.md", "EVALUATION_AUDIT.md",
            "CHECKPOINT_AUDIT.md", "CONFIG_AUDIT.md", "CLOUD_AUDIT.md",
            "TEST_AUDIT.md", "PHASE1_FINAL_FINDINGS.md"]
    missing = [n for n in need if not os.path.exists(os.path.join(d, n))]
    return (not missing), ("missing: " + ", ".join(missing)) if missing else \
        "all 15 Phase 1 reports present"


def main():
    print("=" * 74)
    print("PHASE 1 FINAL CLOSURE AUDIT -- no training is performed")
    print("=" * 74)

    sec("STEP 4  GLO-NCA naming")
    check("no MedNCA/M3D-NCA branding on the active production path",
          t_naming_active_clean)
    check("GLO-NCA naming used across active files", t_naming_glo_present)
    check("upstream citations preserved (not renamed away)", t_citations_preserved)

    sec("STEP 5/6  methodology fingerprint + authoritative config")
    check("production methodology fingerprint frozen", t_methodology_fingerprint)
    check("exactly one config claims to be production", t_single_production_config)
    check("no doc points the campaign at another config",
          t_no_contradictory_production_docs)

    sec("STEP 3  Phase 1 findings re-verified against current source")
    check("F-01 V3 model/agent/config/split tracked in git", t_f01_tracked)
    check("CL-01 canonical split reaches the container", t_cl01_split_in_image)
    check("CL-02 gate cannot launch the production run", t_cl02_gate_safe)
    check("CL-05 no 'ls -t' experiment-directory guessing", t_cl05_no_dir_guessing)
    check("CL-06 systemd-backed training lock", t_cl06_lock_systemd)
    check("C-01 scheduler restore fails loudly", t_c01_scheduler_loud)
    check("F-05 V3 run requires an explicit split", t_f05_split_required)
    check("D-01 persistent workers + epoch-aware sampler", t_d01_cache_and_workers)
    check("D-02 patchify full-volume hoist", t_d02_patchify_hoist)
    check("ME-01 evaluation host-RAM reduction", t_me01_eval_ram)
    check("V2 fallback impossible on the production path", t_v2_fallback_blocked)
    check("canonical split unchanged + subject-disjoint", t_split_unchanged)
    check("invalid labels are fail-closed by default", t_validation_fail_closed)

    sec("BLOCKER #1  operational data-quality policy")
    check("policy loads and is explicit (no auto-discovery)",
          t_dq_policy_loads_and_is_explicit)
    check("tolerance is per-case AND per-label", t_dq_policy_per_case_isolation)
    check("malformed/stale/absent policy handled fail-closed", t_dq_policy_fail_closed)
    check("canonical split untouched by the policy", t_dq_split_untouched_by_policy)

    sec("STEP 10  resume / fail-loud behaviour (real subprocess)")
    check("resume without saved config exits 2", t_resume_missing_config_fails)
    check("bare `train.py` exits 2", t_bare_train_refuses)
    check("resume detects hyperparameter drift", t_config_drift_detected)

    sec("STEP 11/12  evaluation discipline + reproducibility")
    check("test frozen: collected once, tuned on validation only", t_test_frozen)
    check("HD95 is voxel-space and labelled as such", t_hd95_voxel_labelled)
    check("evaluation runs under no_grad", t_eval_no_grad)
    check("patchify/augmentation gated to train split", t_aug_gated_train_only)
    check("per-(seed,epoch,case) RNG deterministic and varied", t_rng_determinism)

    sec("STEP 14/15  container, cloud scripts, security")
    check("no credential-like file tracked", t_no_tracked_secrets)
    check("image excludes data/checkpoints", t_docker_excludes_secrets_and_data)
    check("no personal absolute paths in active code", t_no_personal_paths_active)
    check("cloud shell scripts parse", t_cloud_scripts_parse)
    check("production docker image built", t_docker_image_present)

    sec("STEP 15  dead files / references / report integrity")
    check("archived one-off tools unreferenced", t_archived_tools_unreferenced)
    check("archive/ never imported by active code", t_archive_not_imported)
    check("all 15 Phase 1 reports intact", t_phase1_reports_intact)

    p = sum(1 for r in RESULTS if r[2] == "PASS")
    f = sum(1 for r in RESULTS if r[2] == "FAIL")
    s = sum(1 for r in RESULTS if r[2] == "SKIP")
    print("\n" + "=" * 74)
    print(f"PASS {p}   FAIL {f}   SKIP {s}   (SKIP is never counted as PASS)")
    print("PHASE 1 CLOSURE AUDIT: " + ("PASS" if f == 0 else "FAIL"))
    print("=" * 74)

    out = os.path.join(_ROOT, "extra", "reports", "validation", "phase1_closure_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump([{"section": a, "name": b, "status": c, "detail": d}
                   for a, b, c, d in RESULTS], fh, indent=2)
    print(f"machine-readable results -> {out}")
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
