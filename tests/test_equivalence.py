r"""Check that the trainer reproduces the original Kaggle v7 script exactly."""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REFERENCE_COMMIT = "kaggle-v7-reference"   # tag: original script and library
SMALL = {"EPOCHS": 2, "N_PATIENTS": 8, "ENSEMBLE_N": 2, "NUM_WORKERS": 0}
OVERRIDES = ["training.epochs=2", "data.n_patients=8", "evaluation.ensemble_n=2",
             "training.workers=0"]
R = []


def check(name, ok, detail=""):
    R.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}", flush=True)


def max_diff(a, b):
    """Largest absolute difference between two nested JSON structures of numbers."""
    if isinstance(a, dict):
        return max([max_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list):
        return max([max_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)):
        if a != a and b != b:        # both NaN
            return 0.0
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def run(cmd, env, cwd=REPO):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="v7eq_")
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    try:
        # A: the original script with the reduced constants.
        ref = os.path.join(work, "reference")
        blob = subprocess.run(["git", "-C", REPO, "archive", "--format=zip", REFERENCE_COMMIT],
                              capture_output=True, check=True).stdout
        zipfile.ZipFile(io.BytesIO(blob)).extractall(ref)
        src = open(os.path.join(ref, "kaggle_v7.py"), encoding="utf-8").read()
        for k, v in SMALL.items():
            src, n = re.subn(rf"^{k}(\s*)=\s*[^#\n]+", rf"{k}\g<1>= {v} ", src, count=1, flags=re.M)
            assert n == 1, k
        legacy = os.path.join(work, "kaggle_v7_small.py")
        open(legacy, "w", encoding="utf-8").write(src)
        out_a = os.path.join(work, "A")
        rc = run([PY, legacy], dict(env, GLO_V7_REPO=ref, GLO_V7_OUT=out_a, GLO_V7_DATA=args.data_dir))
        check("A: original kaggle_v7.py runs", rc == 0)

        # B: the structured trainer, uninterrupted.
        ov = sum((["--override", o] for o in OVERRIDES), [])
        rc = run([PY, "train.py", "--config", "configs/glo_nca_cascade.yaml", "--data-root", args.data_dir,
                  "--output", work, "--experiment-id", "B", *ov], env)
        check("B: structured trainer runs", rc == 0)

        # C: paused after epoch 1, then resumed.
        rc1 = run([PY, "train.py", "--config", "configs/glo_nca_cascade.yaml", "--data-root", args.data_dir,
                   "--output", work, "--experiment-id", "C", "--stop-after-epoch", "1", *ov], env)
        st = json.load(open(os.path.join(work, "C", "status.json")))
        check("C: pauses after epoch 1, test not evaluated",
              rc1 == 0 and st["state"] == "paused" and st["test_evaluated"] is False, st["state"])
        rc2 = run([PY, "train.py", "--resume", os.path.join(work, "C"), "--data-root", args.data_dir], env)
        log = open(os.path.join(work, "C", "train.log"), encoding="utf-8").read()
        check("C: resume continues at epoch 2 and completes",
              rc2 == 0 and "RESUME: continuing after epoch 1" in log and "ep 2/2" in log)

        ja = json.load(open(os.path.join(out_a, "v7_results.json")))   # output name used by the original script
        for label in ("B", "C"):
            jb = json.load(open(os.path.join(work, label, "results.json")))
            d_hist = max_diff(ja["history"], jb["history"])
            d_test = max_diff(ja["test"], jb["test"])
            check(f"{label}: per-epoch loss / LR / val Dice match original", d_hist <= args.tol,
                  f"max diff {d_hist:.2e}")
            check(f"{label}: best epoch matches original", ja["best_epoch"] == jb["best_epoch"],
                  f"{ja['best_epoch']} vs {jb['best_epoch']}")
            check(f"{label}: final test (ensemble+TTA) matches original", d_test <= args.tol,
                  f"max diff {d_test:.2e}")
            check(f"{label}: parameter count matches original", ja["params"] == jb["params"],
                  str(jb["params"]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n  {sum(R)}/{len(R)} passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
