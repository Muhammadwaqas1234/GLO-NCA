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

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
HAS_GPU = torch.cuda.is_available() and torch.cuda.device_count() > 0
REFERENCE_COMMIT = "kaggle-v7-reference"   # tag: original script and library
SMALL = {"EPOCHS": 2, "N_PATIENTS": 8, "ENSEMBLE_N": 2, "NUM_WORKERS": 0}
OVERRIDES = ["training.epochs=2", "data.n_patients=8", "evaluation.ensemble_n=2",
             "training.workers=0"]
R = []


def check(name, ok, detail=""):
    r"""Record and print one check result."""
    R.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}", flush=True)


def max_diff(a, b, path=""):
    r"""Largest absolute difference between two nested JSON structures, with its location."""
    if isinstance(a, dict):
        return max([max_diff(a[k], b[k], f"{path}.{k}") for k in a] or [(0.0, path)])
    if isinstance(a, list):
        return max([max_diff(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))] or [(0.0, path)])
    if isinstance(a, (int, float)):
        if a != a and b != b:        # both NaN
            return 0.0, path
        return abs(float(a) - float(b)), path
    return (0.0 if a == b else float("inf")), path


def run(cmd, env, cwd=REPO):
    r"""Run a command and return its exit code."""
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
    return p.returncode


def main():
    r"""Run the original script and the trainer, then compare their outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--dice-tol", type=float, default=None,
                    help="tolerance for Dice/IoU/HD95 (default: --tol on CPU, 1e-3 on GPU, where a voxel "
                         "at the 0.5 threshold can flip between runs)")
    args = ap.parse_args()
    dice_tol = args.dice_tol if args.dice_tol is not None else (1e-3 if HAS_GPU else args.tol)
    work = tempfile.mkdtemp(prefix="v7eq_")
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", GLO_DETERMINISTIC="1")
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
        if not HAS_GPU:
            # CPU-only machine (e.g. CI): the original script hard-codes cuda:0.
            src = src.replace('"device": "cuda:0"', '"device": "cpu"')
            src = src.replace('torch.device("cuda:0" if torch.cuda.is_available() else "cpu")', 'torch.device("cpu")')
        # Deterministic cuDNN kernels in both programs, so GPU runs compare exactly.
        src = src.replace("\nimport torch\n", "\nimport torch\ntorch.backends.cudnn.deterministic = True\n"
                          "torch.backends.cudnn.benchmark = False\n", 1)
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
            train_keys = ("epoch", "loss", "lr")
            d_train, p_train = max_diff({k: ja["history"][k] for k in train_keys},
                                        {k: jb["history"][k] for k in train_keys})
            d_val, p_val = max_diff({k: v for k, v in ja["history"].items() if k not in train_keys},
                                    {k: v for k, v in jb["history"].items() if k not in train_keys})
            d_test, p_test = max_diff(ja["test"], jb["test"])
            check(f"{label}: per-epoch loss / LR match original", d_train <= args.tol,
                  f"max diff {d_train:.2e} {p_train}")
            check(f"{label}: per-epoch validation Dice matches original", d_val <= dice_tol,
                  f"max diff {d_val:.2e} {p_val}")
            check(f"{label}: best epoch matches original", ja["best_epoch"] == jb["best_epoch"],
                  f"{ja['best_epoch']} vs {jb['best_epoch']}")
            check(f"{label}: final test (ensemble+TTA) matches original", d_test <= dice_tol,
                  f"max diff {d_test:.2e} {p_test}")
            check(f"{label}: parameter count matches original", ja["params"] == jb["params"],
                  str(jb["params"]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n  {sum(R)}/{len(R)} passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
