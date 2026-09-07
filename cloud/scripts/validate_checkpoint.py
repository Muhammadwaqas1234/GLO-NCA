#!/usr/bin/env python
r"""Validate a GLO-NCA checkpoint before resuming (infrastructure-only; does not
touch the model). Reuses the Phase 1 loader. Exit 0 if usable, 1 otherwise.

Usage:
    python cloud/scripts/validate_checkpoint.py <experiment_dir>
Prints the best usable checkpoint path on success (last.pth preferred, else the
newest periodic), or an error and the available alternatives on failure.
"""
import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load(path):
    from src.experiment.checkpoint import load_checkpoint
    return load_checkpoint(path, map_location="cpu")


def _is_valid_full(path):
    """A full (last/periodic) checkpoint must carry the state needed to resume."""
    try:
        ck = _load(path)
    except Exception as exc:
        return False, f"unreadable: {exc}"
    required = ["epoch", "model", "optimizer", "scheduler", "config"]
    missing = [k for k in required if k not in ck]
    if missing:
        return False, f"missing keys: {missing}"
    if not ck["model"]:
        return False, "empty model state"
    return True, f"epoch={ck['epoch']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"FAIL: no checkpoints/ under {args.experiment_dir}")
        return 1

    last = os.path.join(ckpt_dir, "last.pth")
    periodic = sorted(glob.glob(os.path.join(ckpt_dir, "periodic", "epoch_*.pth")),
                      reverse=True)

    # Prefer last.pth; never replace a valid checkpoint with an older one.
    candidates = ([last] if os.path.exists(last) else []) + periodic
    if not candidates:
        print(f"FAIL: no last.pth or periodic checkpoints in {ckpt_dir}")
        return 1

    for cand in candidates:
        ok, detail = _is_valid_full(cand)
        if ok:
            print(f"OK: {cand} ({detail})")
            print(f"RESUME_CHECKPOINT={cand}")
            if cand != last:
                print(f"NOTE: last.pth invalid/missing; using newest valid "
                      f"periodic checkpoint instead. Corrupt file left in place.")
            return 0
        else:
            print(f"invalid: {cand} -> {detail}")

    print("FAIL: no valid checkpoint found. Corrupt files were NOT modified.")
    print("Available (all invalid):")
    for c in candidates:
        print(f"  {c}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
