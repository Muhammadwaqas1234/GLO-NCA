#!/usr/bin/env python
r"""Choose the resume source: the local run directory or a staged copy from GCS.

An older checkpoint never replaces a newer one. For each side the newest valid
full checkpoint is found (last.pth, else the newest periodic one) and its
recorded epoch compared; the higher epoch wins and a tie keeps the local copy.
The whole directory is taken from one side, so files are never mixed.

Prints the reasoning and a final ``SOURCE=local|gcs`` line; exits 1 when
neither side holds a valid checkpoint.

Usage:
    python cloud/scripts/select_resume_source.py <local_experiment_dir> <gcs_staged_dir>
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from validate_checkpoint import newest_valid  # noqa: E402


def describe(label, experiment_dir):
    if not os.path.isdir(os.path.join(experiment_dir, "checkpoints")):
        print(f"{label}: no checkpoints")
        return None
    path, epoch, invalid = newest_valid(experiment_dir)
    for cand, detail in invalid:
        print(f"{label}: invalid {cand} -> {detail}")
    if path is None:
        print(f"{label}: no valid checkpoint")
        return None
    print(f"{label}: {path} (epoch {epoch})")
    return epoch


def choose(local_epoch, gcs_epoch):
    """'local', 'gcs' or None; ties keep local."""
    if local_epoch is None and gcs_epoch is None:
        return None
    if gcs_epoch is None:
        return "local"
    if local_epoch is None:
        return "gcs"
    return "gcs" if gcs_epoch > local_epoch else "local"


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    local_epoch = describe("local", argv[1])
    gcs_epoch = describe("gcs", argv[2])
    source = choose(local_epoch, gcs_epoch)
    if source is None:
        print("FAIL: neither the local run nor GCS holds a valid checkpoint.")
        return 1
    reason = {"local": "local is newer or equal" if gcs_epoch is not None else "only local is valid",
              "gcs": "GCS is newer" if local_epoch is not None else "only GCS is valid"}[source]
    print(f"decision: resume from {source} ({reason})")
    print(f"SOURCE={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
