#!/usr/bin/env python
r"""GCP preflight: inexpensive checks BEFORE expensive training. Does NOT train.
Exit 0 only if all CRITICAL checks pass; non-zero otherwise.

Usage:
    python scripts/preflight_gcp.py --data-root "$VM_DATA_DIR" --split split/master_split.json
"""
import argparse
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RESULTS = []


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"{status:<4} {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default=os.path.join("split", "master_split.json"))
    args = ap.parse_args()

    critical_fail = False

    # --- Python / PyTorch / CUDA / GPU ---
    record("Python", "PASS", sys.version.split()[0])
    try:
        import torch
        record("PyTorch", "PASS", torch.__version__)
        if torch.cuda.is_available():
            record("CUDA", "PASS", getattr(torch.version, "cuda", "?"))
            p = torch.cuda.get_device_properties(0)
            record("GPU", "PASS", f"{p.name} {round(p.total_memory/1e9,1)}GB")
        else:
            record("CUDA/GPU", "FAIL", "CUDA GPU unavailable (required for training)")
            critical_fail = True
    except Exception as exc:
        record("PyTorch", "FAIL", str(exc)); critical_fail = True

    # --- dataset + validation ---
    from src.experiment import datasource
    root = datasource.resolve_data_root(args.data_root)
    if root:
        record("Dataset exists", "PASS", root)
        try:
            from src.experiment.dataset_validation import validate_dataset
            rep = validate_dataset(root, limit=None)
            if rep["result"] == "PASS":
                record("Dataset validation", "PASS", f"{rep['patient_count']} cases")
            else:
                record("Dataset validation", "FAIL",
                       f"{rep['failed_patients']} bad cases"); critical_fail = True
        except Exception as exc:
            record("Dataset validation", "FAIL", str(exc)); critical_fail = True
    else:
        record("Dataset exists", "FAIL", "not found"); critical_fail = True

    # --- master split exists + integrity ---
    split_path = args.split if os.path.exists(args.split) else os.path.join(_REPO, args.split)
    if os.path.exists(split_path):
        try:
            m = datasource.load_master_split(split_path)  # verifies fingerprint
            record("Master split", "PASS",
                   f"fp {m['split_sha256'][:12]} (train {m['train_count']}/"
                   f"val {m['val_count']}/test {m['test_count']})")
            if root:
                pop = set(datasource.list_patients(root))
                union = set(m["train"]) | set(m["validation"]) | set(m["test"])
                if union == pop:
                    record("Split covers dataset", "PASS", f"{len(pop)} patients")
                else:
                    record("Split covers dataset", "FAIL",
                           f"union {len(union)} != dataset {len(pop)}"); critical_fail = True
        except Exception as exc:
            record("Master split", "FAIL", str(exc)); critical_fail = True
    else:
        record("Master split", "FAIL",
               f"{args.split} missing -- run create_master_split.py"); critical_fail = True

    # --- config integrity (reuse verify_phase3_ready helpers) ---
    try:
        from scripts.verify_phase3_ready import check_ablation_matrix, check_final_config
        ok_abl, msg_abl = check_ablation_matrix(_REPO)
        record("Ablation configs", "PASS" if ok_abl else "FAIL", msg_abl)
        critical_fail = critical_fail or not ok_abl
        ok_fin, msg_fin = check_final_config(_REPO)
        record("Final config", "PASS" if ok_fin else "FAIL", msg_fin)
        critical_fail = critical_fail or not ok_fin
    except Exception as exc:
        record("Config integrity", "FAIL", str(exc)); critical_fail = True

    # --- git commit ---
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=_REPO, text=True).strip()
        record("Git commit", "PASS", commit[:12])
    except Exception:
        record("Git commit", "WARN", "not a git repo / git unavailable")

    # --- docker (only meaningful on the VM) ---
    if shutil.which("docker"):
        record("Docker", "PASS", "present")
    else:
        record("Docker", "WARN", "not found (fine if not on the VM)")

    # --- disk + output writable ---
    if root:
        free_gb = shutil.disk_usage(root).free / 1e9
        record("Disk", "PASS" if free_gb >= 20 else "WARN", f"{free_gb:.0f}GB free")
    out_dir = os.environ.get("OUT_DIR", os.path.join(_REPO, "experiments"))
    try:
        os.makedirs(out_dir, exist_ok=True)
        record("Output writable", "PASS", out_dir)
    except Exception as exc:
        record("Output writable", "FAIL", str(exc)); critical_fail = True

    print("\nGLO-NCA GCP PREFLIGHT\n" + "-" * 30)
    for name, status, _ in RESULTS:
        print(f"{name}: {status}")
    verdict = "FAIL" if critical_fail else "PASS"
    print(f"\nPREFLIGHT: {verdict}")
    return 1 if critical_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
