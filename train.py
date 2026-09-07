r"""
================================================================================
GLO-NCA V2 -- training entry point (config-driven, reproducible, resumable)
================================================================================
Global Context-Aware Neural Cellular Automata for multi-modal (BraTS) brain
tumour segmentation. This CLI is a thin wrapper: all training logic lives in
``src/experiment/`` and the GLO-NCA V2 methodology (model, dataset, loss, EMA,
grad-clip, cosine LR, smoothed best-epoch, val-only threshold tuning, single-pass
evaluation) is UNCHANGED from the established recipe.

Every run creates a fresh, self-contained experiment directory under
``experiments/`` (never overwriting a previous run) containing checkpoints,
metrics CSVs, graphs, TensorBoard logs, a training log, the exact config, the
patient split, the environment capture, a status file and a manifest.

Usage:
    python train.py --config configs/smoke_test.yaml
    python train.py --config configs/gcp_full.yaml
    python train.py --resume experiments/GLO-NCA-V2-YYYYMMDD-HHMMSS
    python train.py --config configs/gcp_full.yaml --device cpu

Legacy note: the old env-variable interface (EPOCHS/PATCH/AUG_LEVEL/...) is
replaced by YAML configs so every run has an exact, saved record. The kaggle_v*
scripts remain untouched as experiment history.
================================================================================
"""
import argparse
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_args():
    ap = argparse.ArgumentParser(
        description="GLO-NCA V2 training (config-driven, reproducible).")
    ap.add_argument("--config", default=os.path.join("configs", "gcp_full.yaml"),
                    help="path to a YAML config (default: configs/gcp_full.yaml)")
    ap.add_argument("--resume", metavar="EXPERIMENT_DIR", default=None,
                    help="resume an existing experiment directory")
    ap.add_argument("--output", default=os.path.join(_HERE, "experiments"),
                    help="base directory for experiments (default: ./experiments)")
    ap.add_argument("--experiment", default=None,
                    help="override the experiment name (else from config)")
    ap.add_argument("--device", default=None,
                    help="force a device, e.g. 'cpu' or 'cuda:0' (else auto)")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    from src.experiment.config import load_config
    from src.experiment.workspace import Workspace
    from src.experiment import runner

    if args.resume:
        ws = Workspace.open_existing(args.resume)
        cfg_path = ws.path("config", "config.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = args.config  # fall back to the given config
        cfg = load_config(cfg_path)
        resume = True
    else:
        cfg = load_config(args.config)
        name = args.experiment or cfg.name
        ws = Workspace.create(args.output, name)
        resume = False

    try:
        result = runner.run(cfg, ws, resume=resume, device_str=args.device)
    except KeyboardInterrupt:
        ws.write_status("failed", error="interrupted by user")
        print("\nInterrupted -- status set to failed; last.pth is on disk to resume.")
        return 130
    except Exception as exc:  # top-level guard: record, then exit non-zero
        with open(ws.path("reports", "failure_report.txt"), "a", encoding="utf-8") as fh:
            fh.write("\nUNCAUGHT EXCEPTION\n" + "=" * 40 + "\n")
            fh.write(f"{type(exc).__name__}: {exc}\n\n")
            fh.write(traceback.format_exc())
        ws.write_status("failed", error=f"{type(exc).__name__}: {exc}")
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print(f"See {ws.path('reports', 'failure_report.txt')} and "
              f"{ws.path('logs', 'training.log')}")
        return 1

    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
