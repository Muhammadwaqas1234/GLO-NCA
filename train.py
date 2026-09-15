r"""
================================================================================
GLO-NCA V3 -- training entry point (config-driven, reproducible, resumable)
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
    python train.py --config configs/v3_multilevel_ckpt.yaml   # production V3
    python train.py --resume experiments/GLO-NCA-V2-YYYYMMDD-HHMMSS
    python train.py --config configs/v3_multilevel_ckpt.yaml --device cpu

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
    # Phase 2: no default config. V3 is the production architecture, and the old
    # default (configs/gcp_full.yaml) was the V2 baseline -- a bare
    # `python train.py` silently trained the wrong model. Be explicit.
    ap.add_argument("--config", default=None,
                    help="path to a YAML config, e.g. configs/v3_multilevel_ckpt.yaml "
                         "(required unless --resume is given)")
    ap.add_argument("--resume", metavar="EXPERIMENT_DIR", default=None,
                    help="resume an existing experiment directory")
    ap.add_argument("--output", default=os.path.join(_HERE, "experiments"),
                    help="base directory for experiments (default: ./experiments)")
    ap.add_argument("--experiment", default=None,
                    help="override the experiment name (else from config)")
    ap.add_argument("--device", default=None,
                    help="force a device, e.g. 'cpu' or 'cuda:0' (else auto)")
    ap.add_argument("--experiment-id", default=None,
                    help="use this EXACT experiment directory name instead of "
                         "generating '<name>-<timestamp>'. The cloud entrypoint "
                         "passes it so the GCS sync watcher knows the experiment "
                         "id up front instead of guessing the newest directory.")
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
            # Phase 2 (P0): NEVER fall back to --config here. Its default is the
            # V2 baseline (configs/gcp_full.yaml), so the old fallback could
            # resume a V3 experiment under a V2 configuration. A resume without
            # its own saved config is unreproducible -- fail loudly instead.
            print(f"FAILED: cannot resume {args.resume}: missing {cfg_path}.\n"
                  "The experiment's own config is required to resume "
                  "reproducibly; refusing to substitute a different config.")
            return 2
        cfg = load_config(cfg_path)
        resume = True
    else:
        if not args.config:
            print("FAILED: --config is required (or use --resume <dir>).\n"
                  "  production V3: python train.py --config configs/v3_multilevel_ckpt.yaml")
            return 2
        cfg = load_config(args.config)
        name = args.experiment or cfg.name
        ws = Workspace.create(args.output, name,
                              experiment_id=args.experiment_id)
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
