r"""GLO-NCA training entry point (config-driven, reproducible, resumable).

Thin CLI wrapper; all training logic lives in ``src/experiment/``. Each run
writes a fresh experiment directory under ``experiments/``.

Usage:
    # Production run.
    python train.py --config configs/glo_nca_production.yaml

    # Resume an interrupted run (e.g. after a Spot preemption).
    python train.py --resume experiments/<run-id>

    # Continue a completed run past its planned budget.
    python train.py --resume experiments/<run-id> --extend-to 310 \
                    --extension-reason "validation still improving"

configs/glo_nca_production.yaml is the only production configuration. Verify
it before a long run:

    python scripts/verify_glo_nca_production_config.py \
        configs/glo_nca_production.yaml
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
        description="GLO-NCA training (config-driven, reproducible).")
    # No default config: a bare run must fail rather than train the wrong model.
    ap.add_argument("--config", default=None,
                    help="path to a YAML config, e.g. configs/glo_nca_production.yaml "
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

    # --resume finishes an unfinished run; --extend-to continues a completed one.
    ap.add_argument("--extend-to", metavar="EPOCH", type=int, default=None,
                    help="continue a COMPLETED run past its planned budget, "
                         "e.g. --extend-to 301. Requires --resume and a "
                         "complete parent checkpoint. Refused if any protected "
                         "identity (architecture, seed, split, loss, optimizer) "
                         "has drifted -- that is a new experiment, not a "
                         "continuation.")
    ap.add_argument("--lr-policy", choices=("freeze", "continue", "rebuild"),
                    default="freeze",
                    help="learning-rate policy for --extend-to. "
                         "'freeze' (default, production-safe) holds the LR at "
                         "eta_min, leaving the original 1..N decay intact. "
                         "'continue' steps the saved scheduler onward; the "
                         "production WarmupCosineLR holds at eta_min past the "
                         "horizon (a legacy CosineAnnealingLR would rise again). "
                         "'rebuild' re-fits the cosine to the new horizon, "
                         "retroactively changing the original curve.")
    ap.add_argument("--stop-after-epoch", metavar="EPOCH", type=int, default=None,
                    help="pause after this epoch: write its full checkpoint and exit "
                         "before any end-of-run evaluation (the test split is not "
                         "read). The planned budget and LR schedule are unchanged; "
                         "continue later with --resume.")
    ap.add_argument("--extension-reason", default=None,
                    help="why the run is being continued. Recorded in "
                         "extension.json for the thesis record. Required with "
                         "--extend-to.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    # Validate CLI arguments before importing torch (fail fast, exit code 2).
    if not args.resume and not args.config:
        print("FAILED: --config is required (or use --resume <dir>).\n"
              "  production: python train.py --config configs/glo_nca_production.yaml")
        return 2
    if args.config and not os.path.isfile(args.config):
        print(f"FAILED: config not found: {args.config}\n"
              "  production: python train.py --config configs/glo_nca_production.yaml")
        return 2
    if args.resume:
        _cfg_probe = os.path.join(args.resume, "config", "config.yaml")
        if not os.path.isdir(args.resume) or not os.path.exists(_cfg_probe):
            print(f"FAILED: cannot resume {args.resume}: missing {_cfg_probe}.\n"
                  "The experiment's own config is required to resume "
                  "reproducibly; refusing to substitute a different config.")
            return 2

    # An extension needs the parent run directory and a stated reason.
    if args.extend_to is not None:
        if not args.resume:
            print("FAILED: --extend-to requires --resume <experiment-dir>.\n"
                  "An extension continues a SPECIFIC completed run; it cannot "
                  "start from a config alone.")
            return 2
        if not args.extension_reason or not args.extension_reason.strip():
            print("FAILED: --extend-to requires --extension-reason.\n"
                  "Training past the planned budget is a deliberate departure "
                  "from the experiment plan and must be justifiable in the "
                  "thesis record.")
            return 2
        if args.extend_to < 1:
            print(f"FAILED: --extend-to {args.extend_to} is not a valid epoch.")
            return 2
        print(f"EXTENSION MODE: continuing {args.resume} to epoch "
              f"{args.extend_to} (lr-policy={args.lr_policy})")
        if args.lr_policy != "freeze":
            print(f"  WARNING: lr-policy '{args.lr_policy}' changes the "
                  f"learning-rate trajectory. 'freeze' is the production-safe "
                  f"policy; anything else is a separate experiment.")

    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        print(f"FAILED: --stop-after-epoch {args.stop_after_epoch} is not a valid epoch.")
        return 2

    from src.experiment.config import load_config
    from src.experiment.workspace import Workspace
    from src.experiment import runner

    if args.resume:
        ws = Workspace.open_existing(args.resume)
        cfg_path = ws.path("config", "config.yaml")
        if not os.path.exists(cfg_path):
            # Never fall back to --config: resume must use the run's own config.
            print(f"FAILED: cannot resume {args.resume}: missing {cfg_path}.\n"
                  "The experiment's own config is required to resume "
                  "reproducibly; refusing to substitute a different config.")
            return 2
        cfg = load_config(cfg_path)
        resume = True
    else:
        if not args.config:
            print("FAILED: --config is required (or use --resume <dir>).\n"
                  "  production: python train.py --config configs/glo_nca_production.yaml")
            return 2
        cfg = load_config(args.config)
        # Validate --device before creating the experiment directory.
        if args.device:
            try:
                import torch as _torch
                _torch.device(args.device)
            except Exception as _exc:
                print(f"FAILED: invalid --device {args.device!r}: {_exc}\n"
                      "  examples: cpu | cuda | cuda:0")
                return 2
        name = args.experiment or cfg.name
        ws = Workspace.create(args.output, name,
                              experiment_id=args.experiment_id)
        resume = False

    try:
        result = runner.run(cfg, ws, resume=resume, device_str=args.device,
                            extend_to=args.extend_to,
                            lr_policy=args.lr_policy,
                            extension_reason=args.extension_reason,
                            stop_after_epoch=args.stop_after_epoch)
    except KeyboardInterrupt:
        ws.write_status("failed", error="interrupted by user")
        print("\nInterrupted -- status set to failed; last.pth is on disk to resume.")
        return 130
    except Exception as exc:  # Record the failure, then exit non-zero.
        with open(ws.path("reports", "failure_report.txt"), "a", encoding="utf-8") as fh:
            fh.write("\nUNCAUGHT EXCEPTION\n" + "=" * 40 + "\n")
            fh.write(f"{type(exc).__name__}: {exc}\n\n")
            fh.write(traceback.format_exc())
        ws.write_status("failed", error=f"{type(exc).__name__}: {exc}")
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print(f"See {ws.path('reports', 'failure_report.txt')} and "
              f"{ws.path('logs', 'training.log')}")
        return 1

    if result.get("status") == "paused":
        print(f"PAUSED after epoch {result.get('epoch')}; continue with "
              f"--resume {ws.root}")
        return 0
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
