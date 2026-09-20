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

    # --- explicit continuation past the planned budget ----------------------
    # `--resume` continues an UNFINISHED run toward its original
    # `training.epochs`. `--extend-to` continues a run that already REACHED
    # that budget, and records the continuation so the two are never confused
    # in the thesis record.
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
                         "'continue' steps the saved cosine onward, which "
                         "RAISES the LR past T_max (a warm restart). "
                         "'rebuild' re-fits the cosine to the new horizon, "
                         "retroactively changing the original curve.")
    ap.add_argument("--extension-reason", default=None,
                    help="why the run is being continued. Recorded in "
                         "extension.json for the thesis record. Required with "
                         "--extend-to.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    # Phase 5 (B-01): validate the CLI contract BEFORE importing torch. These two
    # guards are pure argument checks that need no heavy module, but they used to
    # sit after `import runner` (which pulls in torch), so a misinvocation paid a
    # ~210 s import before being told it was invalid -- and a bare `train.py`
    # could not be checked under a sane timeout. Same exit code (2), same
    # message, same conditions; only the import is deferred past them. Nothing on
    # the successful training path is reordered.
    if not args.resume and not args.config:
        print("FAILED: --config is required (or use --resume <dir>).\n"
              "  production V3: python train.py --config configs/v3_multilevel_ckpt.yaml")
        return 2
    # Final audit (B-02): a mistyped --config path previously paid the ~13 s torch
    # import and then died with a raw FileNotFoundError traceback (rc=1). A wrong
    # path is the most likely production invocation mistake, so check it here --
    # before the heavy imports -- and report it the same way as the other guards.
    if args.config and not os.path.isfile(args.config):
        print(f"FAILED: config not found: {args.config}\n"
              "  production V3: python train.py --config configs/v3_multilevel_ckpt.yaml")
        return 2
    if args.resume:
        _cfg_probe = os.path.join(args.resume, "config", "config.yaml")
        if not os.path.isdir(args.resume) or not os.path.exists(_cfg_probe):
            print(f"FAILED: cannot resume {args.resume}: missing {_cfg_probe}.\n"
                  "The experiment's own config is required to resume "
                  "reproducibly; refusing to substitute a different config.")
            return 2

    # --- continuation contract (checked before the heavy imports) -----------
    # An extension continues a specific completed run, so it needs that run's
    # directory and a stated justification. Both are cheap argument checks.
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
        # Final audit (B-03): an invalid --device previously failed only once the
        # runner tried to use it -- AFTER Workspace.create() had already made a
        # timestamped experiment directory, leaving an empty junk directory in
        # experiments/ for every typo. Validate the string first; this only parses
        # the device and does not select, initialise or allocate on it, so the
        # normal training path is unaffected.
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
                            extension_reason=args.extension_reason)
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
