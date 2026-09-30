#!/usr/bin/env python
r"""GLO-NCA cascade training entry point (new run, resume, or pause after an epoch)."""
import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class _Tee:
    r"""Mirror stdout into <run>/train.log."""

    def __init__(self, path):
        r"""Open the log file next to stdout."""
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._out = sys.stdout

    def write(self, s):
        r"""Write to stdout and the log."""
        self._out.write(s)
        self._file.write(s)

    def flush(self):
        r"""Flush stdout and the log."""
        self._out.flush()
        self._file.flush()


def main() -> int:
    r"""Parse arguments and start, resume or pause a training run."""
    ap = argparse.ArgumentParser(description="GLO-NCA cascade training (Kaggle v7 recipe).")
    ap.add_argument("--config", default=None, help="YAML config, e.g. configs/glo_nca_cascade.yaml")
    ap.add_argument("--resume", metavar="RUN_DIR", default=None,
                    help="continue a run from its last.pth and saved config")
    ap.add_argument("--data-root", default=None,
                    help="folder searched for the BraTS case folders (default: config data.base)")
    ap.add_argument("--output", default=os.path.join(_HERE, "experiments"),
                    help="base folder for new runs (default: ./experiments)")
    ap.add_argument("--experiment-id", default=None,
                    help="exact run folder name (default: <name>-<timestamp>)")
    ap.add_argument("--stop-after-epoch", type=int, default=None, metavar="EPOCH",
                    help="pause after this epoch (full checkpoint, no final test)")
    ap.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config value (repeatable), e.g. training.epochs=2")
    args = ap.parse_args()

    if bool(args.config) == bool(args.resume):
        print("FAILED: give exactly one of --config or --resume.")
        return 2
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        print(f"FAILED: --stop-after-epoch {args.stop_after_epoch} is not a valid epoch.")
        return 2

    import yaml
    from glo_nca.config import load_config
    from glo_nca.trainer import run

    if args.resume:
        run_dir = os.path.abspath(args.resume)
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            print(f"FAILED: cannot resume {run_dir}: missing {cfg_path}.")
            return 2
        if args.override:
            print("FAILED: --override is not allowed with --resume (the run keeps its config).")
            return 2
        C = load_config(cfg_path)
    else:
        if not os.path.isfile(args.config):
            print(f"FAILED: config not found: {args.config}")
            return 2
        C = load_config(args.config, args.override)
        run_id = args.experiment_id or f"{C.NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = os.path.join(os.path.abspath(args.output), run_id)
        if os.path.exists(os.path.join(run_dir, "last.pth")):
            print(f"FAILED: {run_dir} already holds a run; use --resume.")
            return 2
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump(C.raw, fh, sort_keys=False)

    sys.stdout = _Tee(os.path.join(run_dir, "train.log"))
    print(f"run: {run_dir}")
    try:
        result = run(C, run_dir, data_root=args.data_root, resume=bool(args.resume),
                     stop_after_epoch=args.stop_after_epoch)
    except Exception as exc:
        traceback.print_exc(file=sys.stdout)
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0 if result.get("status") in ("completed", "paused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
