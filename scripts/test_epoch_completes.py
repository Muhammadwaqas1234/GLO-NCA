#!/usr/bin/env python
r"""
Prove a real epoch completes on real data through the PRODUCTION RUNNER.

Phase 1 verified components; it never executed runner.run() end to end. Two
defects survived that gap and only appeared on a rented GPU: the diagnostics
module was shadowed by a local variable, and Data_Container cached every case
in RAM until a DataLoader worker was OOM-killed.

This runs the actual runner on a small slice of the canonical split, with the
production architecture and preprocessing, and watches resident memory while
it does. It is the gate that should precede any cloud spend.

Usage:
  python scripts/test_epoch_completes.py --data-root DIR [--cases 12] [--epochs 1]
Exit:
  0 if an epoch completed with bounded memory and finite losses.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ROOT = (r"C:\Users\raiwa\Downloads\MICCAI-LH-BraTS2025-MET-Challenge-Training")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")


class MemoryWatch(threading.Thread):
    """Sample this process's RSS while the runner works."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mb = 0.0
        self._halt = threading.Event()

    def run(self):
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            return
        while not self._halt.is_set():
            try:
                rss = proc.memory_info().rss / 2 ** 20
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss / 2 ** 20
                    except Exception:
                        pass
                self.peak_mb = max(self.peak_mb, rss)
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    print("=" * 74)
    print("REAL EPOCH THROUGH THE PRODUCTION RUNNER")
    print("=" * 74)

    if not os.path.isdir(args.data_root):
        print(f"\n  BLOCKED: dataset not found at {args.data_root}")
        return 1

    from src.datasets.Data_Instance import Data_Container
    cap = Data_Container().max_entries
    check("RAM cache is bounded", cap > 0 and cap < 100, f"cap {cap} cases")

    per_case_gb = (128 ** 3 * 4 * 4 + 128 ** 3 * 3 * 4) / 2 ** 30
    check("bounded cache fits in host RAM",
          per_case_gb * cap < 4.0,
          f"{per_case_gb * cap:.2f} GB/worker (unbounded 898 = "
          f"{per_case_gb * 898:.0f} GB)")

    # A small config: production architecture, a slice of the canonical split.
    from src.experiment.config import load_config
    base = load_config(os.path.join(_HERE, "configs", "glo_nca_production.yaml"))
    raw = json.loads(json.dumps(base.raw))
    raw["training"]["epochs"] = args.epochs
    raw["training"]["workers"] = 0          # surface errors in-process
    raw["dataset"]["number_of_patients"] = args.cases
    raw["experiment"]["name"] = "epoch-gate"
    raw["data"].pop("split_file", None)
    raw["data"]["allow_seeded_split"] = True
    # The data-quality policy names two specific cases and fails closed when a
    # subset excludes them. That guard is correct; it just does not apply to a
    # slice, so the gate runs without it.
    raw["data"]["quality_policy_file"] = None   # explicit opt-out for a slice
    raw["data"]["cache"] = {"enabled": False}
    raw["performance"] = dict(raw.get("performance") or {})
    raw["performance"]["precision"] = "fp32"
    raw["evaluation"]["tune_thresholds"] = False

    os.environ["DATA_ROOT"] = args.data_root
    tmp = tempfile.mkdtemp(prefix="glo_epoch_gate_")
    cfg_path = os.path.join(tmp, "epoch_gate.yaml")
    import yaml
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    from src.experiment import runner
    from src.experiment.config import load_config as lc
    from src.experiment.workspace import Workspace

    cfg = lc(cfg_path)
    ws = Workspace.create(os.path.join(tmp, "experiments"), "epoch-gate")

    watch = MemoryWatch()
    watch.start()
    t0 = time.perf_counter()
    status, err = "unknown", ""
    try:
        result = runner.run(cfg, ws, resume=False, device_str=None)
        status = str(result.get("status", "unknown"))
    except Exception as exc:
        status, err = "exception", f"{type(exc).__name__}: {exc}"
    finally:
        watch.stop()
        watch.join(timeout=5)
    elapsed = time.perf_counter() - t0

    check("runner.run() completed", status == "completed",
          status + (f" -- {err}" if err else ""))
    if watch.peak_mb:
        check("peak RSS stayed bounded", watch.peak_mb < 12000,
              f"{watch.peak_mb:.0f} MB")

    hist_path = ws.path("metrics", "train.csv")
    if os.path.isfile(hist_path):
        import csv
        rows = list(csv.DictReader(open(hist_path, encoding="utf-8")))
        check("epoch rows written", len(rows) >= args.epochs,
              f"{len(rows)} epoch(s)")
        if rows:
            losses = [float(r["loss"]) for r in rows if r.get("loss")]
            finite = all(l == l and abs(l) != float("inf") for l in losses)
            check("training losses finite", finite,
                  ", ".join(f"{l:.4f}" for l in losses[:3]))
    else:
        check("epoch rows written", False, "metrics/train.csv missing")

    for artifact in ("reports/validation_per_case.csv",):
        p = ws.path(*artifact.split("/"))
        check(f"artifact {os.path.basename(artifact)}", os.path.isfile(p),
              "written" if os.path.isfile(p) else "missing")

    print()
    print(f"  elapsed {elapsed / 60:.1f} min for {args.epochs} epoch(s) "
          f"on {args.cases} cases")

    shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
