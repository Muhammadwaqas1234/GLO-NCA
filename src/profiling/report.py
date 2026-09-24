r"""Human-readable profiling report: sections ranked by measured mean with percent-of-iteration."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _is_timing(name: str) -> bool:
    """Timing sections only; memory gauges (MB) are reported separately."""
    return not name.startswith(("memory/", "data/file_size_mb"))


def _table(stats: Dict[str, Dict[str, Any]], total_key: Optional[str]) -> str:
    total_mean = None
    if total_key and total_key in stats:
        total_mean = stats[total_key]["mean_ms"]

    rows = [(n, s) for n, s in stats.items()
            if n != total_key and _is_timing(n)]
    rows.sort(key=lambda kv: kv[1]["mean_ms"], reverse=True)

    out = ["| Component | Mean (ms) | Median | Min | Max | p95 | % iteration |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in rows:
        # data/* stages run inside the loader wait (often in workers), outside the measured
        # iteration, so no percent-of-iteration is shown for them.
        if name.startswith("data/") and name != "data/dataloader_wait":
            pct = "input pipeline*"
        elif name == "data/dataloader_wait":
            pct = (f"{100.0 * s['mean_ms'] / total_mean:.1f}%†"
                   if total_mean else "n/a")
        else:
            pct = (f"{100.0 * s['mean_ms'] / total_mean:.1f}%"
                   if total_mean else "n/a")
        p95 = f"{s['p95_ms']:.2f}" if s.get("p95_ms") is not None else "n/a"
        out.append(f"| `{name}` | {s['mean_ms']:.2f} | {s['median_ms']:.2f} | "
                   f"{s['min_ms']:.2f} | {s['max_ms']:.2f} | {p95} | {pct} |")
    if total_mean:
        t = stats[total_key]
        out.append(f"| **`{total_key}`** | **{t['mean_ms']:.2f}** | "
                   f"{t['median_ms']:.2f} | {t['min_ms']:.2f} | "
                   f"{t['max_ms']:.2f} | "
                   f"{t['p95_ms'] if t.get('p95_ms') is not None else 'n/a'} | "
                   "**100%** |")
    return "\n".join(out)


def _contributors(stats: Dict[str, Dict[str, Any]], total_key: Optional[str],
                  top: int = 5) -> str:
    rows = [(n, s["mean_ms"]) for n, s in stats.items()
            if n != total_key and _is_timing(n)]
    if not rows:
        return "_No measurements recorded._"
    rows.sort(key=lambda kv: kv[1], reverse=True)
    total_mean = stats.get(total_key, {}).get("mean_ms") if total_key else None
    lines = []
    for i, (name, mean) in enumerate(rows[:top], 1):
        pct = f" ({100.0 * mean / total_mean:.1f}% of iteration)" if total_mean else ""
        lines.append(f"{i}. `{name}` — {mean:.2f} ms{pct}")
    return "\n".join(lines)


def write_reports(profiler, out_dir: str, *, environment: Dict[str, Any],
                  configuration: Dict[str, Any],
                  memory: Optional[Dict[str, Any]] = None,
                  validation: Optional[Dict[str, Any]] = None,
                  notes: Optional[str] = None,
                  total_key: str = "total/iteration") -> str:
    """Write ``reports/profiling_report.md`` under ``out_dir`` and return its path."""
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    stats = profiler.statistics() if hasattr(profiler, "statistics") else {}

    p = ["# GLO-NCA Phase 2 — profiling report",
         "",
         "**Observational only.** No architecture, resolution, NCA-step, loss, "
         "optimizer, EMA, split, threshold or checkpoint-policy change was made "
         "to produce these numbers.",
         "",
         "## Environment", "",
         "```", json.dumps(environment, indent=2), "```", "",
         "## Configuration profiled", "",
         "```", json.dumps(configuration, indent=2), "```", ""]

    if notes:
        p += ["## Scope / limitations", "", notes, ""]

    p += ["## Training iteration — measured components", "",
          _table(stats, total_key) if stats else "_No samples collected._", "",
          r"\* `data/*` stages are measured inside the input pipeline, which "
          "runs BEFORE/alongside the timed iteration (and, with `workers > 0`, "
          "in separate processes). They are not a subset of "
          "`total/iteration`, so no percentage of it is computed for them.",
          "",
          "† `data/dataloader_wait` IS inside the timed loop: it is the time the "
          "training loop sat blocked waiting for a batch.", "",
          "### Largest measured contributors", "",
          _contributors(stats, total_key), "",
          "_Ranking is by measured mean. No component is called 'the "
          "bottleneck' beyond what these measurements show._", ""]

    oob = (profiler.out_of_band_statistics()
           if hasattr(profiler, "out_of_band_statistics") else {})
    if oob:
        p += ["## Validation / out-of-iteration sections", "",
              "Validation runs AFTER the training loop, so it is measured "
              "separately -- this is what distinguishes a slow epoch caused by "
              "training from one caused by full-volume validation.", "",
              "| Section | Count | Mean (ms) | Median | Min | Max | Total (ms) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for n, v in sorted(oob.items(), key=lambda kv: -kv[1]["total_ms"]):
            p += [f"| `{n}` | {v['count']} | {v['mean_ms']:.2f} | "
                  f"{v['median_ms']:.2f} | {v['min_ms']:.2f} | "
                  f"{v['max_ms']:.2f} | {v['total_ms']:.2f} |"]
        p += [""]

    if validation:
        p += ["## Validation — measured separately", "",
              "Validation is profiled on its own path so a slow epoch can be "
              "attributed to training vs validation rather than guessed.", "",
              "```", json.dumps(validation, indent=2), "```", ""]

    gauges = {n: v for n, v in stats.items() if not _is_timing(n)}
    if gauges:
        p += ["## Memory gauges sampled per iteration (MB, not milliseconds)", "",
              "| Gauge | Mean (MB) | Min | Max |", "|---|---:|---:|---:|"]
        for n, v in sorted(gauges.items()):
            p += [f"| `{n}` | {v['mean_ms']:.1f} | {v['min_ms']:.1f} | "
                  f"{v['max_ms']:.1f} |"]
        p += ["", "_These are gauges, not durations; no '% of iteration' is "
              "computed for them._", ""]

    if memory:
        p += ["## Memory", "",
              "Host RSS, system RAM, GPU allocated, GPU reserved and GPU peak "
              "are reported as distinct quantities and never mixed.", "",
              "```", json.dumps(memory, indent=2), "```", ""]

    p += ["## Potential optimizations — NOT IMPLEMENTED", "",
          "Phase 2 measures only. Any opportunity visible in the table above is "
          "recorded for a later, separately approved phase; nothing was changed "
          "or tuned here.", ""]

    path = os.path.join(out_dir, "reports", "profiling_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(p))
    return path
