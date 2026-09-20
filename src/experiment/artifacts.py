r"""Experiment artifact manifest.

One writer for the artifact set an examiner or a future engineer needs to
reconstruct what a run actually did. It *composes* the sources that already
exist -- `environment`, `checkpoint`, `diagnostics`, `early_stopping`,
`state_machine`, the CSV loggers -- rather than re-deriving any of them, so
there is a single place where the answer to "what ran?" is assembled and no
second system can drift away from the first.

Two rules the module is built around.

**Nothing is invented.** A value that cannot be obtained is written as
``NOT MEASURED``, ``NOT AVAILABLE`` or ``BLOCKED``. A fabricated zero or a
placeholder timing is worse than an absent field, because it looks like
evidence.

**Artifacts must agree.** Sixteen files that each state the parameter count
are sixteen chances to disagree. :func:`audit_consistency` re-reads what was
written and fails if the identity fields diverge -- and the fix for a
mismatch is the producer, never the artifact.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import time
from typing import Any, Dict, List, Optional

NOT_MEASURED = "NOT MEASURED"
NOT_AVAILABLE = "NOT AVAILABLE"
BLOCKED = "BLOCKED"

#: The complete artifact set. Names are fixed so downstream tooling and the
#: consistency audit can rely on them.
ARTIFACTS = (
    "run_metadata.json",
    "architecture_identity.json",
    "hardware_metadata.json",
    "precision_benchmark.json",
    "timing_summary.json",
    "epoch_metrics.csv",
    "validation_history.csv",
    "checkpoint_manifest.json",
    "best_checkpoint_metadata.json",
    "top3_checkpoint_metadata.json",
    "periodic_checkpoint_metadata.json",
    "final_checkpoint_metadata.json",
    "resume_verification.json",
    "overfitting_report.json",
    "gpu_diagnostics.csv",
    "final_experiment_report.md",
)

#: Fields every artifact that mentions identity must agree on.
_IDENTITY_KEYS = ("parameters", "split_sha256", "seed", "git_commit",
                  "config_fingerprint")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


def _atomic_csv(path: str, rows: List[Dict[str, Any]],
                fields: Optional[List[str]] = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, path)
    return path


class ArtifactManager:
    """Writes and audits the artifact set for one experiment run."""

    def __init__(self, directory: str, *, run_id: str = ""):
        self.directory = directory
        self.run_id = run_id or os.path.basename(os.path.abspath(directory))
        os.makedirs(directory, exist_ok=True)
        self.written: Dict[str, str] = {}

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def _done(self, name: str, p: str) -> str:
        self.written[name] = p
        return p

    # ----------------------------------------------------------- identity
    def architecture_identity(self, *, model, config: Dict[str, Any],
                              split: Dict[str, Any],
                              git: Optional[Dict[str, Any]] = None,
                              config_fingerprint: str = "") -> str:
        """Derive the identity FROM THE BUILT MODEL, not from documentation.

        Reading the live module is the point: a config can say one thing while
        the constructed network does another, and only the network trains.
        """
        m_cfg = config.get("model", {}) or {}
        levels = [{
            "index": i + 1,
            "resolution": lv.resolution,
            "channels": lv.channels,
            "nca_steps": lv.nca_steps,
            "perception_kernel": lv.kernel_size,
        } for i, lv in enumerate(model.levels)]
        gc = getattr(model.ncas[0], "gc", None)
        payload = {
            "generated_utc": _utc(),
            "run_id": self.run_id,
            "parameters": sum(p.numel() for p in model.parameters()),
            "working_volume": int(((config.get("data", {}) or {})
                                   .get("training_patch", {}) or {})
                                  .get("working_volume", 96)),
            "levels": levels,
            "level3_absent": len(model.levels) == 2,
            "total_nca_steps": sum(lv.nca_steps for lv in model.levels),
            "spatial_gc_kernel": (gc.conv.kernel_size[0]
                                  if gc is not None else NOT_AVAILABLE),
            "se_enabled": all(n.se is not None for n in model.ncas),
            "spatial_gc_enabled": all(n.gc is not None for n in model.ncas),
            "fusion": {"in_channels": model.fuse.in_channels,
                       "out_channels": model.fuse.out_channels,
                       "learned": True},
            "global_context_source": "full working volume",
            "roi_fraction": getattr(model, "roi_fraction", NOT_AVAILABLE),
            "patchify": bool(((config.get("data", {}) or {})
                              .get("training_patch", {}) or {})
                             .get("enabled", False)),
            "batch_size": int((config.get("training", {}) or {})
                              .get("batch_size", 1)),
            "seed": int((config.get("experiment", {}) or {}).get("seed", 42)),
            "split_counts": [split.get("train_count"), split.get("val_count"),
                             split.get("test_count")],
            "split_sha256": split.get("split_sha256", NOT_AVAILABLE),
            "config_fingerprint": config_fingerprint or NOT_AVAILABLE,
            "git_commit": (git or {}).get("commit", NOT_AVAILABLE),
            "git_branch": (git or {}).get("branch", NOT_AVAILABLE),
            "batchnorm": type(model.ncas[0].bn).__name__,
            "verification": "derived from the constructed model, not from docs",
        }
        return self._done("architecture_identity.json",
                          _atomic_json(self.path("architecture_identity.json"),
                                       payload))

    def run_metadata(self, *, config: Dict[str, Any], identity: Dict[str, Any],
                     state: Optional[Dict[str, Any]] = None,
                     epoch: Any = NOT_MEASURED, step: Any = NOT_MEASURED,
                     precision: str = NOT_AVAILABLE) -> str:
        tr = config.get("training", {}) or {}
        opt = config.get("optimizer", {}) or {}
        payload = {
            "generated_utc": _utc(),
            "run_id": self.run_id,
            "git_commit": identity.get("git_commit", NOT_AVAILABLE),
            "git_branch": identity.get("git_branch", NOT_AVAILABLE),
            "parameters": identity.get("parameters"),
            "config_fingerprint": identity.get("config_fingerprint"),
            "split_sha256": identity.get("split_sha256"),
            "split_counts": identity.get("split_counts"),
            "seed": identity.get("seed"),
            "batch_size": identity.get("batch_size"),
            "precision": precision,
            "planned_max_epochs": tr.get("epochs", NOT_AVAILABLE),
            "current_epoch": epoch,
            "current_step": step,
            "optimizer": {"type": "AdamW",
                          "learning_rate": opt.get("learning_rate"),
                          "minimum_learning_rate":
                              opt.get("minimum_learning_rate"),
                          "weight_decay": opt.get("weight_decay")},
            "scheduler": "CosineAnnealingLR",
            "ema": config.get("ema", {}) or NOT_AVAILABLE,
            "loss": config.get("loss", {}) or NOT_AVAILABLE,
            "augmentation": tr.get("augmentation", NOT_AVAILABLE),
            "early_stopping": tr.get("early_stopping", NOT_AVAILABLE),
            "training_state": (state or {}).get("current_state", NOT_MEASURED),
            "host": {"node": platform.node(), "platform": platform.platform(),
                     "python": platform.python_version()},
        }
        return self._done("run_metadata.json",
                          _atomic_json(self.path("run_metadata.json"), payload))

    def hardware_metadata(self, *, env: Optional[Dict[str, Any]] = None) -> str:
        payload: Dict[str, Any] = {
            "generated_utc": _utc(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or NOT_AVAILABLE,
            "cpu_count": os.cpu_count() or NOT_AVAILABLE,
        }
        try:
            import psutil
            payload["ram_total_gb"] = round(
                psutil.virtual_memory().total / 1024 ** 3, 2)
        except Exception:
            payload["ram_total_gb"] = NOT_AVAILABLE
        try:
            import torch
            payload["torch"] = torch.__version__
            payload["cuda"] = torch.version.cuda or NOT_AVAILABLE
            payload["cudnn"] = (torch.backends.cudnn.version()
                                if torch.backends.cudnn.is_available()
                                else NOT_AVAILABLE)
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(0)
                props = torch.cuda.get_device_properties(0)
                payload.update({
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_compute_capability": f"{cap[0]}.{cap[1]}",
                    "gpu_total_vram_mb": round(props.total_memory / 1024 ** 2, 1),
                    "gpu_multiprocessors": props.multi_processor_count,
                    "bf16_native": cap >= (8, 0),
                })
            else:
                payload["gpu_name"] = NOT_AVAILABLE
                payload["gpu_count"] = 0
        except Exception:
            payload["torch"] = NOT_AVAILABLE
        if env:
            payload["environment_snapshot"] = env
        return self._done("hardware_metadata.json",
                          _atomic_json(self.path("hardware_metadata.json"),
                                       payload))

    # ------------------------------------------------------------- metrics
    def epoch_metrics(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            rows = [{"status": NOT_MEASURED,
                     "reason": "no epoch completed in this run"}]
        return self._done("epoch_metrics.csv",
                          _atomic_csv(self.path("epoch_metrics.csv"), rows))

    def validation_history(self, rows: List[Dict[str, Any]]) -> str:
        """Validation-only history. The test split is never written here."""
        if not rows:
            rows = [{"status": NOT_MEASURED,
                     "reason": "no validation evaluated in this run"}]
        return self._done("validation_history.csv",
                          _atomic_csv(self.path("validation_history.csv"), rows))

    def timing_summary(self, summary: Optional[Dict[str, Any]]) -> str:
        payload = summary or {"status": NOT_MEASURED,
                              "reason": "timing instrumentation produced no "
                                        "sections in this run"}
        payload.setdefault(
            "methodology",
            "Wall clock with CUDA synchronisation at section boundaries only. "
            "Sections NEST, so percentages deliberately do NOT sum to 100%.")
        return self._done("timing_summary.json",
                          _atomic_json(self.path("timing_summary.json"),
                                       payload))

    def precision_benchmark(self, source: Optional[str] = None) -> str:
        """Copy the §17 benchmark, or record why it is absent.

        Never synthesised: an invented precision comparison would be
        indistinguishable from a measured one in the final report.
        """
        payload: Dict[str, Any]
        if source and os.path.isfile(source):
            try:
                with open(source, encoding="utf-8") as fh:
                    payload = json.load(fh)
                payload["copied_from"] = source
            except (OSError, ValueError) as exc:
                payload = {"status": NOT_MEASURED,
                           "reason": f"benchmark unreadable: {exc}"}
        else:
            payload = {
                "status": NOT_MEASURED,
                "reason": ("scripts/benchmark_precision.py has not been run "
                           "for this machine"),
                "how_to_produce": "python scripts/benchmark_precision.py",
            }
        payload.setdefault("other_hardware", {
            "NVIDIA L4": NOT_MEASURED, "Tesla T4": NOT_MEASURED,
            "note": "no measurement exists on this repository for these GPUs",
        })
        return self._done("precision_benchmark.json",
                          _atomic_json(self.path("precision_benchmark.json"),
                                       payload))

    # --------------------------------------------------------- checkpoints
    def checkpoints(self, *, checkpoint_dir: str, identity: Dict[str, Any],
                    best: Optional[Dict[str, Any]] = None,
                    top_k: Optional[List[Dict[str, Any]]] = None,
                    periodic: Optional[List[str]] = None,
                    final: Optional[str] = None,
                    state: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Write the manifest plus the four per-kind metadata files."""
        def stat(p: str) -> Dict[str, Any]:
            if not p or not os.path.isfile(p):
                return {"path": p or NOT_AVAILABLE, "exists": False}
            return {"path": os.path.relpath(p, self.directory),
                    "exists": True,
                    "size_bytes": os.path.getsize(p),
                    "modified_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(os.path.getmtime(p)))}

        common = {k: identity.get(k) for k in _IDENTITY_KEYS}
        entries: List[Dict[str, Any]] = []

        best_meta = {"generated_utc": _utc(), "kind": "best", **common,
                     "selection_metric": ("validation mean foreground Dice, "
                                          "3-epoch rolling mean"),
                     "selection_split": "validation only",
                     "test_split_used": False}
        if best:
            best_meta.update(best)
            best_meta.update(stat(best.get("path", "")))
            entries.append({"kind": "best", "is_best": True, **best_meta})
        else:
            best_meta["status"] = NOT_MEASURED
            best_meta["reason"] = "no best checkpoint produced in this run"

        top_meta = {"generated_utc": _utc(), "kind": "top_k", **common,
                    "k": len(top_k or []),
                    "weight_averaging": False, "swa": False,
                    "note": ("Ranked by the same validation metric as best. "
                             "Weights are never averaged.")}
        if top_k:
            top_meta["ranking"] = top_k
            for i, e in enumerate(top_k):
                entries.append({"kind": "top_k", "rank": i + 1,
                                "is_top_k": True, **common, **e})
        else:
            top_meta["status"] = NOT_MEASURED
            top_meta["reason"] = "no ranked checkpoints produced"

        per_meta = {"generated_utc": _utc(), "kind": "periodic", **common,
                    "purpose": "Spot/preemptible recovery",
                    "count": len(periodic or [])}
        if periodic:
            per_meta["checkpoints"] = [stat(p) for p in periodic]
            for p in periodic:
                entries.append({"kind": "periodic", "is_periodic": True,
                                **common, **stat(p)})
        else:
            per_meta["status"] = NOT_MEASURED
            per_meta["reason"] = "no periodic snapshot reached in this run"

        fin_meta = {"generated_utc": _utc(), "kind": "final", **common,
                    "distinct_from_best": True,
                    "note": ("The final checkpoint is the LAST state, not the "
                             "selected model. Evaluation loads best.")}
        if final:
            fin_meta.update(stat(final))
            entries.append({"kind": "final", "is_final": True,
                            **common, **stat(final)})
        else:
            fin_meta["status"] = NOT_MEASURED
            fin_meta["reason"] = "no final checkpoint written"

        manifest = {
            "generated_utc": _utc(), "run_id": self.run_id,
            "checkpoint_dir": checkpoint_dir,
            "training_state": (state or {}).get("current_state", NOT_MEASURED),
            **common,
            "count": len(entries), "checkpoints": entries,
            "policy": {"best": "validation metric only",
                       "top_k": "ranked, never averaged",
                       "periodic": "recovery snapshots",
                       "final": "distinct from best"},
        }
        out = {
            "checkpoint_manifest.json": _atomic_json(
                self.path("checkpoint_manifest.json"), manifest),
            "best_checkpoint_metadata.json": _atomic_json(
                self.path("best_checkpoint_metadata.json"), best_meta),
            "top3_checkpoint_metadata.json": _atomic_json(
                self.path("top3_checkpoint_metadata.json"), top_meta),
            "periodic_checkpoint_metadata.json": _atomic_json(
                self.path("periodic_checkpoint_metadata.json"), per_meta),
            "final_checkpoint_metadata.json": _atomic_json(
                self.path("final_checkpoint_metadata.json"), fin_meta),
        }
        for k, v in out.items():
            self._done(k, v)
        return out

    def resume_verification(self, checks: Optional[Dict[str, Any]] = None,
                            executed: bool = False) -> str:
        """Record a resume check that was actually executed, or say it wasn't."""
        if not executed or not checks:
            payload = {"generated_utc": _utc(), "status": NOT_MEASURED,
                       "reason": ("resume was not exercised in this run; a "
                                  "claim of verified resume requires actually "
                                  "reloading a checkpoint")}
        else:
            payload = {"generated_utc": _utc(), "status": "VERIFIED",
                       "executed": True, "checks": checks,
                       "all_passed": all(bool(v) for v in checks.values())}
        return self._done("resume_verification.json",
                          _atomic_json(self.path("resume_verification.json"),
                                       payload))

    def overfitting_report(self, *, train_loss: List[float],
                           val_metric: List[float],
                           early_stopping: Optional[Dict[str, Any]] = None,
                           best_epoch: Any = NOT_MEASURED,
                           top_k: Optional[List[Dict[str, Any]]] = None) -> str:
        """Describe the observed trends. Draw no conclusion the data cannot bear."""
        payload: Dict[str, Any] = {
            "generated_utc": _utc(),
            "monitor": "validation mean foreground Dice (3-epoch rolling mean)",
            "selection_split": "validation only",
            "test_split_used_for_selection": False,
            "best_epoch": best_epoch,
            "early_stopping": early_stopping or NOT_MEASURED,
            "epochs_observed": len(val_metric),
        }
        # Three points is the minimum for a trend to mean anything; below that
        # the honest answer is that nothing was observed.
        if len(val_metric) < 3 or len(train_loss) < 3:
            payload.update({
                "status": NOT_MEASURED,
                "reason": (f"only {len(val_metric)} validation point(s) and "
                           f"{len(train_loss)} training point(s); too few to "
                           f"describe a trend"),
                "train_loss_trend": NOT_MEASURED,
                "validation_trend": NOT_MEASURED,
                "divergence": NOT_MEASURED,
            })
        else:
            half = max(1, len(train_loss) // 2)
            tl_first = sum(train_loss[:half]) / half
            tl_last = sum(train_loss[-half:]) / half
            vhalf = max(1, len(val_metric) // 2)
            v_first = sum(val_metric[:vhalf]) / vhalf
            v_last = sum(val_metric[-vhalf:]) / vhalf
            diverging = tl_last < tl_first and v_last < v_first
            payload.update({
                "status": "MEASURED",
                "train_loss_first_half_mean": round(tl_first, 6),
                "train_loss_second_half_mean": round(tl_last, 6),
                "train_loss_trend": ("decreasing" if tl_last < tl_first
                                     else "not decreasing"),
                "validation_first_half_mean": round(v_first, 6),
                "validation_second_half_mean": round(v_last, 6),
                "validation_trend": ("improving" if v_last > v_first
                                     else "not improving"),
                "best_validation": max(val_metric),
                "final_validation": val_metric[-1],
                "divergence": diverging,
                "interpretation": (
                    "Training loss fell while validation did not improve: the "
                    "classic overfitting signature. Early stopping exists for "
                    "exactly this case." if diverging else
                    "No train/validation divergence observed over the epochs "
                    "recorded."),
            })
        if top_k:
            payload["top_k_epochs"] = [e.get("epoch") for e in top_k]
            payload["top_k_scores"] = [e.get("score") for e in top_k]
        payload["scope"] = ("Describes the observed run only. It is NOT a "
                            "segmentation-quality claim.")
        return self._done("overfitting_report.json",
                          _atomic_json(self.path("overfitting_report.json"),
                                       payload))

    def gpu_diagnostics(self, diag_obj) -> str:
        """Delegate to the §18 sampler; record absence honestly."""
        p = self.path("gpu_diagnostics.csv")
        written = diag_obj.write_csv(p) if diag_obj is not None else None
        if not written:
            _atomic_csv(p, [{"status": NOT_MEASURED,
                             "reason": "no GPU samples collected in this run"}])
        return self._done("gpu_diagnostics.csv", p)

    # ------------------------------------------------------------- report
    def final_report(self, *, identity: Dict[str, Any],
                     extra_sections: Optional[Dict[str, str]] = None) -> str:
        lv = identity.get("levels", [])
        lines = [
            "# GLO-NCA — Experiment Report",
            "",
            f"Run `{self.run_id}` · generated {_utc()}",
            "",
            "## Architecture",
            "",
            "| | |", "|---|---|",
            f"| parameters | **{identity.get('parameters')}** |",
            f"| working volume | {identity.get('working_volume')}³ |",
        ]
        for l in lv:
            lines.append(
                f"| level {l['index']} | {l['resolution']}³ / "
                f"{l['channels']} ch / {l['nca_steps']} steps / "
                f"k={l['perception_kernel']} |")
        lines += [
            f"| level 3 | {'absent' if identity.get('level3_absent') else 'PRESENT'} |",
            f"| total NCA steps | {identity.get('total_nca_steps')} |",
            f"| spatial GC kernel | k={identity.get('spatial_gc_kernel')} |",
            f"| SE | {'ON' if identity.get('se_enabled') else 'OFF'} |",
            f"| fusion | Conv3d({identity.get('fusion', {}).get('in_channels')}"
            f" → {identity.get('fusion', {}).get('out_channels')}) |",
            f"| patchify | {'ON' if identity.get('patchify') else 'OFF'} |",
            f"| ROI | {identity.get('roi_fraction')} |",
            f"| batch | {identity.get('batch_size')} |",
            f"| seed | {identity.get('seed')} |",
            f"| split | {identity.get('split_counts')} |",
            f"| split SHA | `{identity.get('split_sha256')}` |",
            f"| git commit | `{identity.get('git_commit')}` |",
            "",
            "## Artifacts",
            "",
            "| artifact | written |", "|---|---|",
        ]
        for name in ARTIFACTS:
            p = self.path(name)
            lines.append(f"| `{name}` | {'yes' if os.path.isfile(p) else 'NO'} |")
        lines += [
            "",
            "## Scientific quality",
            "",
            "**No segmentation-quality claim is made here.** Dice, IoU and "
            "HD95 describe whatever run produced them; they become thesis "
            "evidence only after a full training and evaluation on the frozen "
            "split. Parameter count and timing say nothing about quality.",
            "",
        ]
        if extra_sections:
            for title, body in extra_sections.items():
                lines += [f"## {title}", "", body, ""]
        path = self.path("final_experiment_report.md")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        os.replace(tmp, path)
        return self._done("final_experiment_report.md", path)

    # ------------------------------------------------------------- auditing
    def audit_consistency(self, *, expected: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        """Re-read what was written and check the artifacts agree.

        Sixteen files that each state the parameter count are sixteen chances
        to disagree. A mismatch is reported, never silently repaired: the fix
        belongs in the producer.
        """
        report: Dict[str, Any] = {"generated_utc": _utc(), "checks": [],
                                  "missing": [], "unparseable": []}

        def check(name: str, ok: bool, detail: str = "") -> None:
            report["checks"].append({"check": name, "pass": bool(ok),
                                     "detail": detail})

        loaded: Dict[str, Any] = {}
        for name in ARTIFACTS:
            p = self.path(name)
            if not os.path.isfile(p):
                report["missing"].append(name)
                continue
            if name.endswith(".json"):
                try:
                    with open(p, encoding="utf-8") as fh:
                        loaded[name] = json.load(fh)
                except ValueError as exc:
                    report["unparseable"].append({"artifact": name,
                                                  "error": str(exc)[:120]})
            elif name.endswith(".csv"):
                try:
                    with open(p, newline="", encoding="utf-8") as fh:
                        loaded[name] = list(csv.DictReader(fh))
                except Exception as exc:                       # noqa: BLE001
                    report["unparseable"].append({"artifact": name,
                                                  "error": str(exc)[:120]})

        check("all 16 artifacts present", not report["missing"],
              f"missing: {report['missing']}" if report["missing"] else "")
        check("all artifacts parse", not report["unparseable"],
              str(report["unparseable"]) if report["unparseable"] else "")

        ident = loaded.get("architecture_identity.json", {})
        for key in _IDENTITY_KEYS:
            ref = ident.get(key)
            if ref is None:
                continue
            mismatches = []
            for name, obj in loaded.items():
                if not isinstance(obj, dict) or key not in obj:
                    continue
                if obj[key] not in (ref, NOT_AVAILABLE, NOT_MEASURED):
                    mismatches.append(f"{name}={obj[key]!r}")
            check(f"'{key}' agrees across artifacts", not mismatches,
                  f"identity={ref!r} but {mismatches}" if mismatches else
                  f"{ref!r}")

        for key, want in (expected or {}).items():
            got = ident.get(key)
            check(f"identity.{key} == {want!r}", got == want, f"got {got!r}")

        report["passed"] = all(c["pass"] for c in report["checks"])
        _atomic_json(self.path("artifact_consistency_audit.json"), report)
        return report
