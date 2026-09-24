r"""GLO-NCA training runner: orchestrates a full, resumable experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import torch

# Repository root, for resolving relative paths such as the split file.
_HERE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_GLO_NCA_Cell import GLO_NCA_Cell
from src.losses.LossFunctions import FocalTverskyCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.agents.Agent_GLO_NCA_V3 import Agent_GLO_NCA_V3
from src.models.Model_GLO_NCA_V3 import build_v3_from_config

from . import checkpoint as ckpt_io
from . import datasource, diagnostics as diag, early_stopping, environment
from . import extension, graphs
from . import state_machine as sm
from . import metrics_eval as ME
from . import reproducibility as repro
from . import statistics as STATS
from .config import Config
from .dataset_validation import (validate_dataset, summarize,
                                 ALLOWED_SEG_LABELS as BASE_ALLOWED_SEG_LABELS)
from .data_quality import (load_policy as dq_load_policy,
                           DataQualityPolicyError)
from . import diagnostics as diag
from .run_diagnosis import diagnose
from . import postprocess as PP
from . import per_case_diagnostics as PCD
from . import lesion_strata as LS
from src.profiling import configure as _configure_profiler
from src.profiling.memory import MemorySampler
from .logutil import CSVLogger, TensorBoard, get_logger
from .workspace import Workspace

REGIONS = ["WT", "TC", "ET"]

TRAIN_CSV_FIELDS = ["epoch", "loss", "lr", "epoch_seconds",
                    "cumulative_seconds", "gpu_mem_gb"]
VAL_CSV_FIELDS = ["epoch", "dice_mean", "smooth",
                  "dice_WT", "dice_TC", "dice_ET",
                  "iou_WT", "iou_TC", "iou_ET",
                  "hd95_WT", "hd95_TC", "hd95_ET"]



# DataLoader worker helpers live at module level so Windows spawn can pickle them.
class _WorkerInit:
    """Seed each DataLoader worker deterministically."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        s = (self.base_seed + worker_id) % (2 ** 32)
        np.random.seed(s)
        import random as _r
        _r.seed(s)


class _EpochSampler(torch.utils.data.Sampler):
    """Deterministic per-epoch shuffle yielding (epoch, index) for per-case augmentation seeds."""

    def __init__(self, n, base_seed):
        self.n, self.base_seed, self.epoch = n, base_seed, 0

    def set_epoch(self, ep):
        self.epoch = ep

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed((self.base_seed * 1_000_003 + self.epoch) % (2 ** 63))
        for i in torch.randperm(self.n, generator=g).tolist():
            yield (self.epoch, i)


# Training step.
def _clipped_batch_step(agent, data, loss_f, grad_clip: float,
                        empty_weight: float = 0.1,
                        amp_dtype=None,
                        ds_weight: float = 0.0) -> Dict[int, float]:
    """One training step; ``amp_dtype`` applies to the model forward only, the loss stays FP32."""
    # Profiling sections are no-ops unless profiling is enabled.
    from src.profiling import get_profiler
    prof = get_profiler()

    with prof.section("train/h2d_transfer", cuda=True):
        data = agent.prepare_data(data)
    with prof.section("train/forward", cuda=True):
        if amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs, targets = agent.get_outputs(data)
            outputs = outputs.float()  # FP32 loss
        else:
            outputs, targets = agent.get_outputs(data)
    with prof.section("train/zero_grad"):
        for opt in agent.optimizer:
            opt.zero_grad(set_to_none=True)
    loss = 0
    loss_ret: Dict[int, float] = {}
    # Small BCE on empty regions prevents Tversky collapse.
    with prof.section("train/loss", cuda=True):
        for m in range(outputs.shape[-1]):
            if 1 in targets[..., m]:
                loss_loc = loss_f(outputs[..., m], targets[..., m])
            else:
                prob = torch.sigmoid(outputs[..., m]).clamp(1e-6, 1. - 1e-6)
                loss_loc = empty_weight * torch.nn.functional.binary_cross_entropy(
                    prob, targets[..., m], reduction="mean")
            loss = loss + loss_loc
            loss_ret[m] = loss_loc.item()

        # Deep supervision: scaled mean aux-head loss; loss_ret reports the primary head only.
        aux_logits = getattr(agent, "last_aux_logits", None) or []
        if aux_logits and ds_weight > 0:
            aux_total = 0
            for a_cf in aux_logits:
                a_cl = a_cf.float().permute(0, 2, 3, 4, 1).contiguous()
                for m in range(a_cl.shape[-1]):
                    if 1 in targets[..., m]:
                        aux_total = aux_total + loss_f(a_cl[..., m], targets[..., m])
                    else:
                        prob = torch.sigmoid(a_cl[..., m]).clamp(1e-6, 1. - 1e-6)
                        aux_total = aux_total + empty_weight *                             torch.nn.functional.binary_cross_entropy(
                                prob, targets[..., m], reduction="mean")
            loss = loss + ds_weight * (aux_total / len(aux_logits))
    if loss != 0:
        # Backward includes NCA-step recompute from gradient checkpointing.
        with prof.section("train/backward", cuda=True):
            loss.backward()
        with prof.section("train/grad_clip", cuda=True):
            if grad_clip and grad_clip > 0:
                for net in agent.model:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        with prof.section("train/optimizer_step", cuda=True):
            for opt, sch in zip(agent.optimizer, agent.scheduler):
                opt.step()
                sch.step()
    return loss_ret


# Model / experiment construction.
def _is_v3(cfg: Config) -> bool:
    return str(cfg.get("model", "version", "")).lower() == "v3"


def _case_ids_for_state(exp, state: str):
    """Ordered case ids for a split, matching collect_probs' iteration order."""
    try:
        paths = exp.data_split.get_images(state)
    except Exception:
        return []
    ids = []
    for p in paths:
        base = os.path.basename(str(p))
        if base.endswith(".nii.gz"):
            base = base[:-len(".nii.gz")]
        for suffix in ("-t1n", "-t1c", "-t2w", "-t2f", "-seg"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        ids.append(base)
    return ids


def _dataset_validation_key(data_root, modalities, limit, policy) -> str:
    """Fingerprint of the dataset files and settings a validation verdict depends on."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(data_root).encode())
    h.update(",".join(sorted(modalities)).encode())
    h.update(str(limit or 0).encode())
    h.update(str(getattr(policy, "policy_version", None) or "").encode())
    try:
        from src.experiment.datasource import discover_cases
        cases = discover_cases(data_root)
    except Exception:
        return "unavailable"
    if limit:
        cases = cases[:limit]
    for cid, rel in cases:
        folder = os.path.join(data_root, rel)
        h.update(cid.encode())
        try:
            for name in sorted(os.listdir(folder)):
                st = os.stat(os.path.join(folder, name))
                h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            return "unavailable"
    return h.hexdigest()


def _load_cached_validation(cache_dir: str, key: str):
    """Return a cached PASS report for this key, or None."""
    if not key or key == "unavailable":
        return None
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception:
        return None
    return report if report.get("result") == "PASS" else None


def _store_cached_validation(cache_dir: str, key: str, report) -> None:
    if not key or key == "unavailable":
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{key}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def _training_et_voxels(ds, exp, ws, logger):
    """ET voxel count per training case (resampled voxels), cached per split for small-lesion sampling."""
    import numpy as np

    cache = ws.path("config", "et_voxels.json")
    ids = _case_ids_for_state(exp, "train")
    key = hashlib.sha256(("|".join(ids)).encode()).hexdigest()[:16]
    try:
        with open(cache, encoding="utf-8") as fh:
            blob = json.load(fh)
        if blob.get("key") == key and len(blob.get("et_voxels", [])) == len(ds):
            logger.info("small-lesion sampler: reusing cached ET counts (%d cases)",
                        len(blob["et_voxels"]))
            return blob["et_voxels"]
    except Exception:
        pass

    prev_state = getattr(ds, "state", None)
    logger.info("small-lesion sampler: counting ET voxels over %d training "
                "cases (once per split; cached afterwards)", len(ds))
    counts, t0 = [], time.time()
    try:
        exp.set_model_state("train")
        for i in range(len(ds)):
            item = ds[i]
            label = np.asarray(item[2])
            counts.append(int((label[..., 2] > 0.5).sum()))
            if (i + 1) % 100 == 0:
                logger.info("  ET scan %d/%d (%.1f min elapsed)",
                            i + 1, len(ds), (time.time() - t0) / 60)
    finally:
        if prev_state is not None:
            try:
                exp.set_model_state(prev_state)
            except Exception:
                pass

    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "unit": "resampled voxels (not mm^3)",
                       "split": "train", "et_voxels": counts}, fh)
    except Exception:
        pass
    pos = sum(1 for c in counts if c > 0)
    logger.info("small-lesion sampler: %d/%d cases ET-positive, ET voxels "
                "min %d median %d max %d (%.1f min)",
                pos, len(counts), min(counts) if counts else 0,
                int(np.median(counts)) if counts else 0,
                max(counts) if counts else 0, (time.time() - t0) / 60)
    return counts


def _build_production_model(cfg: Config, device):
    """Construct the model exactly as the runner does for a given config."""
    if (cfg.raw.get("model", {}) or {}).get("global_context") is not None:
        from src.models.Model_GLO_NCA_GlobalContext import (
            build_glo_nca_global_context)
        return build_glo_nca_global_context(cfg, input_channels=4,
                                            output_channels=3, device=device)
    return build_v3_from_config(cfg, input_channels=4, output_channels=3,
                                device=device)


GLO_NCA_PRODUCTION_NAME = "glo_nca_production"


def _production_identity(cfg: Config) -> bool:
    """True only for the exact production experiment name."""
    raw = str(cfg.get("experiment", "name", "") or getattr(cfg, "name", ""))
    return raw.lower().replace("-", "_") == GLO_NCA_PRODUCTION_NAME


def _build_dispatch(cfg: Config, data_root: str, device, epochs: int,
                    out_model_dir: str):
    """Dispatch to the two-level GLO-NCA builder (production) or the legacy V2 builder."""
    if _is_v3(cfg):
        return _build_v3(cfg, data_root, device, epochs, out_model_dir)
    return _build(cfg, data_root, device, epochs, out_model_dir)


def _build_v3(cfg: Config, data_root: str, device, epochs: int, out_model_dir: str):
    """Build the two-level GLO-NCA model and its single-model agent."""
    fire = float(cfg.get("model", "fire_rate", 0.6))
    aug = str(cfg.get("training", "augmentation"))
    patch = int(cfg.get("training", "patch_size"))  # 128³ working volume

    # Flat Experiment/dataset config; NCA-specific keys are inert for the two-level model.
    # Patchify follows data.training_patch.enabled (off in production).
    _patchify_enabled = bool(
        ((cfg.section("data") or {}).get("training_patch") or {})
        .get("enabled", False))

    config = [{
        "img_path": data_root, "label_path": data_root,
        "model_path": out_model_dir,
        "device": str(device), "unlock_CPU": True,
        "optimizer": "adamw",
        "lr": float(cfg.get("optimizer", "learning_rate")),
        "lr_gamma": 0.9999,
        "betas": (0.9, 0.99),
        "weight_decay": float(cfg.get("optimizer", "weight_decay")),
        "save_interval": 10 ** 9, "evaluate_interval": 10 ** 9, "n_epoch": epochs,
        "batch_size": int(cfg.get("training", "batch_size")), "batch_duplication": 1,
        "channel_n": 16, "inference_steps": 10, "cell_fire_rate": fire,
        "input_channels": 4, "output_channels": 3,
        "hidden_size": int(cfg.get("model", "hidden", 128)),
        "train_model": 0,  # one model; L1 and L2 live inside it
        "use_attention": bool(cfg.get("model", "use_attention", True)),
        # The dataset resamples each case to the working volume.
        "input_size": [[patch, patch, patch]], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "foreground_crop": True, "nonzero_norm": True,
        "augment": aug != "none", "augment_level": ("light" if aug == "light" else "heavy"),
        # Patch-level ET sampling; inert while patchify is off.
        "patchify": _patchify_enabled,
        "priotize_masks": float(cfg.get("sampling", "prioritize_probability", 0.7)),
        "prioritize_region": int(cfg.get("sampling", "prioritize_region", 2)),
    }]

    ds = Dataset_NiiGz_3D_BraTS()
    ds.MODALITIES = list(cfg.get("dataset", "modalities"))

    # Optional training patch (off in production): crop patch_size from working_volume, train only.
    _tp = (cfg.section("data") or {}).get("training_patch") or {}
    if bool(_tp.get("enabled", False)):
        _work = int(_tp.get("working_volume", patch))
        _pat = int(_tp.get("patch_size", patch))
        if _pat >= _work:
            raise ValueError(
                f"data.training_patch: patch_size ({_pat}) must be STRICTLY smaller "
                f"than working_volume ({_work}); otherwise the crop is a no-op.")
        config[0]["input_size"] = [[_work, _work, _work]]   # -> dataset.size
        ds.set_train_patch_size(_pat)

    if _production_identity(cfg) and _patchify_enabled:
        raise ValueError(
            "PRODUCTION IDENTITY VIOLATION: data.training_patch.enabled is true. "
            "The GLO-NCA production path trains on the full working volume; a "
            "crop would restrict the global-context level to part of the brain "
            "during training while it sees the whole brain at evaluation.")

    # Optional deterministic preprocessing cache; augmentation still runs every epoch.
    _cache_cfg = (cfg.section("data") or {}).get("cache") or {}
    if bool(_cache_cfg.get("enabled", False)):
        from src.datasets.preprocess_cache import PreprocessCache
        _cdir = _cache_cfg.get("directory") or os.path.join(_HERE_ROOT, ".cache",
                                                            "preprocessed")
        if not os.path.isabs(_cdir):
            _cdir = os.path.join(_HERE_ROOT, _cdir)
        # The cache stores the resampled working volume, so it is sized by dataset.size.
        _cache_size = int(config[0]["input_size"][0][0])
        ds.set_preprocess_cache(PreprocessCache(
            _cdir, dataset_root=data_root,
            modalities=list(cfg.get("dataset", "modalities")),
            size=(_cache_size, _cache_size, _cache_size),
            crop_fg=True, rescale=True, enabled=True))

    # Global context: L1 sees the full 128³ volume; roi_fraction 1.0 keeps L2 full-volume.
    v3_model = _build_production_model(cfg, device)
    ca = [v3_model]  # one-element list for the shared runner
    agent = Agent_GLO_NCA_V3(ca)
    exp = Experiment(config, ds, ca, agent)
    ds.set_experiment(exp)
    return ds, ca, agent, exp, config[0]


def _build(cfg: Config, data_root: str, device, epochs: int, out_model_dir: str):
    input_size = cfg.input_size
    steps = list(cfg.get("model", "steps"))
    fire = float(cfg.get("model", "fire_rate"))
    ch = int(cfg.get("model", "channel_n"))
    hidden = int(cfg.get("model", "hidden"))
    use_attn = bool(cfg.get("model", "use_attention"))
    use_spatial = bool(cfg.get("model", "use_spatial"))
    dropout = float(cfg.get("model", "dropout"))
    aug = str(cfg.get("training", "augmentation"))

    config = [{
        "img_path": data_root, "label_path": data_root,
        "model_path": out_model_dir,
        "device": str(device), "unlock_CPU": True,
        "optimizer": "adamw",
        "lr": float(cfg.get("optimizer", "learning_rate")),
        "lr_gamma": 0.9999,
        "betas": (0.9, 0.99),
        "weight_decay": float(cfg.get("optimizer", "weight_decay")),
        "save_interval": 10 ** 9, "evaluate_interval": 10 ** 9, "n_epoch": epochs,
        "batch_size": int(cfg.get("training", "batch_size")), "batch_duplication": 1,
        "channel_n": ch, "inference_steps": steps, "cell_fire_rate": fire,
        "input_channels": 4, "output_channels": 3, "hidden_size": hidden,
        "train_model": 1, "use_attention": use_attn,
        "input_size": input_size, "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "foreground_crop": True, "nonzero_norm": True,
        "augment": aug != "none", "augment_level": ("light" if aug == "light" else "heavy"),
        "patchify": bool(((cfg.section("data") or {}).get("training_patch")
                          or {}).get("enabled", False)),
        "priotize_masks": 0.7, "prioritize_region": 2,
    }]

    ds = Dataset_NiiGz_3D_BraTS()
    ds.MODALITIES = list(cfg.get("dataset", "modalities"))
    ca = [
        GLO_NCA_Cell(ch, fire, device, hidden, kernel_size=7, input_channels=4,
                   use_attention=use_attn, use_spatial=use_spatial, dropout=dropout),
        GLO_NCA_Cell(ch, fire, device, hidden, kernel_size=3, input_channels=4,
                   use_attention=use_attn, use_spatial=use_spatial, dropout=dropout),
    ]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent)
    ds.set_experiment(exp)
    return ds, ca, agent, exp, config[0]


def _model_info(cfg: Config, ca: List[torch.nn.Module]) -> Dict[str, Any]:
    total = sum(p.numel() for m in ca for p in m.parameters())
    trainable = sum(p.numel() for m in ca for p in m.parameters() if p.requires_grad)
    if _is_v3(cfg):
        # Report the per-level structure measured from the model.
        m = ca[0]
        pr = m.parameter_report() if hasattr(m, "parameter_report") else {}
        active_levels = [lv for lv in ("level1", "level2", "level3")
                         if bool((cfg.get("model", lv, {}) or {}).get("enabled", lv != "level3"))]
        return {
            "version": "v3",
            "architecture": "GLO-NCA-V3-MultiLevel",
            "total_parameters": int(total),        # measured, not hard-coded
            "trainable_parameters": int(trainable),
            "nca_levels": len(active_levels),
            "levels": {lv: cfg.get("model", lv) for lv in active_levels},
            "by_component": pr.get("by_level", {}),
            "fusion": (cfg.get("model", "feature_fusion", {}) or {}).get("type", "concat"),
            "se_enabled": bool(cfg.get("model", "use_attention", True)),
            "spatial_gc_enabled": bool(cfg.get("model", "use_spatial", True)),
            "hidden": int(cfg.get("model", "hidden", 128)),
        }
    return {
        "version": "v2",
        "total_parameters": int(total),         # measured, not hard-coded
        "trainable_parameters": int(trainable),
        "nca_levels": len(ca),
        "steps": list(cfg.get("model", "steps")),
        "se_enabled": bool(cfg.get("model", "use_attention")),
        "spatial_gc_enabled": bool(cfg.get("model", "use_spatial")),
        "channel_n": int(cfg.get("model", "channel_n")),
        "hidden": int(cfg.get("model", "hidden")),
    }


# Run.
def run(cfg: Config, ws: Workspace, *, resume: bool, device_str: str = None,
        extend_to: int = None, lr_policy: str = "freeze",
        extension_reason: str = None,
        stop_after_epoch: int = None) -> Dict[str, Any]:
    """Execute (or resume) a full experiment inside workspace ``ws``.

    ``stop_after_epoch`` pauses a run after that epoch's full checkpoint is written,
    without changing the planned budget or schedule; ``--resume`` continues it.
    """
    logger = get_logger(ws)
    ws.write_status("initializing", progress=0.0)
    # Lifecycle state: status.json for dashboards, state.json as the machine record.
    run_state = sm.TrainingState.load(ws.path("reports"),
                                      run_id=ws.experiment_id)
    gpu_diag = diag.GPUDiagnostics(enabled=True, min_interval_s=10.0)
    logger.info("=" * 70)
    logger.info("GLO-NCA %s experiment: %s",
                "V3" if _is_v3(cfg) else "V2", ws.experiment_id)
    logger.info("config: %s", cfg.path)

    # Environment, git and GPU capture.
    env_summary = environment.write_environment(ws)
    logger.info("torch %s | cuda_available=%s | %s",
                env_summary["gpu"].get("torch_version"),
                env_summary["gpu"].get("cuda_available"),
                env_summary["gpu"].get("device_name", "CPU"))
    logger.info(repro.describe())

    # Device.
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not used (device=%s). GPU is expected for the full "
                       "run; CPU is intended only for smoke tests.", device)

    # Profiling (no-op unless profiling.enabled).
    _prof = _configure_profiler(cfg, out_dir=ws.root)
    _mem = MemorySampler(enabled=getattr(_prof, "memory_enabled", False))         if _prof.enabled else None
    if _prof.enabled:
        logger.info("PROFILING ENABLED (observational): warmup=%d profiled=%d "
                    "-- this run is bounded and is NOT a training campaign",
                    _prof.warmup, _prof.iterations)

    # Seed and config snapshot.
    repro.set_all_seeds(cfg.seed)
    _copy_config_into(ws, cfg)

    # Data root and dataset validation.
    data_root = datasource.resolve_data_root(cfg.get("dataset", "root"))
    if not data_root:
        return _fail(ws, logger, "dataset root not found",
                     "Set dataset.root in the config or $DATA_ROOT.")
    ws.write_status("validating", progress=0.0)
    if run_state.can_transition(sm.PREFLIGHT):
        run_state.transition(sm.PREFLIGHT, "dataset + identity gates")
    n_pat = int(cfg.get("dataset", "number_of_patients", 0)) or None

    # Data-quality policy: per-case tolerated labels; a malformed policy fails loudly.
    _dq_section = cfg.section("data") if cfg.section("data") else {}
    dq_path = _dq_section.get("quality_policy_file", "__unset__")
    try:
        if dq_path is None:
            # No policy: any label outside {0..4} fails validation.
            dq_policy = None
        else:
            dq_policy = dq_load_policy(None if dq_path == "__unset__" else dq_path)
    except DataQualityPolicyError as exc:
        return _fail(ws, logger, "data-quality policy is invalid", str(exc))
    if dq_policy is not None:
        logger.info("data-quality: %s", dq_policy.describe())
    else:
        logger.info("data-quality: no policy file; validator fail-closed on "
                    "any label outside %s", sorted(BASE_ALLOWED_SEG_LABELS))

    # Reuse a cached validation PASS keyed on a file fingerprint; FAILs are never cached.
    _val_key = _dataset_validation_key(
        data_root, list(cfg.get("dataset", "modalities")), n_pat, dq_policy)
    _val_cache = os.path.join(
        os.path.expanduser(str((cfg.section("data") or {}).get("cache", {})
                               .get("directory", ".cache/preprocessed"))),
        "dataset_validation")
    report = _load_cached_validation(_val_cache, _val_key)
    if report is not None:
        logger.info("dataset validation: reusing cached PASS (fingerprint %s); "
                    "delete %s to force a rescan", _val_key[:12], _val_cache)
    else:
        try:
            report = validate_dataset(
                data_root, modalities=list(cfg.get("dataset", "modalities")),
                limit=n_pat, policy=dq_policy)
        except DataQualityPolicyError as exc:
            return _fail(ws, logger, "data-quality policy does not match the dataset",
                         str(exc))
        if report.get("result") == "PASS":
            _store_cached_validation(_val_cache, _val_key, report)
    ws.write_json(os.path.join("reports", "dataset_validation_report.json"), report)
    logger.info(summarize(report))
    if report["result"] != "PASS":
        return _fail(ws, logger, "dataset validation FAILED",
                     "See reports/dataset_validation_report.json. Training refuses "
                     "to start on invalid data (no silent skipping).")

    # Split: the saved split on resume, else the master split file.
    epochs = int(cfg.get("training", "epochs"))
    ds, ca, agent, exp, flat_cfg = _build_dispatch(
        cfg, data_root, device, epochs, ws.path("checkpoints", "model_internal"))

    # Priority: resume split, then data.split_file, then a seeded split (smoke tests only).
    split_file = cfg.get("data", "split_file") if cfg.section("data") else None

    # Real-data runs must name a split file; a seeded split needs allow_seeded_split: true.
    _allow_seeded = bool((cfg.section("data") or {}).get("allow_seeded_split", False)) \
        if cfg.section("data") else False
    if not split_file and _is_v3(cfg) and not _allow_seeded:
        return _fail(ws, logger,
                     "no data.split_file configured for a V3 run",
                     "V3 experiments must use the canonical master split. Add:\n"
                     "  data:\n    split_file: split/master_split.json\n"
                     "Refusing to silently generate a new split for a V3 run.\n"
                     "(Synthetic smoke configs may set data.allow_seeded_split: true.)")

    split_meta = {"source": None, "split_sha256": None, "split_file": split_file}
    existing = datasource.read_split(ws) if resume else None
    if existing:
        tr, va, te = existing
        split_meta["source"] = "resume"
        split_meta["split_sha256"] = datasource.split_fingerprint(tr, va, te)
        logger.info("resume: reusing saved split (train %d/val %d/test %d)",
                    len(tr), len(va), len(te))
    elif split_file:
        # Resolve against CWD, then the repository root.
        cand = split_file if os.path.exists(split_file) else os.path.join(_HERE_ROOT, split_file)
        master = datasource.load_master_split(cand)
        tr, va, te = master["train"], master["validation"], master["test"]
        split_meta.update(source="master", split_sha256=master["split_sha256"],
                          split_file=cand, split_version=master.get("split_version"))
        datasource.write_split(ws, tr, va, te)
        logger.info("MASTER split %s (fp %s) -> train %d/val %d/test %d",
                    cand, master["split_sha256"][:12], len(tr), len(va), len(te))
    else:
        tr, va, te = datasource.make_split(data_root, cfg.split_seed,
                                           n_patients=n_pat or 0)
        split_meta.update(source="seeded", split_sha256=datasource.split_fingerprint(tr, va, te))
        datasource.write_split(ws, tr, va, te)
    logger.info("Split -> train %d | val %d | test %d | fp %s",
                len(tr), len(va), len(te), (split_meta["split_sha256"] or "n/a")[:12])

    # Record dataset identity (read-only provenance).
    all_ids = datasource.list_patients(data_root)
    subject_count = len({datasource.subject_of(c) for c in all_ids})
    dataset_identity = {
        "dataset_root": data_root,
        "dataset_case_count": len(all_ids),        # case/timepoint dirs (not collapsed)
        "subject_count": subject_count,            # distinct base subjects
        "patient_id_hash": datasource.patient_id_hash(all_ids),  # over case ids
        "id_semantics": "case_id = subject + timepoint; split is subject-disjoint",
        "modalities": list(cfg.get("dataset", "modalities")),
        "split_source": split_meta["source"],
        "split_sha256": split_meta["split_sha256"],
        "split_file": split_meta.get("split_file"),
        # Data-quality policy in force and case counts.
        "data_quality_policy": (dq_policy.summary() if dq_policy else None),
        "tolerated_cases": report.get("tolerated_cases") or {},
        "operational_exclusions": (list(dq_policy.excluded) if dq_policy else []),
    }
    ws.write_json(os.path.join("config", "dataset_identity.json"), dataset_identity)

    # Map case ids to paths relative to the data root (handles nested cohorts).
    path_map = datasource.case_path_map(data_root)

    def entry(p):
        return (path_map.get(p, p), p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")

    # LR schedule: warmup, then cosine over the whole run.
    batch_size = int(cfg.get("training", "batch_size"))
    spe = max(1, math.ceil(len(tr) / batch_size))
    total_steps = epochs * spe
    lr_min = float(cfg.get("optimizer", "minimum_learning_rate"))
    # warmup_epochs > 0 uses WarmupCosineLR (same peak LR, minimum LR and step count).
    _warm_ep = float(cfg.get("optimizer", "warmup_epochs", 0) or 0)
    _warm_steps = int(round(_warm_ep * spe))
    # A run no longer than its warmup gets the warmup capped at half the run.
    if _warm_steps >= total_steps:
        _capped = max(0, total_steps // 2)
        logger.warning("LR warmup %d steps >= total %d steps; capping warmup "
                       "to %d (half the run). Production is unaffected.",
                       _warm_steps, total_steps, _capped)
        _warm_steps = _capped
    if _warm_steps > 0:
        from .warmup import WarmupCosineLR
        agent.scheduler = [
            WarmupCosineLR(opt, total_steps=total_steps,
                           warmup_steps=_warm_steps, eta_min=lr_min)
            for opt in agent.optimizer
        ]
        logger.info("LR warmup: %g epoch(s) configured, %d of %d steps applied, "
                    "then cosine to %g (peak %g preserved, total steps unchanged)",
                    _warm_ep, _warm_steps, total_steps, lr_min,
                    float(cfg.get("optimizer", "learning_rate")))
    else:
        agent.scheduler = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
            for opt in agent.optimizer
        ]
    # The scheduler actually built, recorded in the manifest.
    scheduler_record = {"name": type(agent.scheduler[0]).__name__,
                        "warmup_steps": int(_warm_steps), "total_steps": int(total_steps),
                        "eta_min": lr_min}

    # Loss weights from config (defaults match the original constants).
    ce_weight = float(cfg.get("loss", "ce_weight", 0.5))
    empty_weight = float(cfg.get("loss", "empty_region_bce_weight", 0.1))
    loss_f = FocalTverskyCELoss(
        alpha=1 - float(cfg.get("loss", "tversky_beta")),
        beta=float(cfg.get("loss", "tversky_beta")),
        gamma=float(cfg.get("loss", "focal_gamma")), ce_weight=ce_weight)

    # EMA.
    ema_on = bool(cfg.get("ema", "enabled"))
    ema_decay = float(cfg.get("ema", "decay"))
    ema = ([{k: v.detach().clone() for k, v in m.state_dict().items()} for m in ca]
           if ema_on else None)

    def ema_update():
        if ema is None:
            return
        for e, m in zip(ema, ca):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    e[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)

    def _swap_in_ema():
        """Swap EMA weights in for validation; return the raw states."""
        if ema is None:
            return None
        raw = [{k: v.detach().clone() for k, v in m.state_dict().items()}
               for m in ca]
        for m, e in zip(ca, ema):
            m.load_state_dict({k: v.to(m.state_dict()[k].dtype)
                               for k, v in e.items()})
        return raw

    def _restore_raw(raw):
        if raw is None:
            return
        for m, r in zip(ca, raw):
            m.load_state_dict(r)

    _ds_cfg = (cfg.section("model") or {}).get("deep_supervision") or {}
    _ds_weight = (float(_ds_cfg.get("weight", 0.0))
                  if bool(_ds_cfg.get("enabled", False)) else 0.0)

    grad_clip = (float(cfg.get("gradient", "max_norm"))
                 if bool(cfg.get("gradient", "clipping_enabled")) else 0.0)
    best_window = int(cfg.get("evaluation", "smoothing_window"))
    ckpt_freq = int(cfg.get("logging", "checkpoint_frequency", 10))

    # Selection and early stopping use smoothed validation Dice (mean WT/TC/ET); test is never used.
    stopper = early_stopping.from_config(cfg.section("training").get(
        "early_stopping"))
    top_k = int(cfg.get("logging", "top_k_checkpoints", 1))
    epochs_run = epochs          # overwritten if early stopping fires
    if stopper.enabled:
        logger.info("early stopping: monitor=%s mode=%s patience=%d "
                    "min_delta=%.4g (validation only)",
                    stopper.monitor, stopper.mode, stopper.patience,
                    stopper.min_delta)
    else:
        logger.info("early stopping: DISABLED -- training runs the full "
                    "%d-epoch budget", epochs)
    if top_k > 1:
        logger.info("top-%d best checkpoints retained (no weight averaging)",
                    top_k)

    # Precision applies to the model forward only; fp16 is refused (no GradScaler).
    _perf = cfg.section("performance") or {}
    _prec = str(_perf.get("precision", "fp32")).lower()
    amp_dtype = None
    if _prec in ("bf16", "bfloat16"):
        if device.type != "cuda":
            logger.warning("performance.precision=%s ignored: CUDA not in use", _prec)
        elif not torch.cuda.is_bf16_supported():
            logger.warning("performance.precision=%s ignored: bf16 unsupported on %s",
                           _prec, torch.cuda.get_device_name(0))
        else:
            amp_dtype = torch.bfloat16
    elif _prec in ("fp16", "float16"):
        return _fail(ws, logger, "unsupported precision",
                     "performance.precision=fp16 requires a GradScaler, which this "
                     "training step does not implement. Use bf16 (validated) or fp32.")
    elif _prec not in ("fp32", "float32", "none"):
        return _fail(ws, logger, "unknown precision",
                     f"performance.precision={_prec!r}; expected fp32 | bf16.")
    logger.info("precision: %s | autocast=%s | device=%s | dtype=%s",
                _prec, amp_dtype is not None, device,
                (str(amp_dtype).split('.')[-1] if amp_dtype else "float32"))

    # HD95 runs on a configured interval and always on the final epoch.
    hd95_every = int((_perf.get("hd95_every_epochs") or 1))
    if hd95_every < 1:
        return _fail(ws, logger, "invalid hd95_every_epochs",
                     f"performance.hd95_every_epochs={hd95_every}; must be >= 1.")
    if hd95_every > 1:
        logger.info("HD95 computed every %d epochs (Dice/mIoU every epoch; "
                    "model selection and test evaluation unchanged)", hd95_every)
    # Carry HD95 forward on skipped epochs (NaN until first computed).
    _last_hd95 = {r: float("nan") for r in REGIONS}

    info = _model_info(cfg, ca)
    logger.info("model params: total=%d trainable=%d (SE=%s, spatialGC=%s, levels=%d)",
                info["total_parameters"], info["trainable_parameters"],
                info["se_enabled"], info["spatial_gc_enabled"], info["nca_levels"])

    # Resume: restore full state.
    hist = {k: [] for k in ("epoch", "loss", "lr", "val_mean", "val_WT", "val_TC", "val_ET")}
    best, best_epoch, start_epoch = -1.0, 0, 0
    ext_plan = None          # set only when --extend-to is validated below
    if resume and os.path.exists(ws.last_ckpt):
        ck = ckpt_io.load_checkpoint(ws.last_ckpt, map_location=device)
        ckpt_io.restore_into(ck, models=ca, optimizers=agent.optimizer,
                             schedulers=agent.scheduler)
        # Refuse to resume if schedule-defining settings changed since the checkpoint.
        prev_cfg = ck.get("config") or {}
        if prev_cfg:
            _crit = [("training", "epochs"), ("training", "batch_size"),
                     ("training", "patch_size"), ("optimizer", "learning_rate"),
                     ("optimizer", "minimum_learning_rate"),
                     ("loss", "tversky_beta"), ("loss", "focal_gamma"),
                     ("ema", "decay"), ("experiment", "seed"),
                     # Warmup, spatial GC kernel and small-lesion sampling settings.
                     ("optimizer", "warmup_epochs"),
                     ("model", "spatial_kernel_size"),
                     ("lesion_aware_sampling", "enabled"),
                     ("lesion_aware_sampling", "boost"),
                     ("lesion_aware_sampling", "small_lesion_voxels")]
            drift = []
            for sec, key in _crit:
                old = (prev_cfg.get(sec) or {}).get(key)
                new = cfg.get(sec, key, None)
                if old is not None and new is not None and old != new:
                    drift.append(f"{sec}.{key}: checkpoint={old!r} -> config={new!r}")
            if drift:
                return _fail(ws, logger,
                             "resume config does not match the checkpoint",
                             "Resuming with changed training hyperparameters would "
                             "silently produce a schedule matching neither config:\n  "
                             + "\n  ".join(drift) +
                             "\nResume with the original config, or start a new "
                             "experiment deliberately.")
            logger.info("resume: config identity verified against checkpoint")

        if ema is not None:
            if ck.get("ema"):
                ema = [{k: v.to(device) for k, v in e.items()} for e in ck["ema"]]
                logger.info("resume: EMA weights restored from checkpoint")
            else:
                # Log when the EMA is re-initialised.
                logger.warning("resume: checkpoint contains NO EMA state; "
                               "re-initialising EMA from the restored weights "
                               "(it will re-converge over ~1/(1-decay) steps).")
        hist = ck.get("history", hist)
        best = ck.get("best_score", -1.0)
        best_epoch = ck.get("best_epoch", 0)
        start_epoch = int(ck.get("epoch", 0))
        if ck.get("early_stopping"):
            stopper.load_state_dict(ck["early_stopping"])
            logger.info("resume: early-stopping state restored "
                        "(best %.4f @ epoch %d, patience %d/%d)",
                        stopper.best if stopper.best is not None else float("nan"),
                        stopper.best_epoch, stopper.counter, stopper.patience)
        repro.restore_rng_state(ck.get("rng_state", {}))
        logger.info("RESUME: experiment=%s ckpt=%s prev_epoch=%d best_epoch=%d "
                    "best=%.4f -> next_epoch=%d",
                    ws.experiment_id, ws.last_ckpt, start_epoch, best_epoch,
                    best, start_epoch + 1)

        # Extension: validated against the loaded checkpoint.
        if extend_to is not None:
            try:
                ext_plan = extension.plan_extension(
                    checkpoint=ck, checkpoint_path=ws.last_ckpt,
                    current_cfg=cfg.to_dict(), target_epoch=int(extend_to),
                    lr_policy=lr_policy, reason=extension_reason or "",
                    parent_run_id=ws.experiment_id)
            except extension.ExtensionError as exc:
                return _fail(ws, logger, "extension refused", str(exc))

            epochs = ext_plan.target_epoch      # the loop's new upper bound
            ext_lr_note = extension.apply_lr_policy(
                agent.scheduler, ext_plan, steps_per_epoch=spe, logger=logger)
            ext_record = extension.write_extension_record(
                ws.path("reports"), ext_plan, ext_lr_note)
            logger.info("EXTENSION: %s", ext_plan.metadata()["reporting_note"])
            logger.info("  parent epoch %d -> target %d (%d epoch(s)), "
                        "lr-policy=%s, record=%s",
                        ext_plan.parent_epoch, ext_plan.target_epoch,
                        ext_plan.extension_epochs, ext_plan.lr_policy,
                        ext_record)
            if ext_plan.early_stopping_was_triggered:
                logger.warning("  parent run had ALREADY early-stopped; this "
                               "continuation is an explicit operator override")
            ws.write_status("extended", current_epoch=start_epoch,
                            total_epochs=epochs)
            if run_state.can_transition(sm.COMPLETED_300):
                run_state.transition(sm.COMPLETED_300,
                                     "parent run reached its planned budget")
            run_state.transition(sm.EXTENDED, ext_plan.reason,
                                 parent_epoch=ext_plan.parent_epoch,
                                 extend_to=ext_plan.target_epoch,
                                 lr_policy=ext_plan.lr_policy)
    elif resume:
        logger.warning("resume requested but no last.pth found; starting fresh.")
    if extend_to is not None and not (resume and os.path.exists(ws.last_ckpt)):
        # Fail closed: an extension needs a parent checkpoint.
        return _fail(ws, logger, "extension refused: no parent checkpoint",
                     f"--extend-to {extend_to} needs a complete checkpoint at "
                     f"{ws.last_ckpt}. An extension continues an existing run; "
                     f"it never starts one.")

    train_csv = CSVLogger(ws.path("metrics", "train.csv"), TRAIN_CSV_FIELDS)
    val_csv = CSVLogger(ws.path("metrics", "validation.csv"), VAL_CSV_FIELDS)
    tb = TensorBoard(ws.path("tensorboard"), enabled=bool(cfg.get("logging", "tensorboard")))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Identity metadata embedded in each ranked best checkpoint.
    _prov = ckpt_io.provenance_metadata(
        config=cfg.to_dict(),
        split_sha=str(split_meta.get("split_sha256") or ""),
        dataset_root=str(data_root or ""))

    t0 = time.time()
    ws.write_status("training", current_epoch=start_epoch, total_epochs=epochs,
                    progress=start_epoch / max(1, epochs), best_epoch=best_epoch,
                    best_score=best)
    run_state.transition(sm.RUNNING,
                         f"training from epoch {start_epoch + 1}",
                         start_epoch=start_epoch, total_epochs=epochs)
    run_state.set_metadata(parameters=sum(p_.numel() for m_ in ca
                                          for p_ in m_.parameters()),
                           planned_max_epochs=epochs,
                           split_sha256=split_meta.get("split_sha256"),
                           seed=cfg.seed, precision=_prec)
    logger.info("Loading + caching volumes (first pass is slow)...")

    workers = int(cfg.get("training", "workers"))

    # Persistent workers keep their caches; the epoch reaches workers through the
    # sampler, so augmentation varies per epoch while staying deterministic.
    _worker_init = _WorkerInit(cfg.seed)

    # Small-lesion sampling when enabled, else uniform; training loader only, epoch length unchanged.
    _las = (cfg.section("lesion_aware_sampling") or {})
    if bool(_las.get("enabled", False)):
        from .lesion_sampler import LesionAwareSampler
        _et = _training_et_voxels(ds, exp, ws, logger)
        sampler = LesionAwareSampler(
            _et, cfg.seed,
            boost=float(_las.get("boost", 2.0)),
            small_lesion_voxels=int(_las.get("small_lesion_voxels", 100)))
        _d = sampler.describe()
        logger.info("small-lesion sampler ACTIVE: %d cases, weights [%.3f, %.3f], "
                    "boost %.2f, small<=%d vox, %d draws/epoch (with replacement); "
                    "validation/test remain uniform",
                    _d["cases"], _d["weight_min"], _d["weight_max"],
                    _d["boost"], _d["small_lesion_voxels"], _d["samples_per_epoch"])
    else:
        sampler = _EpochSampler(len(ds), cfg.seed)
    ds.set_augmentation_seed(cfg.seed)

    persistent = workers > 0
    loader_kwargs = dict(
        sampler=sampler, batch_size=batch_size, num_workers=workers,
        pin_memory=(device.type == "cuda"), worker_init_fn=_worker_init)
    if persistent:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    # Build once when workers persist (keeps their caches warm across epochs).
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs) if persistent else None

    # Epoch loop.
    if stop_after_epoch is not None and not start_epoch < stop_after_epoch <= epochs:
        # Refuse without touching the run: it stays resumable at start_epoch.
        logger.error("--stop-after-epoch %d must be after the resume point (epoch %d) "
                     "and within the planned budget (%d); nothing was trained.",
                     stop_after_epoch, start_epoch, epochs)
        ws.write_status("paused", current_epoch=start_epoch, total_epochs=epochs,
                        progress=start_epoch / max(1, epochs))
        return {"status": "invalid-arguments", "workspace": ws.root}

    for ep in range(start_epoch, epochs):
        ep_start = time.time()
        losses = []
        gpu_diag.set_phase(diag.ACTIVE_GPU)
        # The epoch reaches persistent workers through the sampler.
        sampler.set_epoch(ep)
        if not persistent:  # workers == 0: rebuild the loader each epoch
            loader = torch.utils.data.DataLoader(ds, **loader_kwargs)
        # Iterate the loader manually to time data wait (profiling only).
        # The torch.profiler window is bounded to a few iterations.
        from src.profiling.cuda import torch_profiler as _torch_profiler
        _tp_on = bool((cfg.section("profiling") or {}).get("pytorch_profiler", False))             and _prof.enabled
        _tp_ctx = _torch_profiler(ws.root, enabled=_tp_on,
                                  wait=1, warmup=1, active=3)
        _it = iter(loader)
        _batch_idx = 0
        _tp = _tp_ctx.__enter__() if _tp_on else None
        while True:
            with _prof.section("data/dataloader_wait"):
                try:
                    data = next(_it)
                except StopIteration:
                    break
            _prof.mark_iteration(_batch_idx)
            with _prof.section("total/iteration", cuda=True):
                r = _clipped_batch_step(agent, data, loss_f, grad_clip, empty_weight,
                                        amp_dtype=amp_dtype,
                                        ds_weight=_ds_weight)
                with _prof.section("train/ema", cuda=True):
                    ema_update()
            if _prof.enabled and _mem is not None and _prof.is_profiled_iteration():
                for _k, _v in _mem.sample().items():
                    if _v is not None:
                        _prof.record(f"memory/{_k}", float(_v))
            if r:
                losses.append(sum(r.values()))
            # Sample GPU utilisation while compute is running.
            gpu_diag.sample(epoch=ep + 1, step=_batch_idx)
            if _tp is not None:
                _tp.step()
            _batch_idx += 1
            # A profiling run stops after its profiled iterations.
            if _prof.enabled and _prof.should_stop():
                logger.info("profiling: iteration budget reached (%d) -- "
                            "ending profiled epoch early; THIS IS NOT TRAINING",
                            _batch_idx)
                break

        if _tp_on:
            try:
                _tp_ctx.__exit__(None, None, None)
                logger.info("pytorch profiler trace -> %s",
                            os.path.join(ws.root, "profiler"))
            except Exception as _exc:
                logger.warning("pytorch profiler export failed: %s", _exc)

        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        if _prof.enabled:
            _prof.flush()
        gpu_diag.set_phase(diag.VALIDATION)
        gpu_diag.sample(epoch=ep + 1, step=-1)
        _raw_states = _swap_in_ema()
        with _prof.section("validation/total", cuda=True):
            # Dice/mIoU every epoch; HD95 on its interval, carried forward and flagged in hd95_fresh.
            _hd95_now = (hd95_every == 1 or (ep + 1) % hd95_every == 0
                         or (ep + 1) == epochs)
            if _hd95_now:
                val = ME.evaluate(agent, ds, "val")
                _last_hd95 = {r: val[r]["hd95"] for r in REGIONS}
            else:
                _pairs = ME.collect_probs(agent, ds, "val")
                _th = {r: 0.5 for r in REGIONS}
                val = {r: {"dice": 0.0, "iou": 0.0, "hd95": float("nan")}
                       for r in REGIONS}
                for i, r in enumerate(REGIONS):
                    _d, _i = [], []
                    for prob, gt in _pairs:
                        p, t = prob[..., i], gt[..., i]
                        inter = np.logical_and(p >= _th[r], t >= 0.5).sum()
                        denom = (p >= _th[r]).sum() + (t >= 0.5).sum() + 1e-6
                        _d.append((2 * inter) / denom)
                        _i.append(ME.iou_score(p, t, threshold=_th[r]))
                    val[r] = {"dice": float(np.mean(_d)), "iou": float(np.mean(_i)),
                              "hd95": _last_hd95.get(r, float("nan"))}
                del _pairs
        _restore_raw(_raw_states)
        _raw_states = None
        vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
        hist["epoch"].append(ep + 1)
        hist["loss"].append(float(np.mean(losses)) if losses else 0.0)
        hist["lr"].append(cur_lr)
        hist["val_mean"].append(vm)
        for r in REGIONS:
            hist[f"val_{r}"].append(val[r]["dice"])
        recent = hist["val_mean"][-best_window:]
        vm_smooth = float(np.mean(recent))

        ep_secs = time.time() - ep_start
        cum_secs = time.time() - t0
        gpu_mem = (torch.cuda.max_memory_allocated() / 1e9
                   if device.type == "cuda" else 0.0)

        # CSV
        train_csv.append({"epoch": ep + 1, "loss": hist["loss"][-1], "lr": cur_lr,
                          "epoch_seconds": round(ep_secs, 2),
                          "cumulative_seconds": round(cum_secs, 2),
                          "gpu_mem_gb": round(gpu_mem, 3)})
        val_csv.append({"epoch": ep + 1, "dice_mean": vm, "smooth": vm_smooth,
                        **{f"dice_{r}": val[r]["dice"] for r in REGIONS},
                        **{f"iou_{r}": val[r]["iou"] for r in REGIONS},
                        **{f"hd95_{r}": val[r]["hd95"] for r in REGIONS}})
        # TensorBoard
        tb.add_scalars(ep + 1, {
            "loss/train": hist["loss"][-1], "lr": cur_lr,
            "dice/mean": vm, "dice/smooth": vm_smooth,
            **{f"dice/{r}": val[r]["dice"] for r in REGIONS},
            **{f"miou/{r}": val[r]["iou"] for r in REGIONS},
            **{f"hd95/{r}": val[r]["hd95"] for r in REGIONS},
            "time/epoch_seconds": ep_secs, "time/cumulative_seconds": cum_secs,
            "gpu/mem_gb": gpu_mem,
        })
        logger.info("ep %d/%d | lr %.2e | loss %.3f | val %.3f "
                    "(WT %.3f TC %.3f ET %.3f) | smooth %.3f | %.0fs",
                    ep + 1, epochs, cur_lr, hist["loss"][-1], vm,
                    val["WT"]["dice"], val["TC"]["dice"], val["ET"]["dice"],
                    vm_smooth, ep_secs)

        # --- best (on smoothed val) ---
        if vm_smooth > best:
            best, best_epoch = vm_smooth, ep + 1
            weights = ema if ema is not None else [m.state_dict() for m in ca]
            ckpt_io.save_best_weights(ws.best_ckpt, weights, ep + 1, vm, vm_smooth)
            logger.info("   * new best (smoothed) %.3f saved", vm_smooth)
            if run_state.can_transition(sm.BEST_UPDATED):
                run_state.transition(sm.BEST_UPDATED,
                                     f"new best {vm_smooth:.4f}",
                                     epoch=ep + 1, score=float(vm_smooth))

            # Ranked top-K checkpoints on the same metric; weights are never averaged.
            if top_k > 1:
                _ranked = ckpt_io.update_top_k(
                    ws.path("checkpoints", "top_k"), weights=weights,
                    epoch=ep + 1, score=vm_smooth,
                    metrics={"dice_mean": vm, "dice_smooth": vm_smooth,
                             **{f"dice_{r}": val[r]["dice"] for r in REGIONS},
                             **{f"iou_{r}": val[r]["iou"] for r in REGIONS},
                             **{f"hd95_{r}": val[r]["hd95"] for r in REGIONS}},
                    meta=_prov, top_k=top_k)
                logger.info("   * top-%d: %s", top_k,
                            ", ".join(f"ep{e['epoch']}={e['score']:.4f}"
                                      for e in _ranked))

        # Early stopping (validation only).
        _stop = stopper.update(vm_smooth, ep + 1)
        if stopper.enabled:
            logger.info("   %s", stopper.status())

        # Full last.pth every epoch, plus periodic snapshots.
        full = ckpt_io.build_checkpoint(
            epoch=ep + 1, models=ca, optimizers=agent.optimizer,
            schedulers=agent.scheduler, ema=ema, best_score=best,
            best_epoch=best_epoch, history=hist, config=cfg.to_dict(),
            rng_state=repro.capture_rng_state())
        # Persist patience so it survives a preemption.
        full["early_stopping"] = stopper.state_dict()
        with _prof.section("train/checkpoint_write"):
            ckpt_io.save_checkpoint(ws.last_ckpt, full)
        if ckpt_freq and (ep + 1) % ckpt_freq == 0:
            with gpu_diag.phase(diag.CHECKPOINT):
                ckpt_io.save_checkpoint(ws.periodic_ckpt(ep + 1), full)
                gpu_diag.sample(epoch=ep + 1, step=ep + 1)
        if run_state.can_transition(sm.CHECKPOINTED):
            run_state.transition(sm.CHECKPOINTED, f"epoch {ep + 1} persisted",
                                 epoch=ep + 1)

        ws.write_status("training", current_epoch=ep + 1, total_epochs=epochs,
                        progress=(ep + 1) / max(1, epochs),
                        best_epoch=best_epoch, best_score=best)

        # Planned pause: the full checkpoint for this epoch is on disk. Return before
        # any end-of-run evaluation, so the test split is never read. The budget and
        # LR schedule are unchanged; --resume continues at the next epoch. An early
        # stop at or before the pause also pauses, leaving the decision to the user.
        if stop_after_epoch is not None and (ep + 1 >= stop_after_epoch or _stop):
            train_time = time.time() - t0
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if device.type == "cuda" else 0.0)
            pause = {"paused_after_epoch": ep + 1, "planned_epochs": epochs,
                     "next_epoch": ep + 2, "best_epoch": best_epoch,
                     "best_smoothed_val_dice": best,
                     "early_stopping_triggered": bool(_stop),
                     "session_train_seconds": round(train_time, 1),
                     "session_peak_gpu_mem_gb": round(peak, 3),
                     "checkpoint": os.path.relpath(ws.last_ckpt, ws.root),
                     "test_split_evaluated": False}
            ws.write_json(os.path.join("reports", "pause.json"), pause)
            ws.write_status("paused", current_epoch=ep + 1, total_epochs=epochs,
                            progress=(ep + 1) / max(1, epochs),
                            best_epoch=best_epoch, best_score=best)
            logger.info("PAUSED after epoch %d of %d (checkpoint %s); resume with "
                        "--resume to continue at epoch %d. Test split not evaluated.",
                        ep + 1, epochs, ws.last_ckpt, ep + 2)
            return {"status": "paused", "epoch": ep + 1, "workspace": ws.root}

        # Stop after the full checkpoint is written; best.pth is never overwritten.
        if _stop:
            logger.info("early stopping at epoch %d: %s has not improved by "
                        ">%.4g for %d consecutive validations "
                        "(best %.4f @ epoch %d)",
                        ep + 1, stopper.monitor, stopper.min_delta,
                        stopper.patience, stopper.best, stopper.best_epoch)
            logger.info("   %d of %d epochs used; evaluating the BEST "
                        "checkpoint, not the final one", ep + 1, epochs)
            epochs_run = ep + 1
            run_state.transition(
                sm.EARLY_STOPPED,
                f"{stopper.monitor} did not improve by >{stopper.min_delta:g} "
                f"for {stopper.patience} consecutive validations",
                epoch=ep + 1, best_epoch=stopper.best_epoch,
                best_value=stopper.best)
            break

    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # Extended runs complete in a distinct lifecycle state.
    if not stopper.should_stop:
        if ext_plan is not None:
            run_state.transition(sm.COMPLETED_EXTENSION,
                                 f"reached extension target epoch {epochs}",
                                 epochs_run=epochs_run)
        elif run_state.can_transition(sm.COMPLETED_300):
            run_state.transition(sm.COMPLETED_300,
                                 f"planned budget of {epochs} epochs reached",
                                 epochs_run=epochs_run)

    # Write diagnostics whatever the outcome.
    _diag_csv = gpu_diag.write_csv(ws.path("reports", "gpu_diagnostics.csv"))
    if _diag_csv:
        _dsum = gpu_diag.summary()
        logger.info("GPU diagnostics: %d samples -> %s",
                    _dsum.get("samples", 0), os.path.relpath(_diag_csv, ws.root))
        if _dsum.get("warning"):
            logger.warning("%s", _dsum["warning"])
        if _dsum.get("throttle_warning"):
            logger.warning("%s", _dsum["throttle_warning"])
        ws.write_json(os.path.join("reports", "gpu_diagnostics_summary.json"),
                      _dsum)

    # Evaluation.
    ws.write_status("evaluating", progress=1.0, best_epoch=best_epoch, best_score=best)
    if not os.path.exists(ws.best_ckpt):
        return _fail(ws, logger, "no best checkpoint produced",
                     "Training finished without saving a best model.")
    bw = ckpt_io.load_checkpoint(ws.best_ckpt, map_location=device)
    for m, sd in zip(ca, bw["m"]):
        m.load_state_dict(sd)

    tune = bool(cfg.get("evaluation", "tune_thresholds"))
    _pp_cfg = (cfg.section("evaluation") or {}).get("postprocessing") or {}
    _pp_on = bool(_pp_cfg.get("enabled", False))
    _min_comp = PP.min_component_config(cfg) if _pp_on else {r: 0 for r in REGIONS}
    logger.info("post-processing: %s  min component voxels %s",
                "ON" if _pp_on else "OFF", _min_comp)

    val_pairs = ME.collect_probs(agent, ds, "val")
    if _pp_on:
        val_pairs = PP.apply_to_pairs(val_pairs, {r: 0.5 for r in REGIONS},
                                      _min_comp)
    thresholds = ME.tune_thresholds(val_pairs) if tune else {r: 0.5 for r in REGIONS}
    test_pairs = ME.collect_probs(agent, ds, "test")
    if _pp_on:
        test_pairs = PP.apply_to_pairs(test_pairs, thresholds, _min_comp)
    test_05 = ME.score(test_pairs, {r: 0.5 for r in REGIONS})
    test = ME.score(test_pairs, thresholds)

    # Per-case diagnostics, validation and test in separate files; aux heads are never evaluated.
    try:
        _val_rows = PCD.build_rows(val_pairs, _case_ids_for_state(exp, "val"),
                                   "validation", {r: 0.5 for r in REGIONS})
        PCD.write_csv(ws.path("reports", "validation_per_case.csv"), _val_rows)
        LS.write_csv(ws.path("reports", "validation_by_lesion_size.csv"),
                     LS.stratify(_val_rows))
        _val_summary = PCD.summarise(_val_rows)
        _test_rows = PCD.build_rows(test_pairs, _case_ids_for_state(exp, "test"),
                                    "test", thresholds)
        PCD.write_csv(ws.path("reports", "test_per_case.csv"), _test_rows)
        LS.write_csv(ws.path("reports", "test_by_lesion_size.csv"),
                     LS.stratify(_test_rows))
        _test_summary = PCD.summarise(_test_rows)
        for _split, _sm in (("validation", _val_summary), ("test", _test_summary)):
            for _r in REGIONS:
                _reg = _sm["regions"][_r]
                logger.info("per-case %s %s: HD95 valid %d / invalid %d %s",
                            _split, _r, _reg["hd95_valid_cases"],
                            _reg["hd95_invalid_cases"],
                            _reg["hd95_invalid_reasons"] or "")
        _diag_summary = {"validation": _val_summary, "test": _test_summary}
    except Exception as _exc:
        logger.warning("per-case diagnostics NOT written: %s: %s",
                       type(_exc).__name__, _exc)
        _diag_summary = {"status": "NOT MEASURED",
                         "reason": f"{type(_exc).__name__}: {_exc}"}

    _log_table(logger, f"FINAL TEST @ 0.5 (best @ epoch {bw['ep']})", test_05)
    _log_table(logger, "FINAL TEST @ tuned thresholds "
               + ", ".join(f"{r}={thresholds[r]:.2f}" for r in REGIONS), test)
    logger.info("train time %.0fs | peak VRAM %.2f GB | params %d",
                train_time, peak, info["total_parameters"])

    run_diag = diagnose(hist, bw["ep"], test_05, test, thresholds, peak)
    logger.info("DIAGNOSTIC: %s", run_diag["verdict"])
    for line in run_diag["lines"]:
        logger.info("  %s", line)

    # Packaging.
    ws.write_status("packaging", progress=1.0)
    metrics_test = {"epoch": bw["ep"], **{f"dice_{r}": test[r]["dice"] for r in REGIONS},
                    **{f"iou_{r}": test[r]["iou"] for r in REGIONS},
                    **{f"hd95_{r}": test[r]["hd95"] for r in REGIONS},
                    "dice_mean": float(np.mean([test[r]["dice"] for r in REGIONS]))}
    CSVLogger(ws.path("metrics", "test.csv"), list(metrics_test.keys())).append(metrics_test)

    results = {"test": test, "test_at_0.5": test_05, "thresholds": thresholds,
               "history": hist, "best_epoch": bw["ep"], "diagnostics": run_diag,
               "params": info["total_parameters"], "train_time": train_time,
               "peak_vram": peak}
    ws.write_json(os.path.join("reports", "results.json"), results)
    ws.write_json(os.path.join("reports", "diagnostic_report.json"), diag)
    with open(ws.path("reports", "diagnostic_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("GLO-NCA DIAGNOSTIC REPORT\n" + "=" * 40 + "\n")
        for line in run_diag["lines"]:
            fh.write(line + "\n")
        fh.write("-" * 40 + "\nVERDICT: " + run_diag["verdict"] + "\n")
    _write_thesis_csv(ws, test, test_05, thresholds, info)

    # Per-case metrics and bootstrap statistics on test, using frozen validation-tuned thresholds.
    # Write the threshold comparison first, then free val_pairs to cut peak RAM.
    _write_threshold_comparison(ws, val_pairs, test_pairs, thresholds,
                                test_05_precomputed=test_05,
                                test_tuned_precomputed=test)
    del val_pairs
    import gc as _gc
    _gc.collect()

    per_case = ME.score_per_case(test_pairs, thresholds)
    _write_per_case_csv(ws, per_case)
    stats = STATS.summarize_per_case(per_case, n_boot=2000, seed=cfg.seed)
    ws.write_json(os.path.join("reports", "statistical_summary.json"), stats)

    # Write profiling artifacts (no-op unless enabled).
    if _prof.enabled:
        try:
            from src.profiling import write_reports
            from src.profiling.cuda import device_info
            _files = _prof.write(ws.root)
            _rel = _mem.release() if _mem is not None else None
            if _rel is not None:
                _rel["peak_classification"] = _mem.classify_peak(
                    (_rel.get("before") or {}).get("gpu_peak_mb"))
            _rep = write_reports(
                _prof, ws.root,
                environment={**device_info(),
                             "python": __import__("sys").version.split()[0],
                             "experiment": ws.experiment_id},
                configuration={
                    "config": cfg.path,
                    "levels": {lv: cfg.get("model", lv) for lv in
                               ("level1", "level2", "level3")},
                    "batch_size": batch_size,
                    "workers": workers,
                    "persistent_workers": bool(persistent),
                    "gradient_checkpointing": bool(
                        (cfg.raw.get("memory", {}) or {}).get(
                            "gradient_checkpointing", False)),
                    "augmentation": cfg.get("training", "augmentation"),
                    "optimizer": "adamw",
                    "ema_decay": float(cfg.get("ema", "decay")),
                },
                memory=_rel,
                notes=("Observational profiling run. Bounded to "
                       f"{_prof.warmup} warmup + {_prof.iterations} profiled "
                       "iterations; it is NOT a training campaign and the "
                       "resulting model is not a scientific result."))
            logger.info("profiling artifacts: %s | report=%s", _files, _rep)
        except Exception as exc:   # never let profiling break a run
            logger.warning("profiling report generation failed: %s", exc)

    graph_files = graphs.generate(ws)
    logger.info("graphs: %s", [os.path.basename(g) for g in graph_files])
    tb.close()

    manifest = _manifest(cfg, ws, env_summary, info, tr, va, te, data_root,
                         bw["ep"], test, train_time, peak, "completed",
                         dataset_identity=dataset_identity,
                         split_meta=split_meta, epochs_run=epochs_run,
                         stopper=stopper, extend_to=extend_to,
                         ext_plan=ext_plan, scheduler_record=scheduler_record)
    ws.write_manifest(manifest)
    ws.write_status("completed", progress=1.0, best_epoch=bw["ep"],
                    best_score=best, test_mean=metrics_test["dice_mean"])
    logger.info("Experiment COMPLETED -> %s", ws.root)
    return {"status": "completed", "results": results, "workspace": ws.root}


# Helpers.
def _copy_config_into(ws: Workspace, cfg: Config) -> None:
    import shutil
    dest = ws.path("config", "config.yaml")
    # On resume the config is already the workspace copy.
    if (cfg.path and os.path.exists(cfg.path)
            and os.path.abspath(cfg.path) != os.path.abspath(dest)):
        shutil.copy(cfg.path, dest)
    ws.write_json(os.path.join("config", "config_resolved.json"), cfg.to_dict())


def _log_table(logger, title, res) -> None:
    logger.info("%s", "=" * 60)
    logger.info("%s", title)
    logger.info("%-8s%-12s%-12s%-12s", "region", "Dice", "mIoU", "HD95(vox)")
    for r in REGIONS:
        logger.info("%-8s%-12.4f%-12.4f%-12.3f", r, res[r]["dice"],
                    res[r]["iou"], res[r]["hd95"])
    logger.info("%-8s%-12.4f", "mean", float(np.mean([res[r]["dice"] for r in REGIONS])))


def _write_thesis_csv(ws, test, test_05, thresholds, info) -> None:
    import csv
    with open(ws.path("reports", "thesis_results.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "dice_tuned", "dice_at_0.5", "miou",
                    "hd95_vox", "threshold"])
        for r in REGIONS:
            w.writerow([r, f"{test[r]['dice']:.4f}", f"{test_05[r]['dice']:.4f}",
                        f"{test[r]['iou']:.4f}", f"{test[r]['hd95']:.3f}",
                        f"{thresholds[r]:.2f}"])
        w.writerow(["mean", f"{np.mean([test[r]['dice'] for r in REGIONS]):.4f}",
                    f"{np.mean([test_05[r]['dice'] for r in REGIONS]):.4f}", "", "", ""])
        w.writerow([])
        w.writerow(["total_parameters", info["total_parameters"]])


def _write_per_case_csv(ws, per_case) -> None:
    """Per-case, per-region dice/iou/hd95 rows at the frozen tuned thresholds."""
    import csv
    n = len(per_case[REGIONS[0]]["dice"])
    with open(ws.path("reports", "per_case_test.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_index", "region", "dice", "iou", "hd95_vox"])
        for i in range(n):
            for r in REGIONS:
                w.writerow([i, r,
                            f"{per_case[r]['dice'][i]:.6f}",
                            f"{per_case[r]['iou'][i]:.6f}",
                            f"{per_case[r]['hd95'][i]:.4f}"])


def _write_threshold_comparison(ws, val_pairs, test_pairs, thresholds,
                                test_05_precomputed=None,
                                test_tuned_precomputed=None) -> None:
    """Write threshold_comparison.csv (default 0.5 vs tuned) for val and test.

    Reuses already-computed test scores when supplied, avoiding a second HD95 pass.
    """
    import csv
    half = {r: 0.5 for r in REGIONS}
    val_05, val_tuned = ME.score(val_pairs, half), ME.score(val_pairs, thresholds)
    test_05 = (test_05_precomputed if test_05_precomputed is not None
               else ME.score(test_pairs, half))
    test_tuned = (test_tuned_precomputed if test_tuned_precomputed is not None
                  else ME.score(test_pairs, thresholds))
    with open(ws.path("reports", "threshold_comparison.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "threshold_default", "threshold_tuned",
                    "validation_Dice_default", "validation_Dice_tuned",
                    "test_Dice_default", "test_Dice_tuned"])
        for r in REGIONS:
            w.writerow([r, "0.50", f"{thresholds[r]:.2f}",
                        f"{val_05[r]['dice']:.4f}", f"{val_tuned[r]['dice']:.4f}",
                        f"{test_05[r]['dice']:.4f}", f"{test_tuned[r]['dice']:.4f}"])


def _manifest(cfg, ws, env_summary, info, tr, va, te, data_root,
              best_epoch, test, train_time, peak, status,
              dataset_identity=None, split_meta=None,
              epochs_run=None, stopper=None, extend_to=None,
              ext_plan=None, scheduler_record=None) -> Dict[str, Any]:
    return {
        "experiment_id": ws.experiment_id,
        "name": cfg.name,
        "datetime": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "git": env_summary.get("git"),
        "dataset": {"root": data_root, "modalities": list(cfg.get("dataset", "modalities")),
                    "counts": {"train": len(tr), "val": len(va), "test": len(te)},
                    "case_count": (dataset_identity or {}).get("dataset_case_count"),
                    "subject_count": (dataset_identity or {}).get("subject_count"),
                    "patient_id_hash": (dataset_identity or {}).get("patient_id_hash")},
        "data_quality_policy": (dataset_identity or {}).get("data_quality_policy"),
        "split": {"source": (split_meta or {}).get("source"),
                  "split_sha256": (split_meta or {}).get("split_sha256"),
                  "split_file": (split_meta or {}).get("split_file"),
                  "split_version": (split_meta or {}).get("split_version")},
        "model": info,
        "training": {"epochs": int(cfg.get("training", "epochs")),
                     "batch_size": int(cfg.get("training", "batch_size")),
                     "patch_size": int(cfg.get("training", "patch_size")),
                     "augmentation": cfg.get("training", "augmentation"),
                     "input_size": (None if _is_v3(cfg) else cfg.input_size)},
        "optimizer": {"name": "adamw",
                      "learning_rate": float(cfg.get("optimizer", "learning_rate")),
                      "weight_decay": float(cfg.get("optimizer", "weight_decay"))},
        "scheduler": scheduler_record or {"name": "unknown"},
        "loss": {"type": "FocalTverskyCELoss",
                 "tversky_beta": float(cfg.get("loss", "tversky_beta")),
                 "focal_gamma": float(cfg.get("loss", "focal_gamma"))},
        "ema": {"enabled": bool(cfg.get("ema", "enabled")),
                "decay": float(cfg.get("ema", "decay"))},
        "seed": cfg.seed, "split_seed": cfg.split_seed,
        "gpu": env_summary.get("gpu"),
        "software": env_summary.get("software"),
        "best_epoch": best_epoch,
        # planned_max_epochs vs actual_completed_epochs (differ on early stop or extension).
        "planned_max_epochs": int(cfg.get("training", "epochs")),
        "actual_completed_epochs": epochs_run,
        "stopped_by_early_stopping": bool(stopper.should_stop),
        "stop_reason": (
            f"{stopper.monitor} did not improve by >{stopper.min_delta:g} for "
            f"{stopper.patience} consecutive validations"
            if stopper.should_stop else
            ("extension target reached" if extend_to is not None
             else "planned epoch budget reached")),
        "early_stopping": stopper.report(),
        "extension": (ext_plan.metadata() if ext_plan is not None else
                      {"extension_mode": False}),
        "final_test": {r: test[r] for r in REGIONS},
        "status": status,
    }


def _dump_profiling_on_exit(ws, logger) -> None:
    """Persist profiling data even on a failed run."""
    try:
        from src.profiling import get_profiler
        prof = get_profiler()
        if getattr(prof, "enabled", False):
            prof.flush()
            files = prof.write(ws.root)
            if files:
                logger.info("profiling data preserved: %s", files)
    except Exception:
        pass


def _fail(ws: Workspace, logger, reason: str, detail: str) -> Dict[str, Any]:
    _dump_profiling_on_exit(ws, logger)
    logger.error("FAILURE: %s -- %s", reason, detail)
    with open(ws.path("reports", "failure_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("GLO-NCA FAILURE REPORT\n" + "=" * 40 + "\n")
        fh.write(f"reason: {reason}\n\ndetail: {detail}\n\n")
        fh.write("traceback (if any):\n" + traceback.format_exc() + "\n")
    ws.write_status("failed", error=reason, detail=detail)
    return {"status": "failed", "reason": reason, "detail": detail}
