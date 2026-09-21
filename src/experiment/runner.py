r"""The GLO-NCA V2 training runner.

Orchestrates a full, fault-tolerant experiment while reusing the EXISTING model,
dataset, agent and loss code unchanged. The training methodology (Focal-Tversky
+ BCE, empty-region handling, EMA, grad-norm clip, cosine LR, smoothed
best-epoch, val-only threshold tuning, single-pass eval) is identical to the
original train.py -- this layer only adds experiment management around it.
"""
from __future__ import annotations

import math
import os
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import torch

# repo root (…/src/experiment/runner.py -> repo root), for resolving relative
# paths like a configured master split file on both the VM and locally.
_HERE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
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
from .run_diagnosis import diagnose
from . import postprocess as PP
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



# --------------------------------------------------------------------------- #
# DataLoader worker plumbing -- MODULE LEVEL for Windows `spawn` compatibility.
#
# These were previously nested inside `run()`. A closure and a locally-defined
# class cannot be pickled, so on Windows (where DataLoader workers are SPAWNED,
# not forked) any `workers > 0` run failed with:
#     AttributeError: Can't get local object 'run.<locals>._worker_init'
# Hoisting them to module scope makes them picklable. The seeding formula, the
# shuffling behaviour and the yielded (epoch, index) contract are UNCHANGED.
# --------------------------------------------------------------------------- #
class _WorkerInit:
    """Picklable replacement for the former `_worker_init` closure."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        s = (self.base_seed + worker_id) % (2 ** 32)
        np.random.seed(s)
        import random as _r
        _r.seed(s)


class _EpochSampler(torch.utils.data.Sampler):
    """Shuffles like `shuffle=True`, but yields (epoch, index) so the dataset
    can derive a deterministic per-(epoch, case) augmentation seed. The
    generator is seeded from (base_seed, epoch), so the ORDER is deterministic
    and differs per epoch -- reproducible, not repetitive."""

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


# --------------------------------------------------------------------------- #
# Training step (verbatim methodology from train.py, incl. the empty-region fix)
# --------------------------------------------------------------------------- #
def _clipped_batch_step(agent, data, loss_f, grad_clip: float,
                        empty_weight: float = 0.1,
                        amp_dtype=None) -> Dict[int, float]:
    """One training iteration.

    ``amp_dtype`` (``torch.bfloat16`` / ``torch.float16`` / ``None``) enables
    mixed precision for the MODEL FORWARD ONLY. The loss is always evaluated in
    FP32: PyTorch explicitly refuses to autocast ``binary_cross_entropy``
    (unsafe in reduced precision) and ``FocalTverskyCELoss`` applies sigmoid ->
    BCE internally. Closing autocast before the loss and casting the logits to
    float keeps the loss MATHEMATICS identical to the FP32 path, so this is a
    performance change, not a methodology change. ``None`` reproduces the
    previous behaviour exactly.
    """
    # Phase 2 profiling is OBSERVATIONAL: `prof` is a NullProfiler unless
    # explicitly enabled, and then every section is a no-op context with no CUDA
    # synchronisation. The order of operations, the maths and every
    # hyperparameter below are unchanged.
    from src.profiling import get_profiler
    prof = get_profiler()

    with prof.section("train/h2d_transfer", cuda=True):
        data = agent.prepare_data(data)
    with prof.section("train/forward", cuda=True):
        if amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs, targets = agent.get_outputs(data)
            outputs = outputs.float()      # loss runs in FP32 (see docstring)
        else:
            outputs, targets = agent.get_outputs(data)
    with prof.section("train/zero_grad"):
        for opt in agent.optimizer:
            opt.zero_grad(set_to_none=True)
    loss = 0
    loss_ret: Dict[int, float] = {}
    # empty_weight: small BCE on absent regions (prevents Tversky collapse).
    # Now supplied by the caller from config (default = the historical 0.1).
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
    if loss != 0:
        # Backward is timed separately from forward on purpose: with gradient
        # checkpointing ON, each NCA step is RECOMPUTED here, so backward is
        # expected to carry recompute cost that forward does not.
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


# --------------------------------------------------------------------------- #
# Model / experiment construction from a Config
# --------------------------------------------------------------------------- #
def _is_v3(cfg: Config) -> bool:
    return str(cfg.get("model", "version", "")).lower() == "v3"


GLO_NCA_PRODUCTION_IDENTITY = {
    "working_volume": 96,
    "level1_resolution": 48,
    "level2_resolution": 64,
    "level1_channels": 24,
    "level2_channels": 24,
    "level1_nca_steps": 15,
    "level2_nca_steps": 15,
    "level1_kernel": 5,
    "level2_kernel": 5,
    "spatial_kernel_size": 5,
    "level3_enabled": False,
    "use_attention": True,
    "use_spatial": True,
    "fusion_type": "concat",
    "hidden": 128,
    "batch_size": 1,
    "seed": 42,
    "patchify": False,
}


GLO_NCA_PRODUCTION_NAME = "glo_nca_production"


def _production_identity(cfg: Config) -> bool:
    """True only for the single authoritative GLO-NCA production experiment.

    Matched on the EXACT experiment name, not a prefix: other configs are also
    named "GLO-NCA-..." and must not inherit production invariants.
    """
    raw = str(cfg.get("experiment", "name", "") or getattr(cfg, "name", ""))
    return raw.lower().replace("-", "_") == GLO_NCA_PRODUCTION_NAME


def _build_dispatch(cfg: Config, data_root: str, device, epochs: int,
                    out_model_dir: str):
    """Version-aware construction. V2 (default) and V3 share the SAME dataset,
    Experiment, split handling and downstream runner; only the model + agent
    differ. Returns (ds, ca, agent, exp, flat_cfg)."""
    if _is_v3(cfg):
        return _build_v3(cfg, data_root, device, epochs, out_model_dir)
    return _build(cfg, data_root, device, epochs, out_model_dir)


def _build_v3(cfg: Config, data_root: str, device, epochs: int, out_model_dir: str):
    """Construct the unified V3 multi-level model + its runner adapter agent.

    Reuses the same Dataset, Experiment and (crucially) the same runner training
    loop as V2. The V3 model is a single nn.Module presented to the runner as a
    one-element list via ``Agent_GLO_NCA_V3`` so grad-clip, EMA, checkpoint and
    param-count code are all unchanged.
    """
    fire = float(cfg.get("model", "fire_rate", 0.6))
    aug = str(cfg.get("training", "augmentation"))
    patch = int(cfg.get("training", "patch_size"))  # finest = V3 output size

    # Flat config for the Experiment/dataset. V3 does its own seeding/upscaling
    # inside the model, so NCA-specific keys are given safe, inert values; the V3
    # agent never reads inference_steps/channel_n/input_size.
    # Patchify is configuration-driven. `data.training_patch.enabled` is the
    # single switch: when false no train patch size is set, so the dataset's
    # crop degenerates to the full working volume.
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
        "train_model": 0,  # V3 is a single model (no V2 multi-level cascade in the agent)
        "use_attention": bool(cfg.get("model", "use_attention", True)),
        # V3 output volume is a single cube (finest level); no two-level cascade.
        "input_size": [[patch, patch, patch]], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "foreground_crop": True, "nonzero_norm": True,
        "augment": aug != "none", "augment_level": ("light" if aug == "light" else "heavy"),
        # ET-aware patch sampling is a thesis choice, surfaced into config.
        "patchify": _patchify_enabled,
        "priotize_masks": float(cfg.get("sampling", "prioritize_probability", 0.7)),
        "prioritize_region": int(cfg.get("sampling", "prioritize_region", 2)),
    }]

    ds = Dataset_NiiGz_3D_BraTS()
    ds.MODALITIES = list(cfg.get("dataset", "modalities"))

    # --- optional GENUINE training patch (DEFAULT OFF) -----------------------
    # `training.patch_size` sets BOTH the resample target and the patch size, so
    # by default the crop is a no-op (patch == working volume). When
    # `data.training_patch.enabled` is true, the volume is resampled to the
    # larger `working_volume` and a REAL spatial crop of `patch_size` is taken,
    # using the existing ET-aware sampler. TRAIN-ONLY: `patchify_multimodal` is
    # gated on `state == "train"`, so validation/test stay full-volume.
    # Omitting the block reproduces the previous behaviour exactly, which is why
    # the frozen thesis reference is unaffected.
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

    # --- optional deterministic preprocessing cache (DEFAULT OFF) ------------
    # Caches ONLY the deterministic head of __getitem__ (load -> crop ->
    # resample -> label conversion). All stochastic work (patchify,
    # augmentation, per-(epoch,case) RNG) still runs every epoch, and cache
    # lookup is RNG-neutral -- verified by scripts/test_preprocess_cache.py.
    _cache_cfg = (cfg.section("data") or {}).get("cache") or {}
    if bool(_cache_cfg.get("enabled", False)):
        from src.datasets.preprocess_cache import PreprocessCache
        _cdir = _cache_cfg.get("directory") or os.path.join(_HERE_ROOT, ".cache",
                                                            "preprocessed")
        if not os.path.isabs(_cdir):
            _cdir = os.path.join(_HERE_ROOT, _cdir)
        # The cache stores the DETERMINISTIC head of __getitem__, whose output is
        # the resampled WORKING VOLUME -- not the training patch. When
        # `data.training_patch` is enabled the dataset resamples to
        # `working_volume` and the (stochastic) crop to `patch_size` happens
        # afterwards, so sizing the cache with `patch` would make every lookup
        # fail `_validate` (100% miss: silently slow, never incorrect).
        # `_cache_size` is the actual dataset.size set above.
        _cache_size = int(config[0]["input_size"][0][0])
        ds.set_preprocess_cache(PreprocessCache(
            _cdir, dataset_root=data_root,
            modalities=list(cfg.get("dataset", "modalities")),
            size=(_cache_size, _cache_size, _cache_size),
            crop_fg=True, rescale=True, enabled=True))

    # GLO-NCA Global Context + Multi-Level Fusion. When `model.global_context` is
    # present the model keeps its GLOBAL level on the whole working volume and
    # may compute the high-resolution level over a region of interest
    # (`roi_fraction`). With the default `roi_fraction: 1.0` this is
    # mathematically identical to the reference forward, so configs without the
    # block -- including the frozen thesis reference -- are unaffected.
    if (cfg.raw.get("model", {}) or {}).get("global_context") is not None:
        from src.models.Model_GLO_NCA_GlobalContext import build_glo_nca_global_context
        v3_model = build_glo_nca_global_context(cfg, input_channels=4,
                                                output_channels=3, device=device)
    else:
        v3_model = build_v3_from_config(cfg, input_channels=4, output_channels=3,
                                        device=device)
    ca = [v3_model]  # presented as a one-element list to the shared runner
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
        BasicNCA3D(ch, fire, device, hidden, kernel_size=7, input_channels=4,
                   use_attention=use_attn, use_spatial=use_spatial, dropout=dropout),
        BasicNCA3D(ch, fire, device, hidden, kernel_size=3, input_channels=4,
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
        # V3: a single unified multi-level model. Report the per-level structure
        # from the model itself (measured, not hard-coded).
        m = ca[0]
        pr = m.parameter_report() if hasattr(m, "parameter_report") else {}
        active_levels = [lv for lv in ("level1", "level2", "level3")
                         if bool((cfg.get("model", lv, {}) or {}).get("enabled", True))]
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


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(cfg: Config, ws: Workspace, *, resume: bool, device_str: str = None,
        extend_to: int = None, lr_policy: str = "freeze",
        extension_reason: str = None) -> Dict[str, Any]:
    """Execute (or resume) a full experiment inside workspace ``ws``."""
    logger = get_logger(ws)
    ws.write_status("initializing", progress=0.0)
    # Explicit lifecycle. `write_status` remains for the existing dashboard;
    # `state.json` is the machine-readable record of what actually happened.
    run_state = sm.TrainingState.load(ws.path("reports"),
                                      run_id=ws.experiment_id)
    gpu_diag = diag.GPUDiagnostics(enabled=True, min_interval_s=10.0)
    logger.info("=" * 70)
    logger.info("GLO-NCA %s experiment: %s",
                "V3" if _is_v3(cfg) else "V2", ws.experiment_id)
    logger.info("config: %s", cfg.path)

    # --- environment / git / gpu capture ---
    env_summary = environment.write_environment(ws)
    logger.info("torch %s | cuda_available=%s | %s",
                env_summary["gpu"].get("torch_version"),
                env_summary["gpu"].get("cuda_available"),
                env_summary["gpu"].get("device_name", "CPU"))
    logger.info(repro.describe())

    # --- device ---
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not used (device=%s). GPU is expected for the full "
                       "run; CPU is intended only for smoke tests.", device)

    # --- Phase 2 profiling (OBSERVATIONAL, disabled unless `profiling.enabled`)
    # Returns a NullProfiler by default, so production runs are unaffected.
    _prof = _configure_profiler(cfg, out_dir=ws.root)
    _mem = MemorySampler(enabled=getattr(_prof, "memory_enabled", False))         if _prof.enabled else None
    if _prof.enabled:
        logger.info("PROFILING ENABLED (observational): warmup=%d profiled=%d "
                    "-- this run is bounded and is NOT a training campaign",
                    _prof.warmup, _prof.iterations)

    # --- seed + config snapshot ---
    repro.set_all_seeds(cfg.seed)
    _copy_config_into(ws, cfg)

    # --- data root + validation ---
    data_root = datasource.resolve_data_root(cfg.get("dataset", "root"))
    if not data_root:
        return _fail(ws, logger, "dataset root not found",
                     "Set dataset.root in the config or $DATA_ROOT.")
    ws.write_status("validating", progress=0.0)
    if run_state.can_transition(sm.PREFLIGHT):
        run_state.transition(sm.PREFLIGHT, "dataset + identity gates")
    n_pat = int(cfg.get("dataset", "number_of_patients", 0)) or None

    # --- explicit operational data-quality policy (canonical split preserved) --
    # Optional file listing, per case id, any stray segmentation labels that are
    # accepted for THAT case only, with a written reason. Absent file => no
    # tolerances (validator stays fail-closed exactly as before). A file that
    # exists but is malformed, or names unknown case ids, fails LOUDLY -- a
    # broken policy must never silently degrade to "no policy".
    _dq_section = cfg.section("data") if cfg.section("data") else {}
    dq_path = _dq_section.get("quality_policy_file", "__unset__")
    try:
        if dq_path is None:
            # Explicit opt-out (`quality_policy_file: null`): no tolerances, so
            # the validator stays fail-closed on ANY label outside {0,1,2,3,4}.
            # Used by harness configs that discover only a subset of cases and
            # therefore cannot satisfy a policy naming specific case ids.
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

    try:
        report = validate_dataset(data_root,
                                  modalities=list(cfg.get("dataset", "modalities")),
                                  limit=n_pat, policy=dq_policy)
    except DataQualityPolicyError as exc:
        return _fail(ws, logger, "data-quality policy does not match the dataset",
                     str(exc))
    ws.write_json(os.path.join("reports", "dataset_validation_report.json"), report)
    logger.info(summarize(report))
    if report["result"] != "PASS":
        return _fail(ws, logger, "dataset validation FAILED",
                     "See reports/dataset_validation_report.json. Training refuses "
                     "to start on invalid data (no silent skipping).")

    # --- split (reuse saved split on resume; else make + materialise) ---
    epochs = int(cfg.get("training", "epochs"))
    ds, ca, agent, exp, flat_cfg = _build_dispatch(
        cfg, data_root, device, epochs, ws.path("checkpoints", "model_internal"))

    # Split resolution priority:
    #   1. resume        -> reuse the split already saved in this experiment dir
    #   2. data.split_file configured -> load that MASTER split (fail if missing;
    #      never silently regenerate -- prevents experimental drift across A0-A3)
    #   3. otherwise      -> deterministic seeded split (synthetic smoke tests)
    split_file = cfg.get("data", "split_file") if cfg.section("data") else None

    # Phase 2 (P1): a config with NO `data:` section fell through to the seeded
    # synthetic split and trained on it silently. For a real-dataset run that
    # would quietly abandon the canonical subject-disjoint split -- invalidating
    # the V2-vs-V3 comparison with nothing but a manifest field to reveal it.
    # Any V3 run on real data must name its split file explicitly.
    # A config may opt out ONLY by saying so explicitly (`data.allow_seeded_split:
    # true`), which the synthetic local-smoke config does.
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
        # resolve relative to CWD, then repo root, so it works on VM + local
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

    # --- automatic dataset identity (read-only provenance) ------------------
    # Reuses the fingerprint helpers; never modifies the dataset. Stored so the
    # researcher does not need a separate manual command per experiment.
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
        # Operational data-quality policy actually in force for THIS run, plus
        # the counts an examiner needs: canonical vs final operational.
        "data_quality_policy": (dq_policy.summary() if dq_policy else None),
        "tolerated_cases": report.get("tolerated_cases") or {},
        "operational_exclusions": (list(dq_policy.excluded) if dq_policy else []),
    }
    ws.write_json(os.path.join("config", "dataset_identity.json"), dataset_identity)

    # Resolve each case id to its path RELATIVE to the dataset root so nested
    # cohorts (e.g. BraTS-MET's 'UCSD - Training/') load correctly. For a flat
    # dataset rel_path == case id, so the entries are identical to before.
    path_map = datasource.case_path_map(data_root)

    def entry(p):
        return (path_map.get(p, p), p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")

    # --- optimizer schedule (cosine over the whole run) ---
    batch_size = int(cfg.get("training", "batch_size"))
    spe = max(1, math.ceil(len(tr) / batch_size))
    total_steps = epochs * spe
    lr_min = float(cfg.get("optimizer", "minimum_learning_rate"))
    agent.scheduler = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
        for opt in agent.optimizer
    ]

    # Phase 2 (§5 centralisation): ce_weight and the empty-region BCE weight are
    # thesis hyperparameters that were previously hard-coded and therefore did
    # NOT appear in the saved per-run config record. They are now read from the
    # config, with defaults equal to the historical constants so behaviour is
    # unchanged for every existing config.
    ce_weight = float(cfg.get("loss", "ce_weight", 0.5))
    empty_weight = float(cfg.get("loss", "empty_region_bce_weight", 0.1))
    loss_f = FocalTverskyCELoss(
        alpha=1 - float(cfg.get("loss", "tversky_beta")),
        beta=float(cfg.get("loss", "tversky_beta")),
        gamma=float(cfg.get("loss", "focal_gamma")), ce_weight=ce_weight)

    # --- EMA ---
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
        """Load EMA weights into the live models; return the raw states.

        Validation must score the SAME weights that are saved as best, so the
        selected checkpoint is the model that produced the selected metric.
        """
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

    grad_clip = (float(cfg.get("gradient", "max_norm"))
                 if bool(cfg.get("gradient", "clipping_enabled")) else 0.0)
    best_window = int(cfg.get("evaluation", "smoothing_window"))
    ckpt_freq = int(cfg.get("logging", "checkpoint_frequency", 10))

    # --- overfitting protection ---------------------------------------------
    # Model selection and stopping both read the SAME validation metric: the
    # `best_window`-epoch rolling mean of validation mean foreground Dice
    # (mean over WT/TC/ET). Smoothing is what stops a single lucky epoch from
    # being selected. The test split is never involved.
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

    # --- mixed precision (performance only; default OFF) ---------------------
    # `performance.precision`: "fp32" (default) | "bf16" | "fp16". Applies to the
    # MODEL FORWARD only -- the loss always runs in FP32 (see
    # `_clipped_batch_step`), so the loss mathematics is unchanged. fp16 needs a
    # GradScaler, which this training step does not use, so it is refused rather
    # than silently producing unscaled-gradient behaviour.
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

    # --- HD95 schedule (performance only; default = every epoch) -------------
    # HD95 was measured at ~97% of validation scoring cost. Dice and mIoU still
    # run EVERY epoch, so model selection (3-epoch rolling validation Dice) and
    # threshold tuning are untouched. HD95 is reported on the configured epoch
    # interval and ALWAYS on the final epoch; the frozen-test evaluation is
    # unaffected. 1 = previous behaviour.
    hd95_every = int((_perf.get("hd95_every_epochs") or 1))
    if hd95_every < 1:
        return _fail(ws, logger, "invalid hd95_every_epochs",
                     f"performance.hd95_every_epochs={hd95_every}; must be >= 1.")
    if hd95_every > 1:
        logger.info("HD95 computed every %d epochs (Dice/mIoU every epoch; "
                    "model selection and test evaluation unchanged)", hd95_every)
    # Carried-forward HD95 for epochs where it is skipped. Initialised to NaN so
    # a resumed run that skips HD95 on its first epoch reports NaN rather than
    # raising -- NaN is already the established "not available" value here.
    _last_hd95 = {r: float("nan") for r in REGIONS}

    info = _model_info(cfg, ca)
    logger.info("model params: total=%d trainable=%d (SE=%s, spatialGC=%s, levels=%d)",
                info["total_parameters"], info["trainable_parameters"],
                info["se_enabled"], info["spatial_gc_enabled"], info["nca_levels"])

    # --- resume: restore full state ---
    hist = {k: [] for k in ("epoch", "loss", "lr", "val_mean", "val_WT", "val_TC", "val_ET")}
    best, best_epoch, start_epoch = -1.0, 0, 0
    ext_plan = None          # set only when --extend-to is validated below
    if resume and os.path.exists(ws.last_ckpt):
        ck = ckpt_io.load_checkpoint(ws.last_ckpt, map_location=device)
        ckpt_io.restore_into(ck, models=ca, optimizers=agent.optimizer,
                             schedulers=agent.scheduler)
        # Phase 2 (P2): verify the checkpoint was written by a compatible config.
        # A model-shape change raises on load_state_dict, but hyperparameter
        # drift (e.g. a different `training.epochs`, which changes the cosine
        # T_max) previously passed silently and produced an LR curve matching
        # NEITHER config. Compare the fields that define the training schedule.
        prev_cfg = ck.get("config") or {}
        if prev_cfg:
            _crit = [("training", "epochs"), ("training", "batch_size"),
                     ("training", "patch_size"), ("optimizer", "learning_rate"),
                     ("optimizer", "minimum_learning_rate"),
                     ("loss", "tversky_beta"), ("loss", "focal_gamma"),
                     ("ema", "decay"), ("experiment", "seed")]
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
                # Previously silent: the freshly-initialised EMA (a copy of the
                # restored weights) was kept with no indication in the log.
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

        # --- explicit continuation past the planned budget ------------------
        # Validated AFTER the state is restored, so the plan is checked
        # against what was actually loaded rather than what was requested.
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
        # Fail closed. Silently treating this as a fresh 301-epoch run would
        # produce a model that LOOKS like a continuation but shares nothing
        # with the parent.
        return _fail(ws, logger, "extension refused: no parent checkpoint",
                     f"--extend-to {extend_to} needs a complete checkpoint at "
                     f"{ws.last_ckpt}. An extension continues an existing run; "
                     f"it never starts one.")

    train_csv = CSVLogger(ws.path("metrics", "train.csv"), TRAIN_CSV_FIELDS)
    val_csv = CSVLogger(ws.path("metrics", "validation.csv"), VAL_CSV_FIELDS)
    tb = TensorBoard(ws.path("tensorboard"), enabled=bool(cfg.get("logging", "tensorboard")))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Identity block embedded in every ranked best checkpoint: code revision,
    # config fingerprint, split identity and the software/hardware the weights
    # were produced on. Built once -- none of it changes during a run.
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

    # PHASE 2 (P1, two linked fixes -- see PHASE2_PRODUCTION_HARDENING.md).
    #
    # 1) RNG DIVERSITY. The old seed was `cfg.seed + worker_id`, with no epoch
    #    term. Because a fresh DataLoader was built every epoch, `worker_init_fn`
    #    re-ran each epoch with the SAME handful of seeds, so the augmentation
    #    and patch-sampling draw sequence repeated identically for all 300
    #    epochs. Mixing the epoch in gives each epoch a distinct, still fully
    #    DETERMINISTIC stream (same seed + same epoch -> same augmentations), so
    #    the run stays reproducible while actually varying across epochs.
    #
    # 2) PREPROCESSING CACHE. With workers>0 each worker holds its own copy of
    #    the dataset, so `Data_Container.set_data` wrote into a copy that was
    #    destroyed when the loader was exhausted. `persistent_workers=True` keeps
    #    the workers (and their caches) alive across epochs, so each case is
    #    decompressed + resampled once per worker instead of every epoch.
    #
    # These interact in a way that needs care. With `persistent_workers=True`
    # PyTorch calls `worker_init_fn` ONCE per worker (at the first iteration),
    # and each worker holds its own COPY of the dataset -- so neither an epoch
    # term inside `worker_init_fn` nor a parent-side `ds.set_epoch()` would ever
    # reach the workers. The epoch is therefore carried by the SAMPLER, whose
    # indices genuinely travel from parent to worker every epoch: the dataset
    # reseeds its per-item augmentation RNG from (base_seed, epoch, index),
    # which works identically with or without persistent workers.
    _worker_init = _WorkerInit(cfg.seed)

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

    # ------------------------------------------------------------- epoch loop
    for ep in range(start_epoch, epochs):
        ep_start = time.time()
        losses = []
        gpu_diag.set_phase(diag.ACTIVE_GPU)
        gpu_diag.sample(epoch=ep + 1, step=0)
        # Epoch travels to the workers via the sampler's yielded indices, so it
        # reaches worker dataset copies even when they persist across epochs.
        sampler.set_epoch(ep)
        if not persistent:  # workers==0: no caching benefit, rebuild per epoch
            loader = torch.utils.data.DataLoader(ds, **loader_kwargs)
        # Phase 2: measure DataLoader WAIT (time the training loop spends
        # blocked on the input pipeline) separately from the compute that
        # follows. Iterating the loader manually is the only way to attribute
        # that wait; the sequence of batches is byte-identical to `for data in
        # loader`. All of this is inert when profiling is disabled.
        # Bounded torch.profiler window (opt-in). Deliberately covers only a
        # few iterations: an unbounded profiler over a whole epoch produces
        # multi-GB traces and distorts the very timings being measured.
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
                                        amp_dtype=amp_dtype)
                with _prof.section("train/ema", cuda=True):
                    ema_update()
            if _prof.enabled and _mem is not None and _prof.is_profiled_iteration():
                for _k, _v in _mem.sample().items():
                    if _v is not None:
                        _prof.record(f"memory/{_k}", float(_v))
            if r:
                losses.append(sum(r.values()))
            if _tp is not None:
                _tp.step()
            _batch_idx += 1
            # Bounded profiling run: stop after warmup + profiled iterations so a
            # profiling session can never silently become a training run.
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
            # HD95 is ~97% of validation scoring cost. Dice and mIoU are computed
            # EVERY epoch (model selection and threshold tuning are untouched);
            # HD95 is computed on the configured interval and ALWAYS on the final
            # epoch. On skipped epochs the last computed HD95 is carried forward
            # for reporting and clearly flagged in `hd95_fresh`. `metrics_eval`
            # is NOT modified -- the same `collect_probs`/`score` are reused.
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

            # Ranked best-K set alongside the single best file. Selected on the
            # SAME metric, so best_1 and best.pth always agree. Weights are
            # never averaged: the ranked files exist for inspection.
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

        # --- early stopping (VALIDATION metric only; test is never consulted) -
        _stop = stopper.update(vm_smooth, ep + 1)
        if stopper.enabled:
            logger.info("   %s", stopper.status())

        # --- last.pth (full state) every epoch + periodic snapshots ---
        full = ckpt_io.build_checkpoint(
            epoch=ep + 1, models=ca, optimizers=agent.optimizer,
            schedulers=agent.scheduler, ema=ema, best_score=best,
            best_epoch=best_epoch, history=hist, config=cfg.to_dict(),
            rng_state=repro.capture_rng_state())
        # Patience must survive a preemption. Without this the counter resets
        # on resume and a plateaued run trains for another full `patience`
        # epochs after every restart.
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

        # Stop AFTER the full checkpoint is written, so the run is resumable
        # from exactly where it stopped. The best checkpoint is untouched:
        # stopping never overwrites it, and evaluation below loads it.
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

    # Terminal lifecycle state. An extended run must not be recorded as the
    # original budget, so the two completions are distinct states.
    if not stopper.should_stop:
        if ext_plan is not None:
            run_state.transition(sm.COMPLETED_EXTENSION,
                                 f"reached extension target epoch {epochs}",
                                 epochs_run=epochs_run)
        elif run_state.can_transition(sm.COMPLETED_300):
            run_state.transition(sm.COMPLETED_300,
                                 f"planned budget of {epochs} epochs reached",
                                 epochs_run=epochs_run)

    # Diagnostics and timing are written whatever the outcome: a run that
    # stopped early is exactly when this evidence is most useful.
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

    # ------------------------------------------------------------- evaluation
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

    _log_table(logger, f"FINAL TEST @ 0.5 (best @ epoch {bw['ep']})", test_05)
    _log_table(logger, "FINAL TEST @ tuned thresholds "
               + ", ".join(f"{r}={thresholds[r]:.2f}" for r in REGIONS), test)
    logger.info("train time %.0fs | peak VRAM %.2f GB | params %d",
                train_time, peak, info["total_parameters"])

    diag = diagnose(hist, bw["ep"], test_05, test, thresholds, peak)
    logger.info("DIAGNOSTIC: %s", diag["verdict"])
    for line in diag["lines"]:
        logger.info("  %s", line)

    # ------------------------------------------------------------- packaging
    ws.write_status("packaging", progress=1.0)
    metrics_test = {"epoch": bw["ep"], **{f"dice_{r}": test[r]["dice"] for r in REGIONS},
                    **{f"iou_{r}": test[r]["iou"] for r in REGIONS},
                    **{f"hd95_{r}": test[r]["hd95"] for r in REGIONS},
                    "dice_mean": float(np.mean([test[r]["dice"] for r in REGIONS]))}
    CSVLogger(ws.path("metrics", "test.csv"), list(metrics_test.keys())).append(metrics_test)

    results = {"test": test, "test_at_0.5": test_05, "thresholds": thresholds,
               "history": hist, "best_epoch": bw["ep"], "diagnostics": diag,
               "params": info["total_parameters"], "train_time": train_time,
               "peak_vram": peak}
    ws.write_json(os.path.join("reports", "results.json"), results)
    ws.write_json(os.path.join("reports", "diagnostic_report.json"), diag)
    with open(ws.path("reports", "diagnostic_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("GLO-NCA V2 DIAGNOSTIC REPORT\n" + "=" * 40 + "\n")
        for line in diag["lines"]:
            fh.write(line + "\n")
        fh.write("-" * 40 + "\nVERDICT: " + diag["verdict"] + "\n")
    _write_thesis_csv(ws, test, test_05, thresholds, info)

    # --- per-case metrics + statistical summary (thesis stats) --------------
    # Same metric definitions as `score`; purely for reporting mean/median/std/
    # bootstrap-CI. Uses the FROZEN validation-tuned thresholds on the test set.
    # Phase 2 (memory): write the threshold comparison FIRST -- it is the last
    # consumer of val_pairs -- then release val_pairs before the per-case /
    # bootstrap stage. Identical inputs, identical outputs, identical file
    # contents; only the peak host RAM drops (val+test were both held live).
    # Reuse the test scores computed above (pure function of the same cached
    # pairs + thresholds) instead of recomputing HD95 over every test case.
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

    # --- Phase 2: emit profiling artifacts (no-op unless profiling enabled) ---
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
                         dataset_identity=dataset_identity, split_meta=split_meta)
    ws.write_manifest(manifest)
    ws.write_status("completed", progress=1.0, best_epoch=bw["ep"],
                    best_score=best, test_mean=metrics_test["dice_mean"])
    logger.info("Experiment COMPLETED -> %s", ws.root)
    return {"status": "completed", "results": results, "workspace": ws.root}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _copy_config_into(ws: Workspace, cfg: Config) -> None:
    import shutil
    dest = ws.path("config", "config.yaml")
    # On resume the config already IS the workspace copy -- don't copy onto self.
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
    """One row per (case_index, region) with the per-case dice/iou/hd95 at the
    frozen tuned thresholds -- the raw material for mean/median/std/CI."""
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
    """threshold_comparison.csv: default(0.5) vs tuned, on val and test. Test
    uses the FROZEN thresholds (never tuned on test).

    PHASE 4 (unnecessary work): the caller has ALREADY scored ``test_pairs`` at
    0.5 and at the tuned thresholds. ``ME.score`` is a pure function of
    (pairs, thresholds) -- verified -- so recomputing it here produced
    byte-identical numbers at the cost of a second full HD95 pass over every
    test case. HD95 runs a distance transform per region per case and measured
    ~10.2 s/case at 128^3, so the duplicate test scoring cost ~2 x 198 cases
    ~= 67 min of pure recomputation at the end of a run.

    The already-computed results are now passed in and reused. Falls back to
    recomputing if a caller does not supply them, so behaviour is unchanged for
    any other call site. The CSV contents are identical either way.
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
              dataset_identity=None, split_meta=None) -> Dict[str, Any]:
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
        "scheduler": "CosineAnnealingLR",
        "loss": {"type": "FocalTverskyCELoss",
                 "tversky_beta": float(cfg.get("loss", "tversky_beta")),
                 "focal_gamma": float(cfg.get("loss", "focal_gamma"))},
        "ema": {"enabled": bool(cfg.get("ema", "enabled")),
                "decay": float(cfg.get("ema", "decay"))},
        "seed": cfg.seed, "split_seed": cfg.split_seed,
        "gpu": env_summary.get("gpu"),
        "software": env_summary.get("software"),
        "best_epoch": best_epoch,
        # Budget accounting. `planned_max_epochs` is what the experiment was
        # configured for; `actual_completed_epochs` is what it ran. They differ
        # when early stopping fires or when the run was explicitly extended,
        # and the thesis record must not conflate the two.
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
    """Persist whatever profiling data exists, even on a failed run.

    A profiling harness that loses its measurements because the run ended early
    is useless -- the measurements are the deliverable, not the model."""
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
        fh.write("GLO-NCA V2 FAILURE REPORT\n" + "=" * 40 + "\n")
        fh.write(f"reason: {reason}\n\ndetail: {detail}\n\n")
        fh.write("traceback (if any):\n" + traceback.format_exc() + "\n")
    ws.write_status("failed", error=reason, detail=detail)
    return {"status": "failed", "reason": reason, "detail": detail}
