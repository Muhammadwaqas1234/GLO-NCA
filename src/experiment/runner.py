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

from . import checkpoint as ckpt_io
from . import datasource, environment, graphs
from . import metrics_eval as ME
from . import reproducibility as repro
from . import statistics as STATS
from .config import Config
from .dataset_validation import validate_dataset, summarize
from .diagnostics import diagnose
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
# Training step (verbatim methodology from train.py, incl. the empty-region fix)
# --------------------------------------------------------------------------- #
def _clipped_batch_step(agent, data, loss_f, grad_clip: float) -> Dict[int, float]:
    data = agent.prepare_data(data)
    outputs, targets = agent.get_outputs(data)
    for opt in agent.optimizer:
        opt.zero_grad()
    loss = 0
    loss_ret: Dict[int, float] = {}
    empty_weight = 0.1  # small BCE on absent regions (prevents Tversky collapse)
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
        loss.backward()
        if grad_clip and grad_clip > 0:
            for net in agent.model:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        for opt, sch in zip(agent.optimizer, agent.scheduler):
            opt.step()
            sch.step()
    return loss_ret


# --------------------------------------------------------------------------- #
# Model / experiment construction from a Config
# --------------------------------------------------------------------------- #
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
        "patchify": True, "priotize_masks": 0.7, "prioritize_region": 2,
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
    return {
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
def run(cfg: Config, ws: Workspace, *, resume: bool, device_str: str = None) -> Dict[str, Any]:
    """Execute (or resume) a full experiment inside workspace ``ws``."""
    logger = get_logger(ws)
    ws.write_status("initializing", progress=0.0)
    logger.info("=" * 70)
    logger.info("GLO-NCA V2 experiment: %s", ws.experiment_id)
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

    # --- seed + config snapshot ---
    repro.set_all_seeds(cfg.seed)
    _copy_config_into(ws, cfg)

    # --- data root + validation ---
    data_root = datasource.resolve_data_root(cfg.get("dataset", "root"))
    if not data_root:
        return _fail(ws, logger, "dataset root not found",
                     "Set dataset.root in the config or $DATA_ROOT.")
    ws.write_status("validating", progress=0.0)
    n_pat = int(cfg.get("dataset", "number_of_patients", 0)) or None
    report = validate_dataset(data_root,
                              modalities=list(cfg.get("dataset", "modalities")),
                              limit=n_pat)
    ws.write_json(os.path.join("reports", "dataset_validation_report.json"), report)
    logger.info(summarize(report))
    if report["result"] != "PASS":
        return _fail(ws, logger, "dataset validation FAILED",
                     "See reports/dataset_validation_report.json. Training refuses "
                     "to start on invalid data (no silent skipping).")

    # --- split (reuse saved split on resume; else make + materialise) ---
    epochs = int(cfg.get("training", "epochs"))
    ds, ca, agent, exp, flat_cfg = _build(
        cfg, data_root, device, epochs, ws.path("checkpoints", "model_internal"))

    # Split resolution priority:
    #   1. resume        -> reuse the split already saved in this experiment dir
    #   2. data.split_file configured -> load that MASTER split (fail if missing;
    #      never silently regenerate -- prevents experimental drift across A0-A3)
    #   3. otherwise      -> deterministic seeded split (synthetic smoke tests)
    split_file = cfg.get("data", "split_file") if cfg.section("data") else None
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
    dataset_identity = {
        "dataset_root": data_root,
        "dataset_case_count": len(all_ids),
        "patient_id_hash": datasource.patient_id_hash(all_ids),
        "modalities": list(cfg.get("dataset", "modalities")),
        "split_source": split_meta["source"],
        "split_sha256": split_meta["split_sha256"],
        "split_file": split_meta.get("split_file"),
    }
    ws.write_json(os.path.join("config", "dataset_identity.json"), dataset_identity)

    def entry(p):
        return (p, p, 0)
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

    loss_f = FocalTverskyCELoss(
        alpha=1 - float(cfg.get("loss", "tversky_beta")),
        beta=float(cfg.get("loss", "tversky_beta")),
        gamma=float(cfg.get("loss", "focal_gamma")), ce_weight=0.5)

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

    grad_clip = (float(cfg.get("gradient", "max_norm"))
                 if bool(cfg.get("gradient", "clipping_enabled")) else 0.0)
    best_window = int(cfg.get("evaluation", "smoothing_window"))
    ckpt_freq = int(cfg.get("logging", "checkpoint_frequency", 10))

    info = _model_info(cfg, ca)
    logger.info("model params: total=%d trainable=%d (SE=%s, spatialGC=%s, levels=%d)",
                info["total_parameters"], info["trainable_parameters"],
                info["se_enabled"], info["spatial_gc_enabled"], info["nca_levels"])

    # --- resume: restore full state ---
    hist = {k: [] for k in ("epoch", "loss", "lr", "val_mean", "val_WT", "val_TC", "val_ET")}
    best, best_epoch, start_epoch = -1.0, 0, 0
    if resume and os.path.exists(ws.last_ckpt):
        ck = ckpt_io.load_checkpoint(ws.last_ckpt, map_location=device)
        ckpt_io.restore_into(ck, models=ca, optimizers=agent.optimizer,
                             schedulers=agent.scheduler)
        if ema is not None and ck.get("ema"):
            ema = [{k: v.to(device) for k, v in e.items()} for e in ck["ema"]]
        hist = ck.get("history", hist)
        best = ck.get("best_score", -1.0)
        best_epoch = ck.get("best_epoch", 0)
        start_epoch = int(ck.get("epoch", 0))
        repro.restore_rng_state(ck.get("rng_state", {}))
        logger.info("RESUME: experiment=%s ckpt=%s prev_epoch=%d best_epoch=%d "
                    "best=%.4f -> next_epoch=%d",
                    ws.experiment_id, ws.last_ckpt, start_epoch, best_epoch,
                    best, start_epoch + 1)
    elif resume:
        logger.warning("resume requested but no last.pth found; starting fresh.")

    train_csv = CSVLogger(ws.path("metrics", "train.csv"), TRAIN_CSV_FIELDS)
    val_csv = CSVLogger(ws.path("metrics", "validation.csv"), VAL_CSV_FIELDS)
    tb = TensorBoard(ws.path("tensorboard"), enabled=bool(cfg.get("logging", "tensorboard")))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ws.write_status("training", current_epoch=start_epoch, total_epochs=epochs,
                    progress=start_epoch / max(1, epochs), best_epoch=best_epoch,
                    best_score=best)
    logger.info("Loading + caching volumes (first pass is slow)...")

    workers = int(cfg.get("training", "workers"))

    def _worker_init(worker_id):
        s = (cfg.seed + worker_id) % (2 ** 32)
        np.random.seed(s)
        import random as _r
        _r.seed(s)

    # ------------------------------------------------------------- epoch loop
    for ep in range(start_epoch, epochs):
        ep_start = time.time()
        losses = []
        loader = torch.utils.data.DataLoader(
            ds, shuffle=True, batch_size=batch_size, num_workers=workers,
            pin_memory=(device.type == "cuda"), worker_init_fn=_worker_init)
        for data in loader:
            r = _clipped_batch_step(agent, data, loss_f, grad_clip)
            ema_update()
            if r:
                losses.append(sum(r.values()))

        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        val = ME.evaluate(agent, ds, "val")
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

        # --- last.pth (full state) every epoch + periodic snapshots ---
        full = ckpt_io.build_checkpoint(
            epoch=ep + 1, models=ca, optimizers=agent.optimizer,
            schedulers=agent.scheduler, ema=ema, best_score=best,
            best_epoch=best_epoch, history=hist, config=cfg.to_dict(),
            rng_state=repro.capture_rng_state())
        ckpt_io.save_checkpoint(ws.last_ckpt, full)
        if ckpt_freq and (ep + 1) % ckpt_freq == 0:
            ckpt_io.save_checkpoint(ws.periodic_ckpt(ep + 1), full)

        ws.write_status("training", current_epoch=ep + 1, total_epochs=epochs,
                        progress=(ep + 1) / max(1, epochs),
                        best_epoch=best_epoch, best_score=best)

    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # ------------------------------------------------------------- evaluation
    ws.write_status("evaluating", progress=1.0, best_epoch=best_epoch, best_score=best)
    if not os.path.exists(ws.best_ckpt):
        return _fail(ws, logger, "no best checkpoint produced",
                     "Training finished without saving a best model.")
    bw = ckpt_io.load_checkpoint(ws.best_ckpt, map_location=device)
    for m, sd in zip(ca, bw["m"]):
        m.load_state_dict(sd)

    tune = bool(cfg.get("evaluation", "tune_thresholds"))
    val_pairs = ME.collect_probs(agent, ds, "val")
    thresholds = ME.tune_thresholds(val_pairs) if tune else {r: 0.5 for r in REGIONS}
    test_pairs = ME.collect_probs(agent, ds, "test")
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
    per_case = ME.score_per_case(test_pairs, thresholds)
    _write_per_case_csv(ws, per_case)
    stats = STATS.summarize_per_case(per_case, n_boot=2000, seed=cfg.seed)
    ws.write_json(os.path.join("reports", "statistical_summary.json"), stats)
    _write_threshold_comparison(ws, val_pairs, test_pairs, thresholds)

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


def _write_threshold_comparison(ws, val_pairs, test_pairs, thresholds) -> None:
    """threshold_comparison.csv: default(0.5) vs tuned, on val and test. Test
    uses the FROZEN thresholds (never tuned on test)."""
    import csv
    half = {r: 0.5 for r in REGIONS}
    val_05, val_tuned = ME.score(val_pairs, half), ME.score(val_pairs, thresholds)
    test_05, test_tuned = ME.score(test_pairs, half), ME.score(test_pairs, thresholds)
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
                    "patient_id_hash": (dataset_identity or {}).get("patient_id_hash")},
        "split": {"source": (split_meta or {}).get("source"),
                  "split_sha256": (split_meta or {}).get("split_sha256"),
                  "split_file": (split_meta or {}).get("split_file"),
                  "split_version": (split_meta or {}).get("split_version")},
        "model": info,
        "training": {"epochs": int(cfg.get("training", "epochs")),
                     "batch_size": int(cfg.get("training", "batch_size")),
                     "patch_size": int(cfg.get("training", "patch_size")),
                     "augmentation": cfg.get("training", "augmentation"),
                     "input_size": cfg.input_size},
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
        "final_test": {r: test[r] for r in REGIONS},
        "status": status,
    }


def _fail(ws: Workspace, logger, reason: str, detail: str) -> Dict[str, Any]:
    logger.error("FAILURE: %s -- %s", reason, detail)
    with open(ws.path("reports", "failure_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("GLO-NCA V2 FAILURE REPORT\n" + "=" * 40 + "\n")
        fh.write(f"reason: {reason}\n\ndetail: {detail}\n\n")
        fh.write("traceback (if any):\n" + traceback.format_exc() + "\n")
    ws.write_status("failed", error=reason, detail=detail)
    return {"status": "failed", "reason": reason, "detail": detail}
