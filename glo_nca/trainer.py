r"""Training loop, best-model selection, final test, checkpointing and resume."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import checkpoint as ckpt
from .agent import Agent_GLO_NCA
from .config import REGIONS, GLO_NCA_Config
from .data import Dataset_BraTS_Foreground, find_data_root, make_split
from .evaluation import collect_probs, evaluate, improved_evaluation, metrics_from_cases, predict_case
from .experiment import Experiment
from .losses import FocalTverskyCELoss
from .model import GLO_NCA_Cell
from .reporting import final_report, make_figures


def set_seed(s):
    r"""Seed Python, NumPy and torch; GLO_DETERMINISTIC=1 also selects deterministic cuDNN kernels."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    if os.environ.get("GLO_DETERMINISTIC") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _write_status(out_dir, **fields):
    r"""Atomically write status.json for monitoring."""
    fields["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = os.path.join(out_dir, "status.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(fields, fh, indent=2)
    os.replace(tmp, os.path.join(out_dir, "status.json"))


def _write_history(out_dir, hist):
    r"""Write the per-epoch history to history.csv."""
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "loss", "lr", "val_mean", "val_WT", "val_TC", "val_ET"])
        for i in range(len(hist["epoch"])):
            w.writerow([hist["epoch"][i], hist["loss"][i], hist["lr"][i], hist["val_mean"][i],
                        hist["val_WT"][i], hist["val_TC"][i], hist["val_ET"][i]])


def with_weights(models, states, fn):
    r"""Run fn with the given weights loaded, then restore the live weights."""
    live = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]
    try:
        for m, s in zip(models, states):
            m.load_state_dict(s)
        return fn()
    finally:
        for m, s in zip(models, live):
            m.load_state_dict(s)


def build(C: GLO_NCA_Config, data_root: str, out_dir: str, splits) -> Tuple:
    r"""Dataset, the two NCA levels and the agent, wired to the given (train, val, test) split."""
    tr, va, te = splits
    DATA_ROOT = data_root
    # The configured GPU when one is present, otherwise the CPU.
    dev = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    device = C.DEVICE if dev.type == "cuda" else "cpu"
    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(out_dir, "m"),
        "device": device, "unlock_CPU": True,
        "optimizer": "adamw", "lr": C.LR_START, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": C.EPOCHS,
        "batch_size": C.BATCH_SIZE, "batch_duplication": 1,
        "channel_n": C.CHANNEL_N, "inference_steps": C.STEPS, "cell_fire_rate": C.FIRE_RATE,
        "input_channels": 4, "output_channels": 3, "hidden_size": C.HIDDEN,
        "train_model": 1, "use_attention": True,
        "input_size": C.INPUT_SIZE, "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "prioritize_masks": 0.7, "prioritize_region": C.PRIORITIZE_REGION,
        "crop_et_prob": C.CROP_ET_PROB,
    }]
    cache_dir = C.CACHE_DIR or os.path.join(os.path.dirname(os.path.abspath(out_dir)), ".cache")
    ds = Dataset_BraTS_Foreground(cache=C.CACHE, cache_dir=cache_dir)
    ds.MODALITIES = C.MODALITIES; ds.augment = C.AUGMENT; ds.ET_LABELS = tuple(C.ET_LABELS)
    ds.brain_mask = C.BRAIN_MASK
    ds.use_foreground_crop = C.USE_FOREGROUND_CROP; ds.use_nonzero_norm = C.USE_NONZERO_NORM
    ca = [GLO_NCA_Cell(C.CHANNEL_N, C.FIRE_RATE, dev, C.HIDDEN, kernel_size=7, input_channels=4,
                     use_attention=C.USE_SE, use_spatial=C.USE_SPATIAL, dropout=C.DROPOUT,
                     use_global=C.USE_GLOBAL, norm=C.NORM),
          GLO_NCA_Cell(C.CHANNEL_N, C.FIRE_RATE, dev, C.HIDDEN, kernel_size=3, input_channels=4,
                     use_attention=C.USE_SE, use_spatial=C.USE_SPATIAL, dropout=C.DROPOUT,
                     use_global=C.USE_GLOBAL, norm=C.NORM)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent); ds.set_experiment(exp)
    def entry(p):
        r"""Split entry for one case: (image, label, slice)."""
        return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")
    return ds, ca, agent, dev


def last_gain_epoch(hist, min_delta):
    r"""Last validated epoch whose mean Dice beat the running reference by more than min_delta."""
    ref, last = -math.inf, 0
    for e, v in zip(hist["epoch"], hist["val_mean"]):
        if not math.isnan(v) and v > ref + min_delta:
            ref, last = v, e
    return last


def _example_cases(agent, ds, per_case, C):
    r"""Best, median and worst test case by mean Dice, re-predicted for the example figure."""
    if not per_case:
        return None
    try:
        ranked = sorted(per_case, key=lambda row: np.mean([row[r] for r in REGIONS]))
        picks = [("Best", ranked[-1]), ("Median", ranked[len(ranked) // 2]), ("Worst", ranked[0])]
        names = [m.lower() for m in C.MODALITIES]
        channel = next((names.index(m) for m in ("t2f", "flair") if m in names), len(names) - 1)
        agent.exp.set_model_state("test")
        index = {key[0]: i for i, key in enumerate(ds.images_list)}
        examples = []
        with torch.no_grad():
            for label, row in picks:
                sample = ds[index[row["case"]]]
                case, prob, gt = predict_case(agent, torch.utils.data.default_collate([sample]),
                                              C.ENSEMBLE_N, C.USE_TTA)
                examples.append((label, case, sample[1][..., channel], gt.astype(np.float32), prob))
        return examples
    except Exception as exc:   # the example figure is optional
        print(f"WARNING: example segmentations were not prepared: {exc}")
        return None
    finally:
        agent.exp.set_model_state("train")


class EMA:
    r"""Exponential moving average of the model weights, updated after every optimiser step."""

    def __init__(self, models: List[torch.nn.Module], decay: float):
        r"""Start from a copy of the current weights."""
        self.models, self.decay = models, decay
        self.states = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]

    def update(self) -> None:
        r"""Blend the live weights into the average (floating-point tensors only)."""
        for e, m in zip(self.states, self.models):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    e[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def load(self, states) -> None:
        r"""Restore saved averages onto the devices of the current ones."""
        self.states = [{k: v.to(e[k].device) for k, v in se.items()} for se, e in zip(states, self.states)]


def _load_split(C: GLO_NCA_Config, data_root: str, out_dir: str, resume: bool) -> Tuple[list, list, list]:
    r"""The saved split when resuming, otherwise the seeded split of the case folders; saved as split.json."""
    split_path = os.path.join(out_dir, "split.json")
    if resume and os.path.exists(split_path):
        # Resume on the saved split, even if case folders were added or removed since.
        with open(split_path, encoding="utf-8") as fh:
            sp = json.load(fh)
        tr, va, te = sp["train"], sp["validation"], sp["test"]
    else:
        tr, va, te = make_split(data_root, C.SPLIT_SEED, C.N_PATIENTS)
    with open(split_path, "w") as fh:
        json.dump({"train": tr, "validation": va, "test": te, "seed": C.SPLIT_SEED}, fh, indent=2)
    return tr, va, te


def _print_settings(C: GLO_NCA_Config, splits) -> None:
    r"""Log the split sizes and the recipe."""
    tr, va, te = splits
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"GLO-NCA cascade: ch={C.CHANNEL_N} hidden={C.HIDDEN} steps={C.STEPS} fire={C.FIRE_RATE} "
          f"patch={C.INPUT_SIZE[-1]} beta={C.TVERSKY_BETA} aug={C.USE_AUG} ep={C.EPOCHS}", flush=True)
    if C.AUGMENT or C.REGION_WEIGHTS != [1.0, 1.0, 1.0] or C.improved_eval or C.VAL_EVERY > 1 \
            or C.EARLY_STOP_PATIENCE > 0:
        print(f"Options: augment={C.AUGMENT} region_weights={C.REGION_WEIGHTS} val_every={C.VAL_EVERY} "
              f"early_stop={C.EARLY_STOP_PATIENCE}/{C.EARLY_STOP_MIN_DELTA} "
              f"tune_thresholds={C.TUNE_THRESHOLDS} min_component={C.MIN_COMPONENT} "
              f"full_resolution={C.FULL_RESOLUTION_EVAL}", flush=True)


def _setup_training(C: GLO_NCA_Config, ca, agent, n_train: int) -> FocalTverskyCELoss:
    r"""Per-step cosine schedulers, loss options and element-wise gradient clipping; returns the loss."""
    total = C.EPOCHS * max(1, math.ceil(n_train / C.BATCH_SIZE))
    agent.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=C.LR_MIN)
                       for opt in agent.optimizer]
    loss_f = FocalTverskyCELoss(alpha=1 - C.TVERSKY_BETA, beta=C.TVERSKY_BETA,
                                gamma=C.FOCAL_GAMMA, ce_weight=0.5, bce_logits=C.BCE_LOGITS)
    if C.REGION_WEIGHTS != [1.0, 1.0, 1.0]:
        agent.region_weights = C.REGION_WEIGHTS
    agent.all_regions = C.ALL_REGION_LOSS
    # Gradient clipping via backward hooks (element-wise clamp; batch_step steps the optimizer).
    if C.GRAD_CLIP and C.GRAD_CLIP > 0:
        for m in ca:
            for p in m.parameters():
                if p.requires_grad:
                    p.register_hook(lambda g: torch.clamp(g, -C.GRAD_CLIP, C.GRAD_CLIP))
    return loss_f


def _train_epoch(C, ds, agent, loss_f, ema, ep, first_epoch, steps_per_epoch, t0) -> float:
    r"""One pass over the training split; returns the mean training loss."""
    losses = []
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=C.BATCH_SIZE,
                                         num_workers=C.NUM_WORKERS, pin_memory=True)
    for i, data in enumerate(loader):
        r = agent.batch_step(data, loss_f)
        if ema is not None:
            ema.update()
        if r:
            losses.append(sum(r.values()))
        if first_epoch and (i + 1) % 20 == 0:
            print(f"  [epoch {ep+1}] {i+1}/{steps_per_epoch} ({time.time()-t0:.0f}s)...", flush=True)
    if C.CROP_ET_PROB > 0 and agent.crop_stats[1]:
        print(f"  ET-positive crops this epoch: {agent.crop_stats[0]}/{agent.crop_stats[1]}", flush=True)
    agent.crop_stats = [0, 0]
    return float(np.mean(losses)) if losses else 0


def _validate(C, ds, ca, agent, ema) -> Dict[str, dict]:
    r"""Validation metrics of the live weights, or of the EMA weights with training.validate_ema."""
    score = lambda: evaluate(agent, ds, "val", workers=C.VAL_WORKERS, surface_hd95=C.SURFACE_HD95)
    if C.VALIDATE_EMA and ema is not None:
        return with_weights(ca, ema.states, score)   # the weights best.pth saves
    return score()


def _final_test(C, out_dir, ds, ca, agent, dev, best_path, hist, n_params, train_time, peak,
                best, stopped_early) -> dict:
    r"""Load best.pth, test (single pass, ensemble, tuned post-processing), write reports and figures."""
    ck = torch.load(best_path, map_location=dev, weights_only=False)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    last_epoch = hist["epoch"][-1] if hist["epoch"] else C.EPOCHS
    _write_status(out_dir, state="testing", epoch=last_epoch, epochs=C.EPOCHS, best=best)
    test_plain = evaluate(agent, ds, "test", ensemble=1, tta=False)
    per_case = []
    tuned = None
    if C.improved_eval:
        # One ensemble pass over test serves both the standard metric and the tuned evaluation.
        test_cases = collect_probs(agent, ds, "test", ensemble=C.ENSEMBLE_N, tta=C.USE_TTA,
                                   dtype=np.float32)
        test = metrics_from_cases(test_cases, per_case)
        # Post-processing is tuned on validation only; the test split is scored once.
        _write_status(out_dir, state="tuning", epoch=last_epoch, epochs=C.EPOCHS, best=best)
        val_cases = collect_probs(agent, ds, "val", ensemble=C.ENSEMBLE_N, tta=C.USE_TTA)
        tuned = improved_evaluation(C, ds, val_cases, test_cases)
    else:
        test = evaluate(agent, ds, "test", ensemble=C.ENSEMBLE_N, tta=C.USE_TTA, per_case=per_case)
    final_report(C, out_dir, ck["ep"], test_plain, test, hist, n_params, train_time, peak, tuned,
                 per_case=per_case)
    ens = f"Ensemble x{C.ENSEMBLE_N}" + (" + TTA" if C.USE_TTA else "")
    settings = {"Single pass": test_plain, ens: test}
    if tuned is not None:
        parts = ["tuned" if C.TUNE_THRESHOLDS else "", "full resolution" if C.FULL_RESOLUTION_EVAL else "",
                 "BraTS scoring" if C.BRATS_EMPTY else ""]
        settings["Post-processed (" + ", ".join(p for p in parts if p) + ")"] = tuned["test"]
    make_figures(out_dir, hist, ck["ep"], settings, per_case,
                 _example_cases(agent, ds, per_case, C), title=C.NAME)
    _write_status(out_dir, state="completed", epoch=last_epoch, epochs=C.EPOCHS, best=best,
                  best_epoch=ck["ep"], stopped_early=stopped_early)
    return {"status": "completed", "best_epoch": ck["ep"]}


def run(C: GLO_NCA_Config, out_dir: str, data_root: Optional[str] = None, resume: bool = False,
        stop_after_epoch: Optional[int] = None) -> dict:
    r"""Train, validate, checkpoint and test one run; supports resume and planned pauses."""
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pth")
    best_path = os.path.join(out_dir, "best.pth")
    if resume and not os.path.exists(last_path):
        raise FileNotFoundError(f"cannot resume: {last_path} not found")

    DATA_ROOT, nf = find_data_root(data_root or C.DATA_BASE)
    assert DATA_ROOT, f"BraTS not found under {data_root or C.DATA_BASE}"
    print(f"DATA_ROOT = {DATA_ROOT} ({nf} patients)")

    # Setup; the order of operations fixes the random sequence.
    set_seed(C.SEED)
    splits = _load_split(C, DATA_ROOT, out_dir, resume)
    _print_settings(C, splits)
    ds, ca, agent, dev = build(C, DATA_ROOT, out_dir, splits)
    spe = max(1, math.ceil(len(splits[0]) / C.BATCH_SIZE))
    loss_f = _setup_training(C, ca, agent, len(splits[0]))
    ema = EMA(ca, C.EMA_DECAY) if C.USE_EMA else None
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params, "| device:", dev, flush=True)

    hist = {"epoch": [], "loss": [], "lr": [], "val_mean": [], "val_WT": [], "val_TC": [], "val_ET": []}
    best = -1.0
    start_epoch, prior_seconds = 0, 0.0
    if resume:
        # Restore the exact state after the last saved epoch.
        ck = ckpt.load_last(last_path, ca, agent, dev)
        if ema is not None and ck["ema"] is not None:
            ema.load(ck["ema"])
        best, hist = ck["best"], ck["history"]
        start_epoch, prior_seconds = int(ck["epoch"]), float(ck["train_seconds"])
        ckpt.restore_rng(ck["rng"])
        print(f"RESUME: continuing after epoch {start_epoch} (best {best:.3f}); "
              f"next epoch {start_epoch + 1}", flush=True)
    if stop_after_epoch is not None and not start_epoch < stop_after_epoch <= C.EPOCHS:
        raise ValueError(f"--stop-after-epoch {stop_after_epoch} must be in "
                         f"({start_epoch}, {C.EPOCHS}]")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _write_status(out_dir, state="training", epoch=start_epoch, epochs=C.EPOCHS, best=best)

    stopped_early = None
    print("Loading + caching volumes (slow first pass)...", flush=True)
    for ep in range(start_epoch, C.EPOCHS):
        loss = _train_epoch(C, ds, agent, loss_f, ema, ep, ep == start_epoch, spe, t0)
        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        hist["epoch"].append(ep+1); hist["loss"].append(loss); hist["lr"].append(cur_lr)
        # Validate every VAL_EVERY epochs, and always on the last and the pause epoch.
        validate = ((ep + 1) % C.VAL_EVERY == 0 or ep + 1 == C.EPOCHS
                    or (stop_after_epoch is not None and ep + 1 >= stop_after_epoch))
        if validate:
            val = _validate(C, ds, ca, agent, ema)
            vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
            hist["val_mean"].append(vm)
            for r in REGIONS:
                hist[f"val_{r}"].append(val[r]["dice"])
            print(f"ep {ep+1}/{C.EPOCHS} | lr {cur_lr:.2e} | loss {loss:.3f} | "
                  f"val mean {vm:.3f} (WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} "
                  f"ET {val['ET']['dice']:.3f}) | {time.time()-t0:.0f}s", flush=True)
        else:
            vm = -1.0
            hist["val_mean"].append(float("nan"))
            for r in REGIONS:
                hist[f"val_{r}"].append(float("nan"))
            print(f"ep {ep+1}/{C.EPOCHS} | lr {cur_lr:.2e} | loss {loss:.3f} | "
                  f"validation skipped | {time.time()-t0:.0f}s", flush=True)
        if vm > best:
            best = vm
            save_states = ema.states if ema is not None else [m.state_dict() for m in ca]
            ckpt.save_atomic({"m": save_states, "ep": ep+1, "val_mean": vm}, best_path)
            print(f"   * new best {vm:.3f} saved", flush=True)

        # Full checkpoint, monitoring files and planned pause.
        seconds = prior_seconds + (time.time() - t0)
        ckpt.save_atomic(ckpt.build_last(ep + 1, ca, agent, ema.states if ema is not None else None,
                                         best, hist, seconds, C.raw), last_path)
        _write_history(out_dir, hist)
        _write_status(out_dir, state="training", epoch=ep + 1, epochs=C.EPOCHS, best=best)
        # Early stopping: no validation gain above min_delta for `patience` epochs.
        if validate and C.EARLY_STOP_PATIENCE > 0:
            since = ep + 1 - last_gain_epoch(hist, C.EARLY_STOP_MIN_DELTA)
            if since >= C.EARLY_STOP_PATIENCE:
                stopped_early = ep + 1
                print(f"EARLY STOP after epoch {ep+1}: no validation gain above "
                      f"{C.EARLY_STOP_MIN_DELTA} for {since} epochs", flush=True)
                break
        if stop_after_epoch is not None and ep + 1 >= stop_after_epoch:
            _write_status(out_dir, state="paused", epoch=ep + 1, epochs=C.EPOCHS, best=best,
                          next_epoch=ep + 2, test_evaluated=False)
            print(f"PAUSED after epoch {ep+1} of {C.EPOCHS}; resume with --resume {out_dir}",
                  flush=True)
            return {"status": "paused", "epoch": ep + 1}

    train_time = prior_seconds + (time.time() - t0)
    peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0
    return _final_test(C, out_dir, ds, ca, agent, dev, best_path, hist, n_params, train_time, peak,
                       best, stopped_early)
