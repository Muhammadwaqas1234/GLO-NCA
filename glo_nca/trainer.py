r"""Training loop, best-model selection, final test, checkpointing and resume."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
import types

import numpy as np
import torch

from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.losses.LossFunctions import FocalTverskyCELoss
from src.models.Model_GLO_NCA_Cell import GLO_NCA_Cell
from src.utils.Experiment import Experiment

from . import checkpoint as ckpt
from .config import REGIONS, GLO_NCA_Config
from .data import Dataset_BraTS_Foreground, find_data_root, make_split
from .evaluation import collect_probs, evaluate, improved_evaluation
from .reporting import final_report


def set_seed(s):
    r"""Seed Python, NumPy and torch."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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


def build(C: GLO_NCA_Config, data_root: str, out_dir: str, splits) -> tuple:
    r"""Dataset, the two NCA levels and the agent, wired to the given (train, val, test) split."""
    tr, va, te = splits
    DATA_ROOT = data_root
    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(out_dir, "m"),
        "device": C.DEVICE, "unlock_CPU": True,
        "optimizer": "adamw", "lr": C.LR_START, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": C.EPOCHS,
        "batch_size": C.BATCH_SIZE, "batch_duplication": 1,
        "channel_n": C.CHANNEL_N, "inference_steps": C.STEPS, "cell_fire_rate": C.FIRE_RATE,
        "input_channels": 4, "output_channels": 3, "hidden_size": C.HIDDEN,
        "train_model": 1, "use_attention": True,
        "input_size": C.INPUT_SIZE, "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.7, "prioritize_region": C.PRIORITIZE_REGION,
    }]
    ds = Dataset_BraTS_Foreground(); ds.MODALITIES = C.MODALITIES; ds.augment = C.AUGMENT
    ds.use_foreground_crop = C.USE_FOREGROUND_CROP; ds.use_nonzero_norm = C.USE_NONZERO_NORM
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [GLO_NCA_Cell(C.CHANNEL_N, C.FIRE_RATE, dev, C.HIDDEN, kernel_size=7, input_channels=4,
                     use_attention=True, use_spatial=C.USE_SPATIAL, dropout=C.DROPOUT),
          GLO_NCA_Cell(C.CHANNEL_N, C.FIRE_RATE, dev, C.HIDDEN, kernel_size=3, input_channels=4,
                     use_attention=True, use_spatial=C.USE_SPATIAL, dropout=C.DROPOUT)]
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


def _weighted_batch_step(weights):
    r"""Agent_GLO_NCA_Multi.batch_step with a per-region loss weight."""
    def batch_step(self, data, loss_f):
        r"""One optimisation step with per-region loss weights."""
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        for m in range(self.exp.get_from_config('train_model')+1):
            self.optimizer[m].zero_grad()
        loss = 0
        loss_ret = {}
        for m in range(outputs.shape[-1]):
            if 1 in targets[..., m]:
                loss_loc = weights[m] * loss_f(outputs[..., m], targets[..., m])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        if loss != 0:
            loss.backward()
            for m in range(self.exp.get_from_config('train_model')+1):
                self.optimizer[m].step()
                self.scheduler[m].step()
        return loss_ret
    return batch_step


def run(C: GLO_NCA_Config, out_dir: str, data_root: str = None, resume: bool = False,
        stop_after_epoch: int = None) -> dict:
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
    tr, va, te = make_split(DATA_ROOT, C.SPLIT_SEED, C.N_PATIENTS)
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"GLO-NCA cascade: ch={C.CHANNEL_N} hidden={C.HIDDEN} steps={C.STEPS} fire={C.FIRE_RATE} "
          f"patch={C.INPUT_SIZE[-1]} beta={C.TVERSKY_BETA} aug={C.USE_AUG} ep={C.EPOCHS}", flush=True)
    if C.AUGMENT or C.REGION_WEIGHTS != [1.0, 1.0, 1.0] or C.improved_eval or C.VAL_EVERY > 1 \
            or C.EARLY_STOP_PATIENCE > 0:
        print(f"Options: augment={C.AUGMENT} region_weights={C.REGION_WEIGHTS} val_every={C.VAL_EVERY} "
              f"early_stop={C.EARLY_STOP_PATIENCE}/{C.EARLY_STOP_MIN_DELTA} "
              f"tune_thresholds={C.TUNE_THRESHOLDS} min_component={C.MIN_COMPONENT} "
              f"full_resolution={C.FULL_RESOLUTION_EVAL}", flush=True)
    with open(os.path.join(out_dir, "split.json"), "w") as fh:
        json.dump({"train": tr, "validation": va, "test": te, "seed": C.SPLIT_SEED}, fh, indent=2)

    ds, ca, agent, dev = build(C, DATA_ROOT, out_dir, (tr, va, te))

    spe = max(1, math.ceil(len(tr) / C.BATCH_SIZE)); total = C.EPOCHS * spe
    agent.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=C.LR_MIN)
                       for opt in agent.optimizer]

    loss_f = FocalTverskyCELoss(alpha=1 - C.TVERSKY_BETA, beta=C.TVERSKY_BETA,
                                gamma=C.FOCAL_GAMMA, ce_weight=0.5)
    if C.REGION_WEIGHTS != [1.0, 1.0, 1.0]:
        agent.batch_step = types.MethodType(_weighted_batch_step(C.REGION_WEIGHTS), agent)

    # Gradient clipping via backward hooks (element-wise clamp; batch_step steps the optimizer).
    if C.GRAD_CLIP and C.GRAD_CLIP > 0:
        for m in ca:
            for p in m.parameters():
                if p.requires_grad:
                    p.register_hook(lambda g: torch.clamp(g, -C.GRAD_CLIP, C.GRAD_CLIP))

    # EMA: shadow copy of weights, updated after every optimizer step.
    ema = None
    if C.USE_EMA:
        ema = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in ca]
    def ema_update():
        r"""Blend the live weights into the EMA copy."""
        if ema is None:
            return
        for e, m in zip(ema, ca):
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    e[k].mul_(C.EMA_DECAY).add_(v.detach(), alpha=1 - C.EMA_DECAY)
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params, "| device:", dev, flush=True)

    hist = {"epoch": [], "loss": [], "lr": [], "val_mean": [], "val_WT": [], "val_TC": [], "val_ET": []}
    best = -1.0
    start_epoch, prior_seconds = 0, 0.0

    # Resume: restore the exact state after the last saved epoch.
    if resume:
        ck = ckpt.load_last(last_path, ca, agent, dev)
        if ema is not None and ck["ema"] is not None:
            ema = [{k: v.to(e[k].device) for k, v in se.items()} for se, e in zip(ck["ema"], ema)]
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
        losses = []
        loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=C.BATCH_SIZE,
                                             num_workers=C.NUM_WORKERS, pin_memory=True)
        for i, data in enumerate(loader):
            r = agent.batch_step(data, loss_f)
            ema_update()
            if r:
                losses.append(sum(r.values()))
            if ep == start_epoch and (i + 1) % 20 == 0:
                print(f"  [epoch {ep+1}] {i+1}/{spe} ({time.time()-t0:.0f}s)...", flush=True)
        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        hist["epoch"].append(ep+1); hist["loss"].append(float(np.mean(losses)) if losses else 0)
        hist["lr"].append(cur_lr)
        # Validate every VAL_EVERY epochs, and always on the last and the pause epoch.
        validate = ((ep + 1) % C.VAL_EVERY == 0 or ep + 1 == C.EPOCHS
                    or (stop_after_epoch is not None and ep + 1 >= stop_after_epoch))
        if validate:
            val = evaluate(agent, ds, "val", workers=C.VAL_WORKERS)
            vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
            hist["val_mean"].append(vm)
            for r in REGIONS:
                hist[f"val_{r}"].append(val[r]["dice"])
            print(f"ep {ep+1}/{C.EPOCHS} | lr {cur_lr:.2e} | loss {hist['loss'][-1]:.3f} | "
                  f"val mean {vm:.3f} (WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} "
                  f"ET {val['ET']['dice']:.3f}) | {time.time()-t0:.0f}s", flush=True)
        else:
            vm = -1.0
            hist["val_mean"].append(float("nan"))
            for r in REGIONS:
                hist[f"val_{r}"].append(float("nan"))
            print(f"ep {ep+1}/{C.EPOCHS} | lr {cur_lr:.2e} | loss {hist['loss'][-1]:.3f} | "
                  f"validation skipped | {time.time()-t0:.0f}s", flush=True)
        if vm > best:
            best = vm
            save_states = ema if ema is not None else [m.state_dict() for m in ca]
            ckpt.save_atomic({"m": save_states, "ep": ep+1, "val_mean": vm}, best_path)
            print(f"   * new best {vm:.3f} saved", flush=True)

        # Full checkpoint, monitoring files and planned pause.
        seconds = prior_seconds + (time.time() - t0)
        ckpt.save_atomic(ckpt.build_last(ep + 1, ca, agent, ema, best, hist, seconds, C.raw),
                         last_path)
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

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    # Final test: plain, then with pseudo-ensemble + TTA.
    last_epoch = hist["epoch"][-1] if hist["epoch"] else C.EPOCHS
    _write_status(out_dir, state="testing", epoch=last_epoch, epochs=C.EPOCHS, best=best)
    test_plain = evaluate(agent, ds, "test", ensemble=1, tta=False)
    test = evaluate(agent, ds, "test", ensemble=C.ENSEMBLE_N, tta=C.USE_TTA)
    tuned = None
    if C.improved_eval:
        # Post-processing is tuned on validation only; the test split is scored once.
        _write_status(out_dir, state="tuning", epoch=last_epoch, epochs=C.EPOCHS, best=best)
        val_cases = collect_probs(agent, ds, "val", ensemble=C.ENSEMBLE_N, tta=C.USE_TTA)
        test_cases = collect_probs(agent, ds, "test", ensemble=C.ENSEMBLE_N, tta=C.USE_TTA)
        tuned = improved_evaluation(C, ds, val_cases, test_cases)
    final_report(C, out_dir, ck["ep"], test_plain, test, hist, n_params, train_time, peak, tuned)
    _write_status(out_dir, state="completed", epoch=last_epoch, epochs=C.EPOCHS, best=best,
                  best_epoch=ck["ep"], stopped_early=stopped_early)
    return {"status": "completed", "best_epoch": ck["ep"]}
