#!/usr/bin/env python
r"""
================================================================================
GLO-NCA V3 -- training performance profiler (measurement instrument, NOT prod).
================================================================================
Measures WHERE time goes in the V3 training path, on real or synthetic data,
with a controlled number of cases/resolution/workers/checkpointing. It is a
diagnostic tool: it imports the SAME model/agent/loss as the runner but does not
change the training methodology and is never invoked by train.py.

Usage (on a GPU box, e.g. Kaggle H100):
    python scripts/profile_v3_training.py --cases 4 --resolution 128 \
        --workers 4 --checkpointing true [--data-root /path/to/BraTS] \
        [--config configs/historical/v3_smoke_5epoch.yaml]

If --data-root is omitted (or no data found) it profiles on SYNTHETIC volumes of
the requested resolution so the MODEL/checkpointing cost can still be measured
without the dataset (clearly labelled 'synthetic' in the output).

It reports a per-stage timing breakdown (load / preprocess / H2D transfer /
forward / backward / optimizer / EMA / loss), throughput, NCA-update/forward/
backward CALL COUNTS (to prove no duplicate passes), and peak VRAM -- plus a
CHECKPOINT ON vs OFF comparison at the given resolution.
================================================================================
"""
import argparse, os, sys, time, statistics, contextlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _sync(dev):
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _bool(s): return str(s).lower() in ("1", "true", "yes", "on")


def main() -> int:
    ap = argparse.ArgumentParser(description="GLO-NCA V3 training profiler")
    ap.add_argument("--cases", type=int, default=4, help="number of cases to profile")
    ap.add_argument("--resolution", type=int, default=128, help="finest-level (L3) resolution")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (real data only)")
    ap.add_argument("--checkpointing", default="true", help="gradient checkpointing on/off")
    ap.add_argument("--config", default=os.path.join("configs", "v3_smoke_5epoch.yaml"))
    ap.add_argument("--data-root", default=None, help="BraTS root; omit to use synthetic volumes")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats per case (drop warmup)")
    args = ap.parse_args()

    import torch
    from src.experiment.config import load_config
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    ckpt_on = _bool(args.checkpointing)
    # Only override level3 when the caller asked for a resolution different from
    # the config's own, so the default path is byte-for-byte the previous one.
    _cfg_l3 = int(((cfg.raw.get("model", {}) or {}).get("level3", {}) or {})
                  .get("resolution", 128))
    R_OVERRIDE = int(args.resolution) if int(args.resolution) != _cfg_l3 else None

    print("-" * 66)
    print("GLO-NCA V3 PERFORMANCE PROFILE")
    print("-" * 66)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU              : {p.name}")
        print(f"VRAM             : {p.total_memory/1e9:.1f} GB")
    else:
        print("GPU              : CPU (no CUDA) -- timings not representative")
    print(f"config           : {args.config}")
    print(f"resolution (L3)  : {args.resolution}^3")
    print(f"checkpointing    : {ckpt_on}")
    print(f"cases            : {args.cases}")

    # ---- instrument the NCA update to COUNT calls (proves no duplicate passes) --
    from src.models import Model_BasicNCA3D as M
    counts = {"update": 0, "forward": 0}
    _orig_update = M.BasicNCA3D.update
    _orig_forward = M.BasicNCA3D.forward
    def _cu(self, *a, **k):
        counts["update"] += 1
        return _orig_update(self, *a, **k)
    def _cf(self, *a, **k):
        counts["forward"] += 1
        return _orig_forward(self, *a, **k)
    M.BasicNCA3D.update = _cu
    M.BasicNCA3D.forward = _cf

    # ---- build model (respect checkpointing flag override) --------------------
    # Force the requested checkpointing regardless of the config's memory flag.
    cfg.raw.setdefault("memory", {})["gradient_checkpointing"] = ckpt_on
    # Phase 6 fix: --resolution is documented as "finest-level (L3) resolution",
    # but it previously only sized the synthetic INPUT and the target. The model
    # was still built from the config's level3 (128), and since V3 resamples its
    # input to each level's own resolution, the output stayed 128^3 while the
    # target was R^3 -- so every R != 128 died with a size mismatch before any
    # timing was produced. That made the whole sweep unusable at exactly the
    # safe resolutions Phase 3 says we must use locally (128^3 fwd+bwd SPILLS on
    # this 6 GB GPU). Apply the override to the model too, so the flag means what
    # it says. Level 1/2 and the step counts are NOT touched; the default
    # (--resolution 128) reproduces the previous production geometry exactly.
    if R_OVERRIDE is not None:
        cfg.raw.setdefault("model", {}).setdefault("level3", {})["resolution"] = R_OVERRIDE
    model = build_v3_from_config(cfg, input_channels=4, output_channels=3, device=dev)
    model.train()
    pr = model.parameter_report()
    print(f"parameters       : {pr['total_parameters']} (expected 40656)")
    opt = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1e-4)

    R = args.resolution
    # ---- data source: REAL PIPELINE (actual Dataset+Agent+Experiment) or synthetic
    # For real profiling we build the SAME objects the runner builds (via _build_v3)
    # so the measured load/preprocess/patchify/DataLoader path is the production one,
    # not a simplified reimplementation. Synthetic is ONLY for isolated model cost.
    use_real = bool(args.data_root and os.path.isdir(args.data_root))
    load_times, prep_times = [], []
    batches = []
    real_ds = real_agent = None
    if use_real:
        print(f"data             : REAL PIPELINE ({args.data_root})")
        from src.experiment import runner as RUN
        # Build the production dataset/agent/experiment exactly as train.py does.
        ds, ca, agent, exp, _flat = RUN._build_dispatch(
            cfg, args.data_root, dev, epochs=1,
            out_model_dir=os.path.join(_ROOT, "experiments", "_profile_tmp"))
        # Use the model we already built (with the requested checkpointing flag).
        agent.model = [model]
        real_ds, real_agent = ds, agent
        # Materialise a train split so the dataset can serve real cases.
        from src.experiment import datasource as DS
        ids = DS.list_patients(args.data_root)[: max(args.cases, 1)]
        pmap = DS.case_path_map(args.data_root)
        def _entry(p): return (pmap.get(p, p), p, 0)
        exp.data_split.images["train"] = {p: {0: _entry(p)} for p in ids}
        exp.data_split.labels["train"] = {p: {0: _entry(p)} for p in ids}
        exp.set_model_state("train")
        ds.state = "train"
        ds.images_list = [_entry(p) for p in ids]
        # Time the REAL __getitem__ (load+crop+resize+label+patchify+norm) per case.
        import numpy as np
        for i in range(len(ids)):
            t = time.time()
            item = ds[i]                       # full production preprocessing
            load_times.append(time.time() - t)  # (cold: includes disk; warm: cache)
            img_id, img, label = item
            t = time.time()
            xb = torch.from_numpy(np.asarray(img)).unsqueeze(0).float()  # (1,X,Y,Z,4)
            prep_times.append(time.time() - t)
            batches.append(xb)
    else:
        print("data             : SYNTHETIC (isolated MODEL cost only; load/preproc n/a)")
        for _ in range(args.cases):
            batches.append(torch.randn(1, R, R, R, 4))

    # ---- timed loop: H2D / forward / backward / optimizer / EMA ---------------
    def measure():
        h2d, fwd, bwd, optt, emat = [], [], [], [], []
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for xb in batches:
            _sync(dev); t = time.time()
            x = xb.to(dev, non_blocking=True)
            tgt = torch.randint(0, 2, (1, R, R, R, 3), device=dev).float()
            _sync(dev); h2d.append(time.time() - t)

            opt.zero_grad(set_to_none=True)
            _sync(dev); t = time.time()
            out_cf = model(x)
            _sync(dev); fwd.append(time.time() - t)

            out = out_cf.permute(0, 2, 3, 4, 1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, tgt)
            _sync(dev); t = time.time()
            loss.backward()
            _sync(dev); bwd.append(time.time() - t)

            t = time.time(); opt.step(); _sync(dev); optt.append(time.time() - t)
            t = time.time()
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
            _sync(dev); emat.append(time.time() - t)
        return h2d, fwd, bwd, optt, emat

    # warmup (not timed)
    counts["update"] = 0; counts["forward"] = 0
    _ = measure()
    warm_updates = counts["update"]; warm_forwards = counts["forward"]

    # timed
    all_fwd, all_bwd, all_opt, all_ema, all_h2d = [], [], [], [], []
    counts["update"] = 0; counts["forward"] = 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    for _ in range(args.repeats):
        h2d, fwd, bwd, optt, emat = measure()
        all_h2d += h2d; all_fwd += fwd; all_bwd += bwd; all_opt += optt; all_ema += emat
    total_wall = time.time() - t_all

    def stats(xs):
        xs = sorted(xs)
        return (statistics.mean(xs), statistics.median(xs), min(xs), max(xs))

    n_batches = args.cases * args.repeats
    def line(name, xs):
        m, md, lo, hi = stats(xs)
        print(f"{name:<18}: mean {m*1000:8.1f} ms | median {md*1000:8.1f} | "
              f"min {lo*1000:7.1f} | max {hi*1000:7.1f}")

    print("-" * 66)
    print("PER-BATCH TIMING (per case, batch=1):")
    if use_real:
        line("load(cold)", load_times)
        line("preprocess", prep_times)
    line("H2D transfer", all_h2d)
    line("forward", all_fwd)
    line("backward", all_bwd)
    line("optimizer", all_opt)
    line("EMA", all_ema)
    per_case = statistics.mean([a + b + c + d + e for a, b, c, d, e in
                                zip(all_h2d, all_fwd, all_bwd, all_opt, all_ema)])
    print(f"{'TOTAL/case (gpu)':<18}: mean {per_case*1000:8.1f} ms")
    print("-" * 66)
    print("CALL COUNTS (timed phase; must be consistent, no duplicate passes):")
    exp_forward = args.cases * args.repeats * len(model.levels)
    steps_sum = sum(lv.nca_steps for lv in model.levels)
    # Phase 6 fix: with gradient checkpointing ON, torch.utils.checkpoint invokes
    # `update` TWICE per step -- once in forward, once recomputed during backward.
    # That is the whole point of checkpointing (trade compute for activation
    # memory), not a duplicate pass. The old line always printed the ckpt-OFF
    # expectation, so a correct ckpt-ON run looked like a 2x duplication bug.
    # Expect the multiplier explicitly instead.
    _mult = 2 if ckpt_on else 1
    exp_update = args.cases * args.repeats * steps_sum * _mult
    _why = ("x2 for checkpoint recompute in backward" if ckpt_on
            else "no checkpointing, single pass")
    print(f"  NCA.forward calls: {counts['forward']}  (expected {exp_forward} = cases*repeats*levels)")
    print(f"  NCA.update calls : {counts['update']}  (expected {exp_update} = "
          f"cases*repeats*sum(steps)={steps_sum} x{_mult}: {_why})")
    print(f"  levels/steps     : {[(lv.resolution, lv.nca_steps) for lv in model.levels]}")
    if dev.type == "cuda":
        print("-" * 66)
        print(f"peak allocated   : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"peak reserved    : {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    print("-" * 66)
    proj_epoch = per_case * 896
    print(f"projected train-only /epoch (896 cases): {proj_epoch/60:.1f} min "
          f"(excludes val + first-epoch cold load)")
    print(f"projected 300 epochs (train only)      : {proj_epoch*300/3600:.1f} h")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
