r"""Measure model size, inference latency and peak memory of a trained run (batch one)."""
import argparse
import json
import os
import sys
import time


def main(argv=None) -> int:
    r"""Time predict_case on a synthetic volume of the working size, after warm-up runs."""
    ap = argparse.ArgumentParser(description="Inference latency and memory of a trained run.")
    ap.add_argument("--run", required=True, help="run folder with config.yaml and best.pth")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"], help="cpu forces CPU timing")
    ap.add_argument("--ensemble-n", type=int, default=None, help="stochastic passes (default: the run's)")
    ap.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--output", help="write the results as JSON")
    args = ap.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import torch

    from glo_nca.config import REGIONS, load_config
    from glo_nca.evaluation import predict_case
    from glo_nca.trainer import build, set_seed

    C = load_config(os.path.join(args.run, "config.yaml"))
    set_seed(C.SEED)
    _, ca, agent, dev = build(C, args.run, args.run, ([], [], []))
    best = os.path.join(args.run, "best.pth")
    ck = torch.load(best, map_location=dev, weights_only=False)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    agent.exp.set_model_state("test")
    ensemble = args.ensemble_n if args.ensemble_n is not None else C.ENSEMBLE_N
    tta = C.USE_TTA and not args.no_tta
    size = C.INPUT_SIZE[-1]
    x = np.random.default_rng(0).standard_normal((1, *size, len(C.MODALITIES))).astype(np.float32)
    data = (["_bench_0"], torch.from_numpy(x), torch.zeros((1, *size, len(REGIONS))))

    def once():
        r"""One full prediction (all passes and flips), synchronised on the GPU."""
        with torch.no_grad():
            predict_case(agent, data, ensemble, tta)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        once()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    times, rss = [], []
    for _ in range(args.repeats):
        t = time.perf_counter()
        once()
        times.append(time.perf_counter() - t)
        if proc is not None:
            rss.append(proc.memory_info().rss)
    res = {"device": str(dev), "ensemble_n": ensemble, "tta": tta, "working_size": size,
           "parameters": int(sum(p.numel() for m in ca for p in m.parameters())),
           "checkpoint_mb": round(os.path.getsize(best) / 2**20, 3),
           "latency_median_s": round(float(np.median(times)), 4),
           "latency_p95_s": round(float(np.percentile(times, 95)), 4), "repeats": args.repeats,
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None,
           "process_rss_mb": round(max(rss) / 2**20, 1) if rss else None}
    print(json.dumps(res, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
