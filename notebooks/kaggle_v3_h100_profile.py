r"""
================================================================================
GLO-NCA V3 — Kaggle H100 measurement notebook (TEST 1..10). MEASUREMENT ONLY.
================================================================================
Run each SECTION as a Kaggle cell on an H100 (or A100) with the real BraTS-MET
data attached. This notebook DOES NOT modify any production source and DOES NOT
run 5-epoch training. It measures the real pipeline and writes machine-readable
results to  experiments/profiling/v3_h100/{h100_profile.json,h100_profile.md}.

Setup expected on Kaggle:
  REPO = "/kaggle/working/GLO-NCA"     # repo checkout / uploaded tarball
  DATA = "/kaggle/input/brats-met/MICCAI-LH-BraTS2025-MET-Challenge-Training"
Adjust REPO/DATA below, then Run All. Do NOT run the commented 5-epoch cell here.

Every printed conclusion is labelled FACT / MEASURED / CALCULATED / HYPOTHESIS.
================================================================================
"""
# ============================ CONFIG (edit me) ================================
REPO = "/kaggle/working/GLO-NCA"
DATA = "/kaggle/input/brats-met/MICCAI-LH-BraTS2025-MET-Challenge-Training"
CONFIG = "configs/v3_kaggle_5epoch.yaml"
OUT_DIR = "experiments/profiling/v3_h100"

import os, sys, json, time, random, statistics, contextlib
sys.path.insert(0, REPO); os.chdir(REPO)
os.makedirs(OUT_DIR, exist_ok=True)
import torch, numpy as np
RESULTS = {"labels": "FACT/MEASURED/CALCULATED/HYPOTHESIS"}

def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def _save():
    with open(os.path.join(OUT_DIR, "h100_profile.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print("saved", os.path.join(OUT_DIR, "h100_profile.json"))

# ============================ TEST 1 — environment ============================
def test1_env():
    r = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
    if r["cuda"]:
        p = torch.cuda.get_device_properties(0)
        r.update(gpu=p.name, gpu_count=torch.cuda.device_count(),
                 vram_gb=round(p.total_memory/1e9, 1),
                 cuda=torch.version.cuda)
    RESULTS["test1_env"] = r; print("TEST1 env [MEASURED]:", r); return r

# ============================ TEST 2 — model ==================================
def test2_model():
    from src.experiment.config import load_config
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    cfg = load_config(CONFIG); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = build_v3_from_config(cfg, 4, 3, dev); pr = m.parameter_report()
    r = {"params": pr["total_parameters"], "expected": 40656,
         "match": pr["total_parameters"] == 40656, "by_level": pr["by_level"],
         "levels": [(l.resolution, l.nca_steps) for l in m.levels]}
    RESULTS["test2_model"] = r; print("TEST2 model [MEASURED/FACT]:", r); return cfg, m, dev

# ============================ TEST 3 — real dataset ===========================
def test3_dataset(cfg, dev):
    from src.experiment import runner as RUN
    from src.experiment import datasource as DS
    ds, ca, agent, exp, _ = RUN._build_dispatch(cfg, DATA, dev, epochs=1,
        out_model_dir=os.path.join(REPO, "experiments", "_prof_tmp"))
    ids = DS.list_patients(DATA); pmap = DS.case_path_map(DATA)
    def _e(p): return (pmap.get(p, p), p, 0)
    sub = ids[:5]
    exp.data_split.images["train"] = {p: {0: _e(p)} for p in sub}
    exp.data_split.labels["train"] = {p: {0: _e(p)} for p in sub}
    exp.set_model_state("train"); ds.state = "train"; ds.images_list = [_e(p) for p in sub]
    t = time.time(); item = ds[0]; dt = time.time() - t
    img_id, img, label = item
    r = {"case": str(img_id), "getitem_s": round(dt, 3),
         "img_shape": list(np.asarray(img).shape), "label_shape": list(np.asarray(label).shape),
         "label_unique": sorted(np.unique(np.asarray(label)).tolist())[:10]}
    RESULTS["test3_dataset"] = r; print("TEST3 dataset [MEASURED/FACT]:", r)
    return ds, agent, sub

# ==================== TEST 4/8 — full real iteration + stages =================
def test4_8_iteration(cfg, m, dev, ds, agent, sub, ckpt):
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    cfg.raw.setdefault("memory", {})["gradient_checkpointing"] = ckpt
    model = build_v3_from_config(cfg, 4, 3, dev); model.train()
    agent.model = [model]
    opt = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stages = {k: [] for k in ("getitem","h2d","forward","loss","backward","opt","ema","total")}
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for i in range(len(sub)):
        t0 = time.time()
        t = time.time(); item = ds[i]; stages["getitem"].append(time.time()-t)
        _id, img, label = item
        x = torch.from_numpy(np.asarray(img)).unsqueeze(0).float()
        tgt = torch.from_numpy(np.asarray(label)).unsqueeze(0).float()
        _sync(); t = time.time(); x = x.to(dev, non_blocking=True); tgt = tgt.to(dev, non_blocking=True); _sync(); stages["h2d"].append(time.time()-t)
        opt.zero_grad(set_to_none=True)
        _sync(); t = time.time(); out_cf = model(x); _sync(); stages["forward"].append(time.time()-t)
        out = out_cf.permute(0,2,3,4,1)
        t = time.time(); loss = torch.nn.functional.binary_cross_entropy_with_logits(out, tgt); _sync(); stages["loss"].append(time.time()-t)
        _sync(); t = time.time(); loss.backward(); _sync(); stages["backward"].append(time.time()-t)
        t = time.time(); opt.step(); _sync(); stages["opt"].append(time.time()-t)
        t = time.time()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        _sync(); stages["ema"].append(time.time()-t)
        stages["total"].append(time.time()-t0)
    med = {k: round(statistics.median(v)*1000, 2) for k, v in stages.items()}
    r = {"ckpt": ckpt, "stage_ms_median": med,
         "peak_alloc_gb": round(torch.cuda.max_memory_allocated()/1e9,2) if torch.cuda.is_available() else None,
         "peak_reserved_gb": round(torch.cuda.max_memory_reserved()/1e9,2) if torch.cuda.is_available() else None,
         "samples_per_s": round(1.0/statistics.median(stages["total"]),3)}
    RESULTS[f"test4_8_iter_ckpt_{ckpt}"] = r
    print(f"TEST4/8 iteration ckpt={ckpt} [MEASURED]:", r); return r

# ==================== TEST 5/6 — checkpoint OFF vs ON =========================
def test5_6_checkpoint(cfg, m, dev, ds, agent, sub):
    off = test4_8_iteration(cfg, m, dev, ds, agent, sub, ckpt=False)
    on  = test4_8_iteration(cfg, m, dev, ds, agent, sub, ckpt=True)
    def tot(d): return sum(d["stage_ms_median"][k] for k in ("forward","backward"))
    r = {"off": off, "on": on,
         "slowdown_pct_ckpt_on": round(100*(tot(on)-tot(off))/max(tot(off),1e-6),1),
         "vram_reduction_pct": (round(100*(off["peak_alloc_gb"]-on["peak_alloc_gb"])/off["peak_alloc_gb"],1)
                                if off["peak_alloc_gb"] else None),
         "h100_fits_without_ckpt": (off["peak_alloc_gb"] is not None and off["peak_alloc_gb"] < 70)}
    RESULTS["test5_6_checkpoint"] = r; print("TEST5/6 checkpoint [MEASURED/CALCULATED]:", r); return r

# ==================== TEST 7 — DataLoader workers + cache =====================
def test7_dataloader(ds, sub):
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS  # noqa
    res = {}
    for w in (0, 2, 4):
        try:
            loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=w,
                        shuffle=False, persistent_workers=(w>0))
            times = []
            for epoch in (1, 2):
                t0 = time.time(); first = None
                for j, _ in enumerate(loader):
                    if j == 0: first = time.time()-t0
                    if j >= min(4, len(sub)-1): break
                times.append({"epoch": epoch, "first_batch_s": round(first,3),
                              "n_batches_timed": j+1, "total_s": round(time.time()-t0,3)})
            res[f"workers_{w}"] = times
        except Exception as e:
            res[f"workers_{w}"] = {"error": str(e)[:200]}
    RESULTS["test7_dataloader"] = res
    print("TEST7 dataloader [MEASURED]:", json.dumps(res, indent=2))
    print("  -> compare epoch1 vs epoch2 first_batch/total: if epoch2 ~= epoch1,"
          " cache did NOT persist across epochs (confirms audit A1) [interpretation]")
    return res

# ==================== TEST 9 — per-level NCA timing ===========================
def test9_per_level(cfg, dev):
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    cfg.raw.setdefault("memory", {})["gradient_checkpointing"] = False
    model = build_v3_from_config(cfg, 4, 3, dev); model.train()
    # wrap each level's nca.forward with a CUDA-synced timer (no math change)
    times = {f"L{i+1}": [] for i in range(len(model.ncas))}
    orig = [nca.forward for nca in model.ncas]
    def mk(i, f):
        def w(*a, **k):
            _sync(); t=time.time(); out=f(*a, **k); _sync(); times[f"L{i+1}"].append(time.time()-t); return out
        return w
    for i, nca in enumerate(model.ncas): nca.forward = mk(i, orig[i])
    x = torch.randn(1, cfg.raw["model"]["level3"]["resolution"], cfg.raw["model"]["level3"]["resolution"],
                    cfg.raw["model"]["level3"]["resolution"], 4, device=dev)
    for _ in range(3):  # warmup + timed
        out = model(x); loss = out.float().mean(); loss.backward(); model.zero_grad()
    r = {k: round(statistics.median(v)*1000,2) for k, v in times.items() if v}
    tot = sum(r.values()) or 1
    r_pct = {k: round(100*v/tot,1) for k, v in r.items()}
    RESULTS["test9_per_level"] = {"level_ms_median": r, "level_pct": r_pct}
    print("TEST9 per-level [MEASURED]:", RESULTS["test9_per_level"]); return r

# ==================== TEST 10 — patchify RNG-preserving equivalence ===========
def _current_patchify(img, label, size, prioritize, region):
    contains_mask = prioritize is not None and (random.uniform(0,1) < prioritize)
    px=py=pz=0; fb=None
    for _ in range(50):
        px=random.randint(0,img.shape[0]-size[0]); py=random.randint(0,img.shape[1]-size[1]); pz=random.randint(0,img.shape[2]-size[2])
        if not contains_mask: break
        if label[px:px+size[0],py:py+size[1],pz:pz+size[2],region].max()>0: break
        if region!=0 and fb is None:
            if label[px:px+size[0],py:py+size[1],pz:pz+size[2],0].max()>0: fb=(px,py,pz)
    else:
        if fb is not None: px,py,pz=fb
    return img[px:px+size[0],py:py+size[1],pz:pz+size[2],:], label[px:px+size[0],py:py+size[1],pz:pz+size[2],:]

def _candidate_patchify(img, label, size, prioritize, region):
    """RNG-PRESERVING fast path for patch==volume. Reproduces the EXACT random.*
    draw sequence of _current_patchify but computes label max ONCE (patch==volume
    => every iteration's slice is the full volume, identical .max())."""
    if tuple(img.shape[:3]) != tuple(size):
        return _current_patchify(img, label, size, prioritize, region)
    contains_mask = prioritize is not None and (random.uniform(0,1) < prioritize)
    region_has = label[...,region].max() > 0
    if not contains_mask or region_has:
        random.randint(0,0); random.randint(0,0); random.randint(0,0)  # 1 iteration
    else:
        for _ in range(50):
            random.randint(0,0); random.randint(0,0); random.randint(0,0)  # full budget
    return img, label

def test10_patchify(ds, sub):
    R = ds.size[0]; size = tuple(ds.size)
    region = ds.exp.get_from_config('prioritize_region') or 2
    prio = ds.exp.get_from_config('priotize_masks')
    rows = []; safe = True
    for i in range(len(sub)):
        _id, img, label = ds[i]
        img = np.asarray(img); label = np.asarray(label)
        et_present = bool(label[..., region].max() > 0)
        for seed in (0, 42):
            random.seed(seed); a1, b1 = _current_patchify(img, label, size, prio, region); st1 = random.getstate()
            random.seed(seed); a2, b2 = _candidate_patchify(img, label, size, prio, region); st2 = random.getstate()
            ok = bool(np.array_equal(a1,a2) and np.array_equal(b1,b2) and st1==st2)
            safe = safe and ok
            rows.append({"case": str(_id), "et_present": et_present, "seed": seed,
                         "patch_eq_volume": tuple(img.shape[:3])==size, "identical_out_and_rng": ok})
    r = {"rows": rows, "verdict": ("SAFE TO APPLY (real-data, RNG identical)" if safe
                                   else "NOT SAFE — RNG semantics change")}
    RESULTS["test10_patchify"] = r
    print("TEST10 patchify [MEASURED]:", r["verdict"])
    for row in rows: print("   ", row)
    return r

# ================================ DRIVER ======================================
def run_all():
    test1_env()
    cfg, m, dev = test2_model()
    ds, agent, sub = test3_dataset(cfg, dev)
    test5_6_checkpoint(cfg, m, dev, ds, agent, sub)   # includes TEST4/8 both modes
    test7_dataloader(ds, sub)
    test9_per_level(cfg, dev)
    test10_patchify(ds, sub)
    # write markdown summary
    with open(os.path.join(OUT_DIR, "h100_profile.md"), "w") as fh:
        fh.write("# GLO-NCA V3 H100 profile (machine-generated)\n\n")
        fh.write("```json\n" + json.dumps(RESULTS, indent=2, default=str) + "\n```\n")
    _save()
    print("\nDONE. Paste experiments/profiling/v3_h100/h100_profile.json back to Claude.")
    print("Do NOT run 5-epoch training here — that is a separate, later step.")

# ==================== SYNTHETIC checkpoint bench (NO dataset needed) ==========
def synth_checkpoint(cfg, dev, R=128, n=5, warm=2):
    """TEST 5/6 on synthetic 128^3 tensors: checkpoint OFF vs ON — peak VRAM,
    fwd/bwd/total, samples/s. Answers 'does V3 fit on H100 without checkpointing'
    and 'how much does checkpointing cost' WITHOUT any BraTS data."""
    from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    out = {}
    for ckpt in (False, True):
        cfg.raw.setdefault("memory", {})["gradient_checkpointing"] = ckpt
        cfg.raw["model"]["level3"]["resolution"] = R
        m = build_v3_from_config(cfg, 4, 3, dev); m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1.6e-3, weight_decay=1e-4)
        x = torch.randn(1, R, R, R, 4, device=dev)
        tgt = torch.randint(0, 2, (1, R, R, R, 3), device=dev).float()
        f_t, b_t, tot = [], [], []
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        for i in range(n + warm):
            _sync(); t0 = time.time()
            opt.zero_grad(set_to_none=True)
            _sync(); t = time.time(); out_cf = m(x); _sync(); f = time.time()-t
            o = out_cf.permute(0,2,3,4,1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(o, tgt)
            _sync(); t = time.time(); loss.backward(); _sync(); b = time.time()-t
            opt.step(); _sync()
            if i >= warm:
                f_t.append(f); b_t.append(b); tot.append(time.time()-t0)
        out[str(ckpt)] = {
            "fwd_ms": round(statistics.median(f_t)*1000, 2),
            "bwd_ms": round(statistics.median(b_t)*1000, 2),
            "total_ms": round(statistics.median(tot)*1000, 2),
            "samples_per_s": round(1.0/statistics.median(tot), 3),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated()/1e9, 2) if torch.cuda.is_available() else None,
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved()/1e9, 2) if torch.cuda.is_available() else None,
        }
    off, on = out["False"], out["True"]
    r = {"R": R, "off": off, "on": on,
         "checkpoint_slowdown_pct": round(100*(on["total_ms"]-off["total_ms"])/max(off["total_ms"],1e-6), 1),
         "vram_reduction_pct": (round(100*(off["peak_alloc_gb"]-on["peak_alloc_gb"])/off["peak_alloc_gb"], 1)
                                if off["peak_alloc_gb"] else None),
         "h100_fits_without_ckpt": (off["peak_alloc_gb"] is not None and off["peak_alloc_gb"] < 70)}
    RESULTS[f"synth_checkpoint_R{R}"] = r
    print(f"SYNTH checkpoint R={R} [MEASURED/CALCULATED]:", json.dumps(r, indent=2))
    return r

def run_synthetic():
    """No-dataset profiling: environment, model, checkpoint ON/OFF (96 & 128),
    per-level timing. Answers the checkpoint + per-level questions on H100 without
    the 33GB BraTS dataset. Real-data tests (3,4,7,8,10) are skipped here."""
    test1_env()
    cfg, m, dev = test2_model()
    synth_checkpoint(cfg, dev, R=96)
    synth_checkpoint(cfg, dev, R=128)
    test9_per_level(cfg, dev)
    RESULTS["mode"] = "SYNTHETIC (no BraTS dataset; real-data tests deferred)"
    with open(os.path.join(OUT_DIR, "h100_profile.md"), "w") as fh:
        fh.write("# GLO-NCA V3 H100 profile — SYNTHETIC (machine-generated)\n\n")
        fh.write("```json\n" + json.dumps(RESULTS, indent=2, default=str) + "\n```\n")
    _save()
    print("\nDONE (synthetic). Paste experiments/profiling/v3_h100/h100_profile.json back to Claude.")
    print("Real-data TESTs 3/4/7/8/10 skipped (no dataset). Do NOT run training here.")

if __name__ == "__main__":
    # DEFAULT: synthetic (no dataset). To run the full real-data suite once BraTS
    # is attached, call run_all() instead.
    run_synthetic()

# ------- 5-EPOCH (SEPARATE, do NOT run in this profiling notebook) -----------
# !DATA_ROOT=$DATA python train.py --config configs/v3_kaggle_5epoch.yaml --output /kaggle/working/exp5
