r"""
================================================================================
GLO-NCA — MAX-accuracy + thesis-rigor run (Kaggle, single cell)
================================================================================
Run this AFTER kaggle_experiments.py, to push TC/ET higher and add the
heavy-model comparison. It runs:

  1. GLO-NCA-MAX   : strongest config
        - deeper cascade (3 levels) for more multi-scale global context
        - larger / higher-res patches  -> recovers small ET detail
        - more channels + inference steps -> more capacity
        - Tversky+BCE loss (beta>alpha)   -> penalises missed tumour (helps ET/TC)
        - light augmentation (random flips) -> better generalisation
  2. UNet3D        : heavy baseline (millions of params) for the thesis table
                     ("lightweight NCA vs heavy UNet")

Prints a final table: GLO-NCA-MAX vs UNet3D (Dice/mIoU/HD95 + params).

USAGE (one Kaggle cell, GPU on, BraTS attached):
    !rm -rf GLO-NCA && git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_max.py
================================================================================
"""
import os, sys, time, json, math, random, subprocess

EPOCHS     = 120
N_PATIENTS = None        # None = all
SEED       = 42
RUN_UNET   = True        # set False to skip the heavy baseline
REGIONS    = ["WT", "TC", "ET"]
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

REPO_URL = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"
REPO_DIR = "/kaggle/working/GLO-NCA"
OUT_DIR  = "/kaggle/working/glo_max"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchio", "unet"], check=False)
if not os.path.isdir(os.path.join(REPO_DIR, "src")):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import TverskyCELoss, DiceCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.agents.Agent import iou_score, hd95_score


def find_data_root(base="/kaggle/input"):
    for root, dirs, files in os.walk(base):
        c = 0
        for d in dirs:
            try:
                if any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(os.path.join(root, d))):
                    c += 1
            except Exception:
                pass
        if c >= 2:
            return root, c
    return None, 0


DATA_ROOT, nf = find_data_root("/kaggle/input")
assert DATA_ROOT, "BraTS not found under /kaggle/input — attach the dataset."
print(f"DATA_ROOT = {DATA_ROOT} ({nf} patients)")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_split(seed):
    pats = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    random.Random(seed).shuffle(pats)
    if N_PATIENTS:
        pats = pats[:N_PATIENTS]
    n = len(pats); a, b = int(n*0.70), int(n*0.15)
    return pats[:a], pats[a:a+b], pats[a+b:]


def augment(img, label):
    """Light, label-preserving augmentation: random axis flips (numpy arrays)."""
    for ax in (0, 1, 2):
        if random.random() < 0.5:
            img = np.flip(img, axis=ax); label = np.flip(label, axis=ax)
    return np.ascontiguousarray(img), np.ascontiguousarray(label)


def evaluate(agent, dataset, state):
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            outputs, targets = agent.get_outputs(data, full_img=True)
            prob = torch.sigmoid(outputs).detach().cpu().numpy()
            gt = targets.detach().cpu().numpy()
            for i, r in enumerate(REGIONS):
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                acc[r]["dice"].append((2*inter)/((p >= 0.5).sum()+(t >= 0.5).sum()+1e-6))
                acc[r]["iou"].append(iou_score(p, t)); acc[r]["hd95"].append(hd95_score(p, t))
    agent.exp.set_model_state("train")
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


# ------------------------------------------------------------------ GLO-NCA-MAX
def run_glo_max():
    set_seed(SEED)
    mp = os.path.join(OUT_DIR, "glo_max")
    tr, va, te = make_split(SEED)
    print(f"\n=== GLO-NCA-MAX === train {len(tr)} val {len(va)} test {len(te)}")
    # Deeper: 3 levels (train_model=2). Higher-res patches. More channels/steps.
    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": mp,
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": 16e-4, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": EPOCHS*100, "evaluate_interval": EPOCHS*100, "n_epoch": EPOCHS,
        "batch_size": 1, "batch_duplication": 1,
        "channel_n": 24, "inference_steps": [15, 15, 15], "cell_fire_rate": 0.5,
        "input_channels": 4, "output_channels": 3, "hidden_size": 96,
        "train_model": 2, "use_attention": True,
        "input_size": [[24, 24, 16], [48, 48, 32], [96, 96, 64]], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.7,
    }]
    ds = Dataset_NiiGz_3D_BraTS(); ds.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(24, 0.5, dev, 96, kernel_size=7, input_channels=4, use_attention=True),
          BasicNCA3D(24, 0.5, dev, 96, kernel_size=3, input_channels=4, use_attention=True),
          BasicNCA3D(24, 0.5, dev, 96, kernel_size=3, input_channels=4, use_attention=True)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent); ds.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=1)
    loss_f = TverskyCELoss(alpha=0.3, beta=0.7, ce_weight=0.5)   # recall-focused for ET/TC
    n_params = sum(p.numel() for m in ca for p in m.parameters())

    hist = {"epoch": [], "val_mean": []}
    best, bestp = -1.0, os.path.join(mp, "best.pth"); os.makedirs(mp, exist_ok=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(EPOCHS):
        losses = []
        for data in loader:
            losses.append(_step_with_aug(agent, data, loss_f))
        val = evaluate(agent, ds, "val")
        vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
        hist["epoch"].append(ep+1); hist["val_mean"].append(vm)
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1}/{EPOCHS} loss {np.mean(losses):.3f} | val mean {vm:.3f} "
                  f"(WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} ET {val['ET']['dice']:.3f})",
                  flush=True)
        if vm > best:
            best = vm
            torch.save({"m": [m.state_dict() for m in ca], "ep": ep+1}, bestp)
    tt = time.time() - t0
    peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0
    ck = torch.load(bestp, map_location=dev)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    test = evaluate(agent, ds, "test")
    print(f"  -> TEST WT {test['WT']['dice']:.3f} TC {test['TC']['dice']:.3f} "
          f"ET {test['ET']['dice']:.3f} | {tt:.0f}s | {peak:.2f}GB | {n_params} params", flush=True)
    return {"test": test, "params": n_params, "train_time": tt, "peak_vram": peak,
            "best_epoch": ck["ep"], "history": hist}


def _step_with_aug(agent, data, loss_f):
    """batch_step but with random-flip augmentation applied to the raw arrays."""
    idx, img, label = data
    img = img.numpy(); label = label.numpy()
    for b in range(img.shape[0]):
        img[b], label[b] = augment(img[b], label[b])
    data = (idx, torch.from_numpy(img), torch.from_numpy(label))
    r = agent.batch_step(data, loss_f)
    return sum(r.values()) if r else 0.0


# ------------------------------------------------------------------ UNet3D baseline
def run_unet():
    try:
        from unet import UNet3D
    except Exception as e:
        print("UNet not available, skipping baseline:", e)
        return None
    # Self-contained UNet train/eval loop below (no dependency on src/agents).
    print("\n=== UNet3D baseline === (heavy model for comparison)")
    set_seed(SEED)
    tr, va, te = make_split(SEED)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Reuse the BraTS dataset at a fixed full-volume-ish size.
    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT,
        "model_path": os.path.join(OUT_DIR, "unet"), "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": 1e-4, "lr_gamma": 0.9999, "betas": (0.9, 0.99),
        "weight_decay": 1e-4, "save_interval": EPOCHS*100, "evaluate_interval": EPOCHS*100,
        "n_epoch": EPOCHS, "batch_size": 1, "input_channels": 4, "output_channels": 3,
        "channel_n": 16, "input_size": [[64, 64, 48]], "data_split": [0.7, 0.15, 0.15],
        "keep_original_scale": True, "rescale": True, "patchify": False, "cell_fire_rate": 0.5,
    }]
    ds = Dataset_NiiGz_3D_BraTS(); ds.MODALITIES = MODALITIES
    model = UNet3D(in_channels=4, out_classes=3, padding=1).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    exp = Experiment(config, ds, model, _DummyAgent())  # only need data_split + paths
    ds.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_f = DiceCELoss()
    exp.set_model_state("train")
    loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=1)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(EPOCHS):
        for _, img, label in loader:
            img = img.float().to(dev).permute(0, 4, 1, 2, 3)        # (B,C,X,Y,Z)
            label = label.float().to(dev).permute(0, 4, 1, 2, 3)
            opt.zero_grad()
            out = model(img)
            loss = sum(loss_f(out[:, c], label[:, c]) for c in range(3))
            loss.backward(); opt.step()
        if (ep+1) % 20 == 0:
            print(f"  unet ep {ep+1}/{EPOCHS} loss {float(loss):.3f}", flush=True)
    tt = time.time() - t0
    peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0

    # eval
    exp.set_model_state("test"); model.eval()
    acc = {r: [] for r in REGIONS}
    with torch.no_grad():
        for _, img, label in torch.utils.data.DataLoader(ds, batch_size=1):
            img = img.float().to(dev).permute(0, 4, 1, 2, 3)
            prob = torch.sigmoid(model(img)).cpu().numpy()[0]       # (3,X,Y,Z)
            gt = label.numpy()[0]                                   # (X,Y,Z,3)
            for i, r in enumerate(REGIONS):
                p = prob[i]; t = gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                acc[r].append((2*inter)/((p >= 0.5).sum()+(t >= 0.5).sum()+1e-6))
    test = {r: {"dice": float(np.mean(acc[r])), "iou": float("nan"), "hd95": float("nan")} for r in REGIONS}
    print(f"  -> UNet TEST WT {test['WT']['dice']:.3f} TC {test['TC']['dice']:.3f} "
          f"ET {test['ET']['dice']:.3f} | {tt:.0f}s | {peak:.2f}GB | {n_params} params", flush=True)
    return {"test": test, "params": n_params, "train_time": tt, "peak_vram": peak}


class _DummyAgent:
    def set_exp(self, exp): self.exp = exp
    def initialize(self): pass


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {}
    res["glo_max"] = run_glo_max()
    if RUN_UNET:
        u = run_unet()
        if u:
            res["unet"] = u
    json.dump(res, open(os.path.join(OUT_DIR, "max_results.json"), "w"), indent=2, default=str)

    print("\n" + "=" * 70)
    print("MAX-ACCURACY + BASELINE COMPARISON (Test Dice)")
    print("=" * 70)
    print(f"{'Method':<22}{'WT':<8}{'TC':<8}{'ET':<8}{'mean':<8}{'params':<12}")
    print("-" * 70)
    for key, lab in [("glo_max", "GLO-NCA-MAX (ours)"), ("unet", "UNet3D (heavy)")]:
        if key in res and res[key]:
            t = res[key]["test"]; m = np.mean([t[r]["dice"] for r in REGIONS])
            print(f"{lab:<22}{t['WT']['dice']:<8.3f}{t['TC']['dice']:<8.3f}"
                  f"{t['ET']['dice']:<8.3f}{m:<8.3f}{res[key]['params']:<12}")
    print("=" * 70)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
