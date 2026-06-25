r"""
================================================================================
GLO-NCA v3 — v2 (proven) + safe gains (Kaggle, single cell)
================================================================================
v2 reached WT 0.83 / TC 0.69 / ET 0.68. v3 keeps EVERYTHING that worked and only
adds low-risk improvements (no architecture change that hurt max/maxpush):

  KEPT from v2 (the part that works):
    * 2-LEVEL cascade (NOT 3 — 3 levels trained worse)
    * channel_n 24, hidden 96, steps [15,15]
    * cosine LR decay 16e-4 -> 1e-5
    * Tversky+BCE loss
  ADDED (safe, mostly free):
    * 250 epochs (v2 was still improving at ep141)
    * LR WARMUP over first 5 epochs (smooths the noisy early training)
    * PSEUDO-ENSEMBLE at test: average 10 stochastic inferences (free gain)
    * POST-PROCESSING at test: keep largest connected component / drop specks

The final table reports plain -> +ensemble -> +ensemble+postproc so you can SEE
each technique's contribution. Expected: ~WT 0.85 / TC 0.72 / ET 0.71.

USAGE (one Kaggle cell, GPU on, BraTS attached):
    !rm -rf GLO-NCA && git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_v3.py
================================================================================
"""
import os, sys, time, json, math, random, subprocess

# ---- knobs (v2 values kept; only epochs/warmup/ensemble added) ----
EPOCHS      = 250
WARMUP_EP   = 5
N_PATIENTS  = None
SEED        = 42
CHANNEL_N   = 24            # v2
HIDDEN      = 96            # v2
STEPS       = [15, 15]      # v2 (2 levels)
INPUT_SIZE  = [[28, 28, 20], [56, 56, 40]]   # v2 (2 levels)
TRAIN_MODEL = 1             # v2: 2 NCAs (do NOT raise to 2 — that hurt)
LR_START    = 16e-4
LR_MIN      = 1e-5
ENSEMBLE_N  = 10
POSTPROC    = True
REGIONS     = ["WT", "TC", "ET"]
MODALITIES  = ["t1n", "t1c", "t2w", "t2f"]

REPO_URL = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"
REPO_DIR = "/kaggle/working/GLO-NCA"
OUT_DIR  = "/kaggle/working/glo_v3"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchio"], check=False)
if not os.path.isdir(os.path.join(REPO_DIR, "src")):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import TverskyCELoss
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


# light augmentation only (flips) — heavy aug hurt early training in maxpush
def augment(img, label):
    for ax in (0, 1, 2):
        if random.random() < 0.5:
            img = np.flip(img, axis=ax); label = np.flip(label, axis=ax)
    return np.ascontiguousarray(img), np.ascontiguousarray(label)


def step_with_aug(agent, data, loss_f):
    idx, img, label = data
    img = img.numpy(); label = label.numpy()
    for b in range(img.shape[0]):
        img[b], label[b] = augment(img[b], label[b])
    r = agent.batch_step((idx, torch.from_numpy(img), torch.from_numpy(label)), loss_f)
    return sum(r.values()) if r else 0.0


def postprocess(mask, min_voxels=20):
    if mask.sum() == 0:
        return mask
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    keep = np.zeros_like(mask)
    keep[lbl == int(np.argmax(sizes)) + 1] = 1
    for comp in range(1, n + 1):
        if sizes[comp - 1] >= min_voxels:
            keep[lbl == comp] = 1
    return keep


def evaluate(agent, dataset, state, ensemble=1, postproc=False):
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            probs = []
            for _ in range(ensemble):
                outputs, targets = agent.get_outputs(data, full_img=True)
                probs.append(torch.sigmoid(outputs).detach().cpu().numpy())
            prob = np.mean(probs, axis=0)
            gt = targets.detach().cpu().numpy()
            for i, r in enumerate(REGIONS):
                p = (prob[..., i] >= 0.5).astype(np.uint8)
                if postproc:
                    p = np.stack([postprocess(p[b]) for b in range(p.shape[0])]) if p.ndim == 4 else postprocess(p)
                t = gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                acc[r]["dice"].append((2*inter)/((p >= 0.5).sum()+(t >= 0.5).sum()+1e-6))
                acc[r]["iou"].append(iou_score(p.astype(float), t))
                acc[r]["hd95"].append(hd95_score(p.astype(float), t))
    agent.exp.set_model_state("train")
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    tr, va, te = make_split(SEED)
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"v3 config: ch={CHANNEL_N} hidden={HIDDEN} steps={STEPS} levels={TRAIN_MODEL+1} "
          f"epochs={EPOCHS} warmup={WARMUP_EP} | ensemble={ENSEMBLE_N} postproc={POSTPROC}", flush=True)

    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(OUT_DIR, "m"),
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": LR_START, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": EPOCHS,
        "batch_size": 1, "batch_duplication": 1,
        "channel_n": CHANNEL_N, "inference_steps": STEPS, "cell_fire_rate": 0.5,
        "input_channels": 4, "output_channels": 3, "hidden_size": HIDDEN,
        "train_model": TRAIN_MODEL, "use_attention": True,
        "input_size": INPUT_SIZE, "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.5,
    }]
    ds = Dataset_NiiGz_3D_BraTS(); ds.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(CHANNEL_N, 0.5, dev, HIDDEN, kernel_size=7, input_channels=4, use_attention=True),
          BasicNCA3D(CHANNEL_N, 0.5, dev, HIDDEN, kernel_size=3, input_channels=4, use_attention=True)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent); ds.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")

    # warmup -> cosine (per-batch stepping by the base agent)
    spe = max(1, len(tr)); total = EPOCHS * spe; warm = WARMUP_EP * spe
    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        floor = LR_MIN / LR_START
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    agent.scheduler = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in agent.optimizer]

    loss_f = TverskyCELoss(alpha=0.3, beta=0.7, ce_weight=0.5)
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params, "| device:", dev, flush=True)

    hist = {"epoch": [], "loss": [], "lr": [], "val_mean": [], "val_WT": [], "val_TC": [], "val_ET": []}
    best, best_path = -1.0, os.path.join(OUT_DIR, "best.pth")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for ep in range(EPOCHS):
        losses = []
        for i, data in enumerate(torch.utils.data.DataLoader(ds, shuffle=True, batch_size=1)):
            losses.append(step_with_aug(agent, data, loss_f))
            print(f"\rep {ep+1}/{EPOCHS} training {i+1}/{spe}...", end="", flush=True)
        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        val = evaluate(agent, ds, "val")
        vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
        hist["epoch"].append(ep+1); hist["loss"].append(float(np.mean(losses)))
        hist["lr"].append(cur_lr); hist["val_mean"].append(vm)
        for r in REGIONS:
            hist[f"val_{r}"].append(val[r]["dice"])
        print(f"\rep {ep+1}/{EPOCHS} | lr {cur_lr:.2e} | loss {np.mean(losses):.3f} | "
              f"val mean {vm:.3f} (WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} "
              f"ET {val['ET']['dice']:.3f})        ", flush=True)
        if vm > best:
            best = vm
            torch.save({"m": [m.state_dict() for m in ca], "ep": ep+1, "val_mean": vm}, best_path)
            print(f"   * new best {vm:.3f} saved", flush=True)

    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0

    ck = torch.load(best_path, map_location=dev)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    print(f"\nBest model @ epoch {ck['ep']} (val mean {ck['val_mean']:.3f}); TEST...", flush=True)
    test_plain = evaluate(agent, ds, "test", ensemble=1, postproc=False)
    test_ens   = evaluate(agent, ds, "test", ensemble=ENSEMBLE_N, postproc=False)
    test_full  = evaluate(agent, ds, "test", ensemble=ENSEMBLE_N, postproc=POSTPROC)

    def show(lab, t):
        m = np.mean([t[r]['dice'] for r in REGIONS])
        print(f"{lab:<26}{t['WT']['dice']:<8.3f}{t['TC']['dice']:<8.3f}{t['ET']['dice']:<8.3f}{m:<8.3f}")
    print("\n" + "=" * 62)
    print("GLO-NCA v3 — FINAL TEST (Dice)")
    print("=" * 62)
    print(f"{'setting':<26}{'WT':<8}{'TC':<8}{'ET':<8}{'mean':<8}")
    print("-" * 62)
    show("plain", test_plain)
    show(f"+ensemble x{ENSEMBLE_N}", test_ens)
    show("+ensemble +postproc", test_full)
    print("-" * 62)
    print(f"best epoch {ck['ep']} | train {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")
    print("Full metrics (ensemble+postproc):")
    for r in REGIONS:
        print(f"   {r}: Dice {test_full[r]['dice']:.4f}  mIoU {test_full[r]['iou']:.4f}  "
              f"HD95 {test_full[r]['hd95']:.3f}")
    print("=" * 62)

    json.dump({"plain": test_plain, "ensemble": test_ens, "ensemble_postproc": test_full,
               "history": hist, "best_epoch": ck["ep"], "params": n_params,
               "train_time": train_time, "peak_vram": peak},
              open(os.path.join(OUT_DIR, "v3_results.json"), "w"), indent=2, default=str)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["epoch"], hist["loss"], color="crimson", label="loss")
    a1b = a1.twinx(); a1b.plot(hist["epoch"], hist["lr"], color="gray", ls="--")
    a1.axvline(WARMUP_EP + 0.5, color="green", ls=":", label="warmup end")
    a1.set_title("Loss & LR (warmup -> cosine)"); a1.set_xlabel("epoch"); a1.legend()
    for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
        a2.plot(hist["epoch"], hist[f"val_{r}"], label=f"val {r}", color=c)
    a2.plot(hist["epoch"], hist["val_mean"], "--k", label="val mean")
    a2.set_title("Validation Dice per region"); a2.set_xlabel("epoch")
    a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "v3_curves.png"), dpi=130); plt.show()
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
