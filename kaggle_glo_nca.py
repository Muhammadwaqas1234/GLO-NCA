r"""
================================================================================
GLO-NCA on Kaggle - full BraTS run (clone-repo + thin runner)
================================================================================
One file to run the whole experiment on Kaggle:
  * clones the GLO-NCA GitHub repo and imports its `src/` package
  * 70% train / 15% validation / 15% test split
  * trains 100 epochs, saves the BEST model by mean validation Dice(WT,TC,ET)
  * plots: (1) loss+Dice curves, (2) final test bar chart,
           (3) prediction overlays, (4) NQM variance/quality map
  * prints + saves final WT/TC/ET Dice, mIoU, HD95 on the held-out TEST set

HOW TO USE ON KAGGLE
  1. New Notebook -> add the BraTS dataset (Data panel).
  2. Settings -> Accelerator -> GPU.
  3. Set REPO_URL below to your GitHub repo.
  4. Set DATA_ROOT to the dataset folder (the one whose sub-folders are patients).
  5. Paste this file into a cell (or %run it) and run.
================================================================================
"""
import os
import sys
import time
import json
import random
import subprocess

import numpy as np

# ------------------------------------------------------------------ settings
REPO_URL  = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"   # <-- your repo
REPO_DIR  = "/kaggle/working/GLO-NCA"
DATA_ROOT = "/kaggle/input/brats2024-small-dataset/BraTS2024_small_dataset"  # <-- patient folders
OUT_DIR   = "/kaggle/working/glo_out"
EPOCHS    = 100
SEED      = 42
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]   # BraTS 2024 suffixes
REGIONS    = ["WT", "TC", "ET"]


# ------------------------------------------------------------------ clone repo
def ensure_repo():
    # Skip cloning if we are already inside a checkout that has src/ (local test
    # or repo already cloned). Set GLO_LOCAL=1 to force-use the current dir.
    if os.environ.get("GLO_LOCAL") == "1":
        local = os.path.abspath(os.path.dirname(__file__))
        if local not in sys.path:
            sys.path.insert(0, local)
        return
    if os.path.isdir(os.path.join(REPO_DIR, "src")):
        pass  # already cloned
    elif not os.path.isdir(REPO_DIR):
        print("Cloning", REPO_URL)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)


ensure_repo()

import torch
import matplotlib.pyplot as plt
import nibabel as nib

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA


# ------------------------------------------------------------------ helpers
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split_patients(root, split=(0.70, 0.15, 0.15)):
    pats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    random.Random(SEED).shuffle(pats)
    n = len(pats)
    n_tr, n_val = int(n * split[0]), int(n * split[1])
    return pats[:n_tr], pats[n_tr:n_tr+n_val], pats[n_tr+n_val:]


def evaluate(agent, dataset, regions, state):
    """Per-region mean Dice/mIoU/HD95 on the 'val' or 'test' split.

    Uses the experiment's own state switching (set_model_state) so the dataset
    is correctly pointed at the right patients with the right size, then runs
    full-image inference through the agent.
    """
    from src.agents.Agent import iou_score as _iou, hd95_score as _hd
    import math
    agent.exp.set_model_state(state)          # repoints dataset paths to this split
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in regions}
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            outputs, targets = agent.get_outputs(data, full_img=True)
            prob = torch.sigmoid(outputs).detach().cpu().numpy()
            gt = targets.detach().cpu().numpy()
            for i, r in enumerate(regions):
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                d = (2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6)
                acc[r]["dice"].append(float(d))
                acc[r]["iou"].append(_iou(p, t))
                acc[r]["hd95"].append(_hd(p, t))
    agent.exp.set_model_state("train")
    out = {}
    for r in regions:
        hd_valid = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {
            "dice": float(np.mean(acc[r]["dice"])) if acc[r]["dice"] else float("nan"),
            "iou": float(np.mean(acc[r]["iou"])) if acc[r]["iou"] else float("nan"),
            "hd95": float(np.mean(hd_valid)) if hd_valid else float("nan"),
        }
    return out


# ------------------------------------------------------------------ build
def build(train_ids, val_ids, test_ids):
    os.makedirs(OUT_DIR, exist_ok=True)
    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT,
        "model_path": os.path.join(OUT_DIR, "model"),
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": 16e-4, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": EPOCHS * 100, "evaluate_interval": EPOCHS * 100, "n_epoch": EPOCHS,
        "batch_size": 1, "batch_duplication": 1,
        "channel_n": 16, "inference_steps": [10, 10], "cell_fire_rate": 0.5,
        "input_channels": 4, "output_channels": 3, "hidden_size": 64,
        "train_model": 1, "use_attention": True,
        "input_size": [(28, 28, 20), (56, 56, 40)], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.5,
    }]
    dataset = Dataset_NiiGz_3D_BraTS()
    dataset.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(16, 0.5, dev, 64, kernel_size=7, input_channels=4, use_attention=True),
          BasicNCA3D(16, 0.5, dev, 64, kernel_size=3, input_channels=4, use_attention=True)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)

    def entry(p): return (p, p, 0)
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        exp.data_split.images[split] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[split] = {p: {0: entry(p)} for p in ids}
    return dataset, ca, agent, exp, dev


# ------------------------------------------------------------------ train loop
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_ids, val_ids, test_ids = split_patients(DATA_ROOT)
    print(f"Split -> train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")

    dataset, ca, agent, exp, dev = build(train_ids, val_ids, test_ids)
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params)

    loss_f = DiceCELoss()
    exp.set_model_state("train")
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)

    history = {"epoch": [], "train_loss": [],
               "val_WT": [], "val_TC": [], "val_ET": [], "val_mean": []}
    best_mean, best_path = -1.0, os.path.join(OUT_DIR, "best_model.pth")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for epoch in range(EPOCHS):
        # --- train one epoch ---
        losses = []
        for data in train_loader:
            r = agent.batch_step(data, loss_f)
            if r:
                losses.append(sum(r.values()))
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # --- validate ---
        val = evaluate(agent, dataset, REGIONS, "val")
        vmean = float(np.mean([val[r]["dice"] for r in REGIONS]))
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        for r in REGIONS:
            history[f"val_{r}"].append(val[r]["dice"])
        history["val_mean"].append(vmean)
        print(f"epoch {epoch+1}/{EPOCHS} | loss {avg_loss:.4f} | "
              f"val Dice WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} "
              f"ET {val['ET']['dice']:.3f} | mean {vmean:.3f}")

        # --- save BEST by mean validation Dice(WT,TC,ET) ---
        if vmean > best_mean:
            best_mean = vmean
            torch.save({"model0": ca[0].state_dict(), "model1": ca[1].state_dict(),
                        "epoch": epoch + 1, "val_mean": vmean, "val": val}, best_path)
            print(f"   * new best (mean val Dice {vmean:.3f}) -> saved")

    train_time = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0

    # --- reload best, evaluate on TEST ---
    ck = torch.load(best_path, map_location=dev)
    ca[0].load_state_dict(ck["model0"]); ca[1].load_state_dict(ck["model1"])
    print(f"\nLoaded best model (epoch {ck['epoch']}, val mean {ck['val_mean']:.3f})")
    test = evaluate(agent, dataset, REGIONS, "test")

    print("\n" + "=" * 56)
    print(f"FINAL TEST RESULTS (best model @ epoch {ck['epoch']})")
    print("=" * 56)
    print(f"{'region':<8}{'Dice':<12}{'mIoU':<12}{'HD95':<12}")
    for r in REGIONS:
        print(f"{r:<8}{test[r]['dice']:<12.4f}{test[r]['iou']:<12.4f}{test[r]['hd95']:<12.3f}")
    print("-" * 56)
    print(f"train time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")

    json.dump({"history": history, "test": test, "best_epoch": ck["epoch"],
               "params": n_params, "train_time": train_time, "peak_vram_gb": peak},
              open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2, default=str)

    make_plots(history, test, agent, dataset, dev)
    print("\nAll outputs saved to", OUT_DIR)


# ------------------------------------------------------------------ plots
def make_plots(history, test, agent, dataset, dev):
    # (1) loss + Dice curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["epoch"], history["train_loss"], color="crimson")
    ax1.set_title("Training Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(alpha=.3)
    for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
        ax2.plot(history["epoch"], history[f"val_{r}"], label=f"val Dice {r}", color=c)
    ax2.plot(history["epoch"], history["val_mean"], "--k", label="val mean")
    ax2.set_title("Validation Dice per Region"); ax2.set_xlabel("epoch")
    ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=130); plt.show()

    # (2) final test bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(REGIONS)); w = 0.35
    ax.bar(x - w/2, [test[r]["dice"] for r in REGIONS], w, label="Dice", color="#1f77b4")
    ax.bar(x + w/2, [test[r]["iou"] for r in REGIONS], w, label="mIoU", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(REGIONS); ax.set_ylim(0, 1)
    ax.set_title("Final Test Metrics per Region"); ax.legend(); ax.grid(axis="y", alpha=.3)
    for i, r in enumerate(REGIONS):
        ax.text(i - w/2, test[r]["dice"] + .02, f"{test[r]['dice']:.2f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "test_bars.png"), dpi=130); plt.show()

    # (3) prediction overlay + (4) variance map on one test case
    agent.exp.set_model_state("test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    data = next(iter(loader))
    data = agent.prepare_data(data, eval=True)
    _, inputs, _ = data
    runs = []
    with torch.no_grad():
        for _ in range(10):
            out, targets = agent.get_outputs(data, full_img=True)
            runs.append(torch.sigmoid(out).cpu().numpy())
    mean_pred = np.mean(runs, axis=0)[0]            # (X,Y,Z,3)
    var_map = np.std(runs, axis=0)[0]               # (X,Y,Z,3)
    gt = targets.cpu().numpy()[0]
    img = inputs.cpu().numpy()[0]                   # (X,Y,Z,channels)
    z = img.shape[2] // 2

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(img[:, :, z, 0], cmap="gray"); axs[0].set_title("MRI (T1)"); axs[0].axis("off")
    axs[1].imshow(img[:, :, z, 0], cmap="gray")
    axs[1].imshow(np.argmax(np.concatenate([np.zeros_like(gt[:, :, z, :1]), gt[:, :, z]], -1), -1),
                  alpha=0.5, cmap="jet"); axs[1].set_title("Ground Truth"); axs[1].axis("off")
    axs[2].imshow(img[:, :, z, 0], cmap="gray")
    axs[2].imshow(np.argmax(np.concatenate([np.full_like(mean_pred[:, :, z, :1], .5),
                  mean_pred[:, :, z]], -1), -1), alpha=0.5, cmap="jet")
    axs[2].set_title("Prediction"); axs[2].axis("off")
    im = axs[3].imshow(var_map[:, :, z, 0], cmap="inferno")
    axs[3].set_title("Variance / Quality (NQM)"); axs[3].axis("off")
    fig.colorbar(im, ax=axs[3], fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "overlay_variance.png"), dpi=130); plt.show()
    agent.exp.set_model_state("train")


if __name__ == "__main__":
    main()
