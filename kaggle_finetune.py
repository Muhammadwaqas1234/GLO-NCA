r"""
================================================================================
GLO-NCA — train + FINAL FINE-TUNE phase (Kaggle, single cell)
================================================================================
Two-stage training, then test:

  STAGE 1  Main training : EPOCHS_MAIN epochs, normal hyper-parameters,
                           best model saved by mean validation Dice.
  STAGE 2  Fine-tune     : load the best model, train EPOCHS_FT (=50) MORE
                           epochs with refined hyper-parameters that stabilise
                           and sharpen the final model:
                             * lower LR            16e-4 -> 2e-4  (settle, kill sawtooth)
                             * higher fire rate    0.5   -> 0.7   (steadier output)
                             * stronger ET/TC loss Tversky beta 0.7 -> 0.8 (recall)
  TEST                   : evaluate the best model from EACH stage so you can
                           SEE the fine-tune gain (before vs after).

USAGE (one Kaggle cell, GPU on, BraTS attached):
    !rm -rf GLO-NCA && git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_finetune.py
================================================================================
"""
import os, sys, time, json, math, random, subprocess

EPOCHS_MAIN = 100
EPOCHS_FT   = 50
N_PATIENTS  = None
SEED        = 42
REGIONS     = ["WT", "TC", "ET"]
MODALITIES  = ["t1n", "t1c", "t2w", "t2f"]

# main-stage hyper-parameters
LR_MAIN, FIRE_MAIN, BETA_MAIN = 16e-4, 0.5, 0.7
# fine-tune-stage hyper-parameters (refined)
LR_FT,   FIRE_FT,   BETA_FT   = 2e-4, 0.7, 0.8

REPO_URL = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"
REPO_DIR = "/kaggle/working/GLO-NCA"
OUT_DIR  = "/kaggle/working/glo_ft"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchio"], check=False)
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


def set_lr(optimizers, lr):
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = lr


def set_fire_rate(models, fr):
    for m in models:
        m.fire_rate = fr


def train_loop(agent, dataset, ca, loss_f, n_epochs, tag, ckpt_path, hist):
    """Run n_epochs, save best (by val mean Dice) to ckpt_path, log to hist."""
    best = -1.0
    for ep in range(n_epochs):
        losses = []
        for data in torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1):
            r = agent.batch_step(data, loss_f)
            if r:
                losses.append(sum(r.values()))
        val = evaluate(agent, dataset, "val")
        vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
        hist["epoch"].append(len(hist["epoch"]) + 1)
        hist["val_mean"].append(vm); hist["stage"].append(tag)
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  [{tag}] ep {ep+1}/{n_epochs} loss {np.mean(losses):.3f} | val mean {vm:.3f} "
                  f"(WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} ET {val['ET']['dice']:.3f})",
                  flush=True)
        if vm > best:
            best = vm
            torch.save({"m": [m.state_dict() for m in ca], "ep": ep+1, "val_mean": vm}, ckpt_path)
    return best


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    tr, va, te = make_split(SEED)
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")

    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(OUT_DIR, "m"),
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": LR_MAIN, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": EPOCHS_MAIN,
        "batch_size": 1, "batch_duplication": 1,
        "channel_n": 16, "inference_steps": [10, 10], "cell_fire_rate": FIRE_MAIN,
        "input_channels": 4, "output_channels": 3, "hidden_size": 64,
        "train_model": 1, "use_attention": True,
        "input_size": [[28, 28, 20], [56, 56, 40]], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.5,
    }]
    ds = Dataset_NiiGz_3D_BraTS(); ds.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(16, FIRE_MAIN, dev, 64, kernel_size=7, input_channels=4, use_attention=True),
          BasicNCA3D(16, FIRE_MAIN, dev, 64, kernel_size=3, input_channels=4, use_attention=True)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent); ds.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    hist = {"epoch": [], "val_mean": [], "stage": []}
    main_ckpt = os.path.join(OUT_DIR, "best_main.pth")
    ft_ckpt = os.path.join(OUT_DIR, "best_ft.pth")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # ---------------- STAGE 1: main training ----------------
    print(f"\n=== STAGE 1: main training ({EPOCHS_MAIN} epochs, lr {LR_MAIN}, fire {FIRE_MAIN}) ===")
    loss_main = TverskyCELoss(alpha=0.3, beta=BETA_MAIN, ce_weight=0.5)
    train_loop(agent, ds, ca, loss_main, EPOCHS_MAIN, "main", main_ckpt, hist)

    # evaluate STAGE-1 best on TEST (the "before fine-tune" result)
    ck = torch.load(main_ckpt, map_location=dev)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    test_before = evaluate(agent, ds, "test")
    print(f"\n  [before fine-tune] TEST  WT {test_before['WT']['dice']:.3f}  "
          f"TC {test_before['TC']['dice']:.3f}  ET {test_before['ET']['dice']:.3f}", flush=True)

    # ---------------- STAGE 2: fine-tune ----------------
    print(f"\n=== STAGE 2: fine-tune ({EPOCHS_FT} epochs, lr {LR_FT}, fire {FIRE_FT}, "
          f"beta {BETA_FT}) ===")
    set_lr(agent.optimizer, LR_FT)            # lower LR -> settle, kill sawtooth
    set_fire_rate(ca, FIRE_FT)                # steadier output
    loss_ft = TverskyCELoss(alpha=0.2, beta=BETA_FT, ce_weight=0.5)   # push ET/TC recall
    train_loop(agent, ds, ca, loss_ft, EPOCHS_FT, "finetune", ft_ckpt, hist)

    # evaluate STAGE-2 best on TEST (the "after fine-tune" result)
    ck2 = torch.load(ft_ckpt, map_location=dev)
    for m, sd in zip(ca, ck2["m"]):
        m.load_state_dict(sd)
    test_after = evaluate(agent, ds, "test")
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0

    # ---------------- report ----------------
    print("\n" + "=" * 64)
    print("FINE-TUNE EFFECT (Test Dice)")
    print("=" * 64)
    print(f"{'stage':<22}{'WT':<9}{'TC':<9}{'ET':<9}{'mean':<9}")
    print("-" * 64)
    def row(lab, t):
        m = np.mean([t[r]['dice'] for r in REGIONS])
        print(f"{lab:<22}{t['WT']['dice']:<9.3f}{t['TC']['dice']:<9.3f}{t['ET']['dice']:<9.3f}{m:<9.3f}")
    row("Before fine-tune", test_before)
    row("After fine-tune", test_after)
    dWT = test_after['WT']['dice'] - test_before['WT']['dice']
    dTC = test_after['TC']['dice'] - test_before['TC']['dice']
    dET = test_after['ET']['dice'] - test_before['ET']['dice']
    print("-" * 64)
    print(f"{'delta':<22}{dWT:<+9.3f}{dTC:<+9.3f}{dET:<+9.3f}")
    print("=" * 64)
    print(f"total time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")

    json.dump({"before": test_before, "after": test_after, "history": hist,
               "params": n_params, "peak_vram": peak, "train_time": train_time},
              open(os.path.join(OUT_DIR, "finetune_results.json"), "w"), indent=2, default=str)

    # curve with the fine-tune boundary marked
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["epoch"], hist["val_mean"], color="#1f77b4")
    ax.axvline(EPOCHS_MAIN + 0.5, color="red", ls="--", label="fine-tune starts")
    ax.set_title("Validation mean Dice — main training then fine-tune")
    ax.set_xlabel("epoch"); ax.set_ylabel("val mean Dice"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "finetune_curve.png"), dpi=130); plt.show()
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
