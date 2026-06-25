r"""
================================================================================
GLO-NCA v2 — improved single run (Kaggle, single cell)
================================================================================
Incorporates the lessons from the previous run (the flat fine-tune phase added
nothing; training was a sawtooth because the LR barely decayed):

  * COSINE LR decay over all epochs   -> smooth convergence, kills the sawtooth
  * channel_n 16 -> 24                 -> more capacity for the weak TC / ET
  * hidden_size 64 -> 96               -> richer update rule
  * inference_steps 10 -> 15           -> more "thinking", sharper boundaries
  * 150 epochs in ONE run             -> (the 50 fine-tune epochs folded in here)
  * Tversky+BCE loss (beta>alpha)     -> recall-focused for small ET / TC
  * best model saved by mean val Dice; final TEST on the held-out 15%

USAGE (one Kaggle cell, GPU on, BraTS attached):
    !rm -rf GLO-NCA && git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_v2.py
================================================================================
"""
import os, sys, time, json, math, random, subprocess

# ---- knobs ----
EPOCHS      = 150
N_PATIENTS  = None          # None = all 200
SEED        = 42
CHANNEL_N   = 24            # was 16
HIDDEN      = 96            # was 64
STEPS       = [15, 15]      # was [10, 10]
LR_START    = 16e-4
LR_MIN      = 1e-5          # cosine decays down to this
REGIONS     = ["WT", "TC", "ET"]
MODALITIES  = ["t1n", "t1c", "t2w", "t2f"]

REPO_URL = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"
REPO_DIR = "/kaggle/working/GLO-NCA"
OUT_DIR  = "/kaggle/working/glo_v2"

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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    tr, va, te = make_split(SEED)
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"Config: channel_n={CHANNEL_N} hidden={HIDDEN} steps={STEPS} "
          f"epochs={EPOCHS} | cosine LR {LR_START}->{LR_MIN}")

    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(OUT_DIR, "m"),
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": LR_START, "lr_gamma": 0.9999,   # gamma unused (we override sched)
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": EPOCHS,
        "batch_size": 1, "batch_duplication": 1,
        "channel_n": CHANNEL_N, "inference_steps": STEPS, "cell_fire_rate": 0.5,
        "input_channels": 4, "output_channels": 3, "hidden_size": HIDDEN,
        "train_model": 1, "use_attention": True,
        "input_size": [[28, 28, 20], [56, 56, 40]], "scale_factor": 2,
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

    # --- COSINE LR schedule ---
    # The agent's batch_step() steps the scheduler ONCE PER BATCH, so size the
    # cosine period to the total number of batch updates (epochs * batches) for a
    # smooth per-batch decay from LR_START down to LR_MIN.
    total_steps = EPOCHS * max(1, len(tr))
    agent.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=LR_MIN)
                       for opt in agent.optimizer]

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
        n_train = len(tr)
        for i, data in enumerate(torch.utils.data.DataLoader(ds, shuffle=True, batch_size=1)):
            r = agent.batch_step(data, loss_f)   # batch_step steps optimizer; scheduler stepped below
            if r:
                losses.append(sum(r.values()))
            print(f"\rep {ep+1}/{EPOCHS} training {i+1}/{n_train}...", end="", flush=True)
        # NOTE: batch_step() already calls scheduler.step() per batch, so the
        # cosine LR decays smoothly across batches; we do NOT step it again here.
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
    test = evaluate(agent, ds, "test")

    print("\n" + "=" * 60)
    print(f"GLO-NCA v2 — FINAL TEST (best model @ epoch {ck['ep']})")
    print("=" * 60)
    print(f"{'region':<8}{'Dice':<12}{'mIoU':<12}{'HD95':<12}")
    for r in REGIONS:
        print(f"{r:<8}{test[r]['dice']:<12.4f}{test[r]['iou']:<12.4f}{test[r]['hd95']:<12.3f}")
    mean = np.mean([test[r]['dice'] for r in REGIONS])
    print("-" * 60)
    print(f"{'mean':<8}{mean:<12.4f}")
    print(f"train time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")

    json.dump({"test": test, "history": hist, "best_epoch": ck["ep"], "params": n_params,
               "train_time": train_time, "peak_vram": peak},
              open(os.path.join(OUT_DIR, "v2_results.json"), "w"), indent=2, default=str)

    # plots: loss + LR + val curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["epoch"], hist["loss"], color="crimson", label="train loss")
    a1b = a1.twinx(); a1b.plot(hist["epoch"], hist["lr"], color="gray", ls="--", label="LR")
    a1.set_title("Loss & cosine LR"); a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1b.set_ylabel("LR")
    for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
        a2.plot(hist["epoch"], hist[f"val_{r}"], label=f"val {r}", color=c)
    a2.plot(hist["epoch"], hist["val_mean"], "--k", label="val mean")
    a2.set_title("Validation Dice (should be smoother now)"); a2.set_xlabel("epoch")
    a2.set_ylabel("Dice"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "v2_curves.png"), dpi=130); plt.show()
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
