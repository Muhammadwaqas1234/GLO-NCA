r"""
================================================================================
GLO-NCA v4 — small-data tuned run (Kaggle, single cell)
================================================================================
v3 reached val WT 0.80 / TC 0.70 / ET 0.70 (foreground crop was the big win).
v4 keeps everything that worked and applies the targeted small-data tweaks that
should nudge the weak TC / ET a bit higher:

  KEPT from v3 (proven):
    * foreground crop + nonzero per-channel z-norm   (the TC/ET win)
    * 2-level cascade, ch24 hidden96, cosine LR
    * sliding-window OFF (it washed out TC/ET)
  CHANGED in v4 (the experiments you asked for):
    * USE_AUG = False        -> remove the noise that slowed v3 convergence
    * patch 56^3 -> 64^3     -> higher-res patch (crop makes this affordable) = more ET/TC detail
    * Tversky beta 0.70 -> 0.75  -> penalise missed tumour harder (small-region recall)
    * cell_fire_rate 0.5 -> 0.6  -> steadier, less stochastic noise
    * inference_steps 15 -> 20   -> more "thinking" per level, sharper boundaries
  NOT changed (proven NOT to help): no 3-level, no extra params, no >150 epochs.

USAGE (one Kaggle cell, GPU on, BraTS attached):
    !rm -rf GLO-NCA && git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_v4.py
================================================================================
"""
import os, sys, time, json, math, random, subprocess

# ---- core knobs ----
EPOCHS      = 150
N_PATIENTS  = None
SEED        = 42
CHANNEL_N   = 24
HIDDEN      = 96
STEPS       = [20, 20]          # v3 was [15,15]
FIRE_RATE   = 0.6              # v3 was 0.5
LR_START    = 16e-4
LR_MIN      = 1e-5
# higher-res patches (low-res level, high-res level). Crop makes 64^3 affordable.
INPUT_SIZE  = [[32, 32, 24], [64, 64, 48]]
TVERSKY_BETA = 0.75           # v3 was 0.70

# ---- preprocessing (keep v3's wins) ----
USE_FOREGROUND_CROP = True
USE_NONZERO_NORM    = True
USE_AUG             = False    # OFF in v4 (was the convergence drag in v3)

# ---- speed ----
NUM_WORKERS = 4
BATCH_SIZE  = 1               # 64^3 + steps20 is heavy; batch 1 is the safe default
MULTI_GPU   = False           # batch 1 -> single GPU (DataParallel needs batch>=2)

REEVAL_ONLY = False
REGIONS     = ["WT", "TC", "ET"]
MODALITIES  = ["t1n", "t1c", "t2w", "t2f"]

REPO_URL = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"
REPO_DIR = "/kaggle/working/GLO-NCA"
OUT_DIR  = "/kaggle/working/glo_v4"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchio"], check=False)
if not os.path.isdir(os.path.join(REPO_DIR, "src")):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import TverskyCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.agents.Agent import iou_score, hd95_score


class BraTS_FG(Dataset_NiiGz_3D_BraTS):
    @staticmethod
    def _foreground_bbox(vol_stack):
        fg = np.any(vol_stack > 0, axis=-1)
        if not fg.any():
            return None
        xs = np.where(fg.any(axis=(1, 2)))[0]
        ys = np.where(fg.any(axis=(0, 2)))[0]
        zs = np.where(fg.any(axis=(0, 1)))[0]
        return xs[0], xs[-1] + 1, ys[0], ys[-1] + 1, zs[0], zs[-1] + 1

    def _resize_to(self, vol, size, is_label=False):
        interp = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
        out = np.zeros((size[0], size[1], vol.shape[2]), np.float32)
        for z in range(vol.shape[2]):
            out[:, :, z] = cv2.resize(vol[:, :, z], dsize=(size[1], size[0]), interpolation=interp)
        tmp, out = out, np.zeros(size, np.float32)
        for y in range(tmp.shape[1]):
            out[:, y, :] = cv2.resize(tmp[:, y, :], dsize=(size[2], size[0]), interpolation=interp)
        return out

    def __getitem__(self, idx):
        key = self.images_list[idx]
        cached = self.data.get_data(key=key)
        if not cached:
            folder_name, p_id, _ = key
            folder = os.path.join(self.images_path, folder_name)
            raw = np.stack([self.load_item(self._find_modality_file(folder, folder_name, m))
                            for m in self.MODALITIES], axis=-1)
            seg = self.load_item(self._find_modality_file(folder, folder_name, self.SEG_SUFFIX))
            if USE_FOREGROUND_CROP:
                bbox = self._foreground_bbox(raw)
                if bbox is not None:
                    x0, x1, y0, y1, z0, z1 = bbox
                    raw = raw[x0:x1, y0:y1, z0:z1, :]; seg = seg[x0:x1, y0:y1, z0:z1]
            size = tuple(self.size)
            img = np.stack([self._resize_to(raw[..., c], size) for c in range(raw.shape[-1])], axis=-1)
            seg = self._resize_to(seg, size, is_label=True)
            label = self._labels_to_regions(seg)
            self.data.set_data(key=key, data=("_" + str(p_id) + "_0", img, label))
            cached = self.data.get_data(key=key)

        img_id, img, label = cached
        if self.exp.get_from_config('patchify') is True and self.state == "train":
            img, label = self.patchify_multimodal(img, label)
        # (no augmentation in v4)
        if USE_NONZERO_NORM:
            out = np.empty_like(img, dtype=np.float32)
            for c in range(img.shape[-1]):
                ch = img[..., c]; mask = ch > 0
                out[..., c] = np.where(mask, (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8), 0.0) \
                    if mask.sum() > 0 else ch
            img = out
        return (img_id, img.astype(np.float32), label.astype(np.float32))


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
                acc[r]["dice"].append((2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6))
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
    print(f"v4: ch={CHANNEL_N} hidden={HIDDEN} steps={STEPS} fire={FIRE_RATE} patch={INPUT_SIZE[-1]} "
          f"beta={TVERSKY_BETA} aug={USE_AUG} ep={EPOCHS}", flush=True)

    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(OUT_DIR, "m"),
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": LR_START, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": EPOCHS,
        "batch_size": BATCH_SIZE, "batch_duplication": 1,
        "channel_n": CHANNEL_N, "inference_steps": STEPS, "cell_fire_rate": FIRE_RATE,
        "input_channels": 4, "output_channels": 3, "hidden_size": HIDDEN,
        "train_model": 1, "use_attention": True,
        "input_size": INPUT_SIZE, "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.5,
    }]
    ds = BraTS_FG(); ds.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(CHANNEL_N, FIRE_RATE, dev, HIDDEN, kernel_size=7, input_channels=4, use_attention=True),
          BasicNCA3D(CHANNEL_N, FIRE_RATE, dev, HIDDEN, kernel_size=3, input_channels=4, use_attention=True)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent); ds.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")

    spe = max(1, math.ceil(len(tr) / BATCH_SIZE)); total = EPOCHS * spe
    agent.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=LR_MIN)
                       for opt in agent.optimizer]

    loss_f = TverskyCELoss(alpha=1 - TVERSKY_BETA, beta=TVERSKY_BETA, ce_weight=0.5)
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params, "| device:", dev, flush=True)

    hist = {"epoch": [], "loss": [], "lr": [], "val_mean": [], "val_WT": [], "val_TC": [], "val_ET": []}
    best, best_path = -1.0, os.path.join(OUT_DIR, "best.pth")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if REEVAL_ONLY:
        assert os.path.exists(best_path), f"No checkpoint at {best_path}"
        print("REEVAL_ONLY: loading existing best.pth", flush=True)

    print("Loading + caching volumes (slow first pass)...", flush=True)
    for ep in (range(EPOCHS) if not REEVAL_ONLY else []):
        losses = []
        loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKERS, pin_memory=True)
        for i, data in enumerate(loader):
            r = agent.batch_step(data, loss_f)
            if r:
                losses.append(sum(r.values()))
            if ep == 0 and (i + 1) % 20 == 0:
                print(f"  [epoch 1] {i+1}/{spe} ({time.time()-t0:.0f}s)...", flush=True)
        cur_lr = agent.optimizer[0].param_groups[0]["lr"]
        val = evaluate(agent, ds, "val")
        vm = float(np.mean([val[r]["dice"] for r in REGIONS]))
        hist["epoch"].append(ep+1); hist["loss"].append(float(np.mean(losses)) if losses else 0)
        hist["lr"].append(cur_lr); hist["val_mean"].append(vm)
        for r in REGIONS:
            hist[f"val_{r}"].append(val[r]["dice"])
        print(f"ep {ep+1}/{EPOCHS} | lr {cur_lr:.2e} | loss {hist['loss'][-1]:.3f} | "
              f"val mean {vm:.3f} (WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} "
              f"ET {val['ET']['dice']:.3f})", flush=True)
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
    print(f"GLO-NCA v4 — FINAL TEST (best @ epoch {ck['ep']})")
    print("=" * 60)
    print(f"{'region':<8}{'Dice':<12}{'mIoU':<12}{'HD95':<12}")
    for r in REGIONS:
        print(f"{r:<8}{test[r]['dice']:<12.4f}{test[r]['iou']:<12.4f}{test[r]['hd95']:<12.3f}")
    mean = np.mean([test[r]['dice'] for r in REGIONS])
    print("-" * 60)
    print(f"{'mean':<8}{mean:<12.4f}")
    print(f"train time {train_time:.0f}s | peak VRAM {peak:.2f} GB | params {n_params}")

    json.dump({"test": test, "history": hist, "best_epoch": ck["ep"], "params": n_params,
               "train_time": train_time, "peak_vram": peak,
               "config": {"steps": STEPS, "fire": FIRE_RATE, "patch": INPUT_SIZE,
                          "beta": TVERSKY_BETA, "aug": USE_AUG}},
              open(os.path.join(OUT_DIR, "v4_results.json"), "w"), indent=2, default=str)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["epoch"], hist["loss"], color="crimson"); a1.set_title("Loss"); a1.set_xlabel("epoch")
    for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
        a2.plot(hist["epoch"], hist[f"val_{r}"], label=f"val {r}", color=c)
    a2.plot(hist["epoch"], hist["val_mean"], "--k", label="mean")
    a2.set_title("Validation Dice (v4)"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "v4_curves.png"), dpi=130); plt.show()
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
