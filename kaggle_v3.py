r"""
================================================================================
GLO-NCA v3 — v2 + full Swin-UNETR pipeline + dual-GPU + fast loading
================================================================================
Builds on the proven v2 recipe and adds the verified Swin-UNETR techniques.
Each addition is a TOGGLE so you can disable anything that hurts (augmentation
and the LR/loss swap are the risky ones on a 140-patient set).

  KEPT from v2:  2-level cascade, ch24, hidden96, steps[15,15], cosine LR, Tversky
  ADDED (toggles below):
    USE_FOREGROUND_CROP : crop to brain bbox before resize         (verified, safe)
    USE_NONZERO_NORM    : z-norm on brain voxels per channel        (verified, safe)
    USE_AUG             : random flips + intensity scale/shift       (risky on small data)
    USE_SLIDING_WINDOW  : tiled full-volume inference at test        (safe, test-time)
    SWIN_HYPERPARAMS    : LR 1e-4 + DiceLoss(sigmoid) (their exact)  (risky swap)
  SPEED:
    NUM_WORKERS         : parallel CPU data loading (your real bottleneck)
    BATCH_SIZE / MULTI_GPU : use both Kaggle T4s by wrapping each NCA in
                          DataParallel and training with batch>=2

USAGE (one Kaggle cell, GPU on, BraTS attached):
    !rm -rf GLO-NCA && git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_v3.py
================================================================================
"""
import os, sys, time, json, math, random, subprocess

# ---- core knobs (v2) ----
EPOCHS      = 150
N_PATIENTS  = None
SEED        = 42
CHANNEL_N   = 24
HIDDEN      = 96
STEPS       = [15, 15]
LR_START    = 16e-4
LR_MIN      = 1e-5

# ---- Swin-UNETR technique toggles ----
USE_FOREGROUND_CROP = True     # verified, safe — biggest TC/ET win
USE_NONZERO_NORM    = True     # verified, safe
USE_AUG             = True     # flips + intensity (you asked for all; set False if it hurts)
USE_SLIDING_WINDOW  = False    # OFF: tiling small post-crop volumes washed out TC/ET
                               #      (test must use the SAME inference as validation)
SWIN_HYPERPARAMS    = False    # True = LR 1e-4 + DiceLoss(sigmoid) (their exact recipe)

# ---- speed knobs ----
NUM_WORKERS = 4                # parallel CPU loading (real bottleneck)
BATCH_SIZE  = 2                # >=2 needed to use 2 GPUs
MULTI_GPU   = True             # wrap each NCA in DataParallel across both T4s

# ---- re-evaluate an already-trained best.pth WITHOUT retraining ----
# If you still have /kaggle/working/glo_v3/best.pth from a previous run, set
# REEVAL_ONLY = True to just re-score it (e.g. after changing inference toggles).
REEVAL_ONLY = False

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
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import TverskyCELoss, DiceCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.agents.Agent import iou_score, hd95_score


# ============================================================================
# v3 dataset: foreground crop + nonzero z-norm + (optional) augmentation
# ============================================================================
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

    def _augment(self, img, label):
        # RandFlip on all 3 axes
        for ax in (0, 1, 2):
            if random.random() < 0.5:
                img = np.flip(img, axis=ax); label = np.flip(label, axis=ax)
        # RandScaleIntensity + RandShiftIntensity (Swin-UNETR factors/offsets 0.1)
        if random.random() < 1.0:
            img = img * np.float32(1.0 + random.uniform(-0.1, 0.1))
        if random.random() < 1.0:
            img = img + np.float32(random.uniform(-0.1, 0.1))
        return np.ascontiguousarray(img), np.ascontiguousarray(label)

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
                    raw = raw[x0:x1, y0:y1, z0:z1, :]
                    seg = seg[x0:x1, y0:y1, z0:z1]

            size = tuple(self.size)
            img = np.stack([self._resize_to(raw[..., c], size) for c in range(raw.shape[-1])], axis=-1)
            seg = self._resize_to(seg, size, is_label=True)
            label = self._labels_to_regions(seg)
            self.data.set_data(key=key, data=("_" + str(p_id) + "_0", img, label))
            cached = self.data.get_data(key=key)

        img_id, img, label = cached

        if self.exp.get_from_config('patchify') is True and self.state == "train":
            img, label = self.patchify_multimodal(img, label)
        if USE_AUG and self.state == "train":
            img, label = self._augment(img, label)

        if USE_NONZERO_NORM:
            out = np.empty_like(img, dtype=np.float32)
            for c in range(img.shape[-1]):
                ch = img[..., c]; mask = ch > 0
                if mask.sum() > 0:
                    out[..., c] = np.where(mask, (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8), 0.0)
                else:
                    out[..., c] = ch
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
N_GPU = torch.cuda.device_count()
print(f"GPUs available: {N_GPU}")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_split(seed):
    pats = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    random.Random(seed).shuffle(pats)
    if N_PATIENTS:
        pats = pats[:N_PATIENTS]
    n = len(pats); a, b = int(n*0.70), int(n*0.15)
    return pats[:a], pats[a:a+b], pats[a+b:]


# ---- sliding-window inference (tiled full-volume) ----
def sliding_window_predict(agent, data, patch, overlap=0.5):
    """Run get_outputs over overlapping patches and average. data already prepared."""
    id_, inputs, targets = data
    B, X, Y, Z, C = inputs.shape
    px, py, pz = patch
    sx, sy, sz = max(1, int(px*(1-overlap))), max(1, int(py*(1-overlap))), max(1, int(pz*(1-overlap)))
    out_ch = agent.output_channels
    acc = torch.zeros(B, X, Y, Z, out_ch, device=inputs.device)
    cnt = torch.zeros(B, X, Y, Z, 1, device=inputs.device)
    xs = list(range(0, max(1, X-px+1), sx)) or [0]
    ys = list(range(0, max(1, Y-py+1), sy)) or [0]
    zs = list(range(0, max(1, Z-pz+1), sz)) or [0]
    if xs[-1] != X-px: xs.append(max(0, X-px))
    if ys[-1] != Y-py: ys.append(max(0, Y-py))
    if zs[-1] != Z-pz: zs.append(max(0, Z-pz))
    for x in xs:
        for y in ys:
            for z in zs:
                sub_in = inputs[:, x:x+px, y:y+py, z:z+pz, :]
                sub_tg = targets[:, x:x+px, y:y+py, z:z+pz] if targets.dim() == 4 else targets[:, x:x+px, y:y+py, z:z+pz, :]
                o, _ = agent.get_outputs((id_, sub_in, sub_tg), full_img=True)
                acc[:, x:x+px, y:y+py, z:z+pz, :] += o
                cnt[:, x:x+px, y:y+py, z:z+pz, :] += 1
    return acc / cnt.clamp(min=1)


def evaluate(agent, dataset, state, sliding=False, patch=(56, 56, 40)):
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            if sliding:
                _, _, targets = data
                outputs = sliding_window_predict(agent, data, patch)
                if targets.dim() == 4:
                    targets = targets.unsqueeze(-1)
            else:
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
    lr0 = 1e-4 if SWIN_HYPERPARAMS else LR_START
    print(f"\nSplit -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print(f"v3: ch={CHANNEL_N} hidden={HIDDEN} steps={STEPS} ep={EPOCHS} batch={BATCH_SIZE} | "
          f"crop={USE_FOREGROUND_CROP} nzNorm={USE_NONZERO_NORM} aug={USE_AUG} "
          f"sw={USE_SLIDING_WINDOW} swinHP={SWIN_HYPERPARAMS} | GPUs={N_GPU} multiGPU={MULTI_GPU}",
          flush=True)

    use_mgpu = MULTI_GPU and N_GPU > 1
    eff_batch = max(BATCH_SIZE, 2) if use_mgpu else BATCH_SIZE

    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": os.path.join(OUT_DIR, "m"),
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": lr0, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-5 if SWIN_HYPERPARAMS else 1e-4,
        "save_interval": 10**9, "evaluate_interval": 10**9, "n_epoch": EPOCHS,
        "batch_size": eff_batch, "batch_duplication": 1,
        "channel_n": CHANNEL_N, "inference_steps": STEPS, "cell_fire_rate": 0.5,
        "input_channels": 4, "output_channels": 3, "hidden_size": HIDDEN,
        "train_model": 1, "use_attention": True,
        "input_size": [[28, 28, 20], [56, 56, 40]], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.5,
    }]
    ds = BraTS_FG(); ds.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(CHANNEL_N, 0.5, dev, HIDDEN, kernel_size=7, input_channels=4, use_attention=True),
          BasicNCA3D(CHANNEL_N, 0.5, dev, HIDDEN, kernel_size=3, input_channels=4, use_attention=True)]

    # ---- dual-GPU: wrap each NCA's internal conv/MLP forward across both T4s ----
    if use_mgpu:
        for m in ca:
            m.p0 = torch.nn.DataParallel(m.p0)
            m.fc0 = torch.nn.DataParallel(m.fc0)
            m.fc1 = torch.nn.DataParallel(m.fc1)
        print("DataParallel enabled on NCA sublayers across", N_GPU, "GPUs", flush=True)

    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, ds, ca, agent); ds.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", tr), ("val", va), ("test", te)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")

    spe = max(1, math.ceil(len(tr) / eff_batch)); total = EPOCHS * spe
    agent.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=LR_MIN)
                       for opt in agent.optimizer]

    loss_f = DiceCELoss() if SWIN_HYPERPARAMS else TverskyCELoss(alpha=0.3, beta=0.7, ce_weight=0.5)
    n_params = sum(p.numel() for m in ca for p in m.parameters())
    print("Trainable params:", n_params, flush=True)

    hist = {"epoch": [], "loss": [], "lr": [], "val_mean": [], "val_WT": [], "val_TC": [], "val_ET": []}
    best, best_path = -1.0, os.path.join(OUT_DIR, "best.pth")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    if REEVAL_ONLY:
        print("REEVAL_ONLY: skipping training, loading existing best.pth ...", flush=True)
        assert os.path.exists(best_path), f"No checkpoint at {best_path} to re-evaluate."

    print("Loading + caching volumes (slow first pass)...", flush=True)
    for ep in (range(EPOCHS) if not REEVAL_ONLY else []):
        losses = []
        loader = torch.utils.data.DataLoader(ds, shuffle=True, batch_size=eff_batch,
                                             num_workers=NUM_WORKERS, pin_memory=True)
        for i, data in enumerate(loader):
            r = agent.batch_step(data, loss_f)
            if r:
                losses.append(sum(r.values()))
            if ep == 0 and (i + 1) % 10 == 0:
                print(f"  [epoch 1] batch {i+1}/{spe} ({time.time()-t0:.0f}s)...", flush=True)
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
    test = evaluate(agent, ds, "test", sliding=USE_SLIDING_WINDOW, patch=(56, 56, 40))

    print("\n" + "=" * 60)
    print(f"GLO-NCA v3 — FINAL TEST (best @ epoch {ck['ep']}"
          f"{', sliding-window' if USE_SLIDING_WINDOW else ''})")
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
               "toggles": {"crop": USE_FOREGROUND_CROP, "nzNorm": USE_NONZERO_NORM,
                           "aug": USE_AUG, "sw": USE_SLIDING_WINDOW, "swinHP": SWIN_HYPERPARAMS}},
              open(os.path.join(OUT_DIR, "v3_results.json"), "w"), indent=2, default=str)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["epoch"], hist["loss"], color="crimson"); a1.set_title("Loss"); a1.set_xlabel("epoch")
    for r, c in zip(REGIONS, ["#1f77b4", "#2ca02c", "#9467bd"]):
        a2.plot(hist["epoch"], hist[f"val_{r}"], label=f"val {r}", color=c)
    a2.plot(hist["epoch"], hist["val_mean"], "--k", label="mean")
    a2.set_title("Validation Dice (v3 full)"); a2.set_ylim(0, 1); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "v3_curves.png"), dpi=130); plt.show()
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
