# GLO-NCA: Global Context-Aware Neural Cellular Automata for Brain Tumor Segmentation

> **MS Thesis — Muhammad Waqas, Air University, Islamabad.**
> *For academic and educational use only — not a medical device (see [LICENSE](LICENSE)).*

GLO-NCA segments brain tumors from **multi-modal MRI (BraTS)** using a
lightweight **Neural Cellular Automata** with a **global-context mechanism**.
A plain NCA only communicates locally; GLO-NCA adds a cheap Squeeze-and-
Excitation (SE) block that injects whole-volume context, so the model captures
both **local boundaries** and **global tumor location** while staying tiny
(~30k parameters, fits a 6 GB GPU).

---

## Why GLO-NCA
- **Lightweight** — ~30k parameters vs millions in U-Net / Transformers; runs on
  low-resource hardware (laptop GPU, edge devices).
- **Global-context aware** — a channel Squeeze-and-Excitation block
  (`use_attention=True`) plus a spatial global-context block (`use_spatial=True`)
  inject whole-volume context for a negligible parameter cost; this is the
  thesis novelty.
- **Multi-modal** — fuses the four BraTS modalities (T1, T1ce, T2, FLAIR).
- **Multi-class** — predicts the three standard nested tumor regions
  **WT** (Whole Tumor), **TC** (Tumor Core), **ET** (Enhancing Tumor).
- **Built-in quality control** — variance over stochastic inferences (NQM).

## Method (coarse-to-fine)
1. Downscale the volume and run an NCA on the **full low-res** image → global context.
2. Upscale its features, concatenate with the higher-res image, run a second NCA
   on a **random patch** (keeps VRAM low during training; full volume at inference).
3. Read out 3 channels → sigmoid → WT / TC / ET.

Loss: **Focal-Tversky + BCE** (`FocalTverskyCELoss`), tuned for the small,
imbalanced ET/TC regions. Optimizer: **AdamW** with cosine LR decay, EMA
weights and gradient-norm clipping. Metrics: **Dice**, **mIoU**, **HD95**
per region.

---

## Repository layout
```
.
├── src/                       # GLO-NCA pipeline (all code needed to run it)
│   ├── agents/                #   Agent, Agent_NCA, Agent_Multi_NCA, Agent_GLO_NCA
│   ├── datasets/              #   BraTS loader (Nii_Gz_Dataset_3D: Dataset_NiiGz_3D_BraTS)
│   ├── models/                #   Model_BasicNCA3D (+ SE & spatial global-context blocks)
│   ├── losses/                #   FocalTverskyCELoss, TverskyCELoss, DiceCELoss
│   └── utils/                 #   Experiment, helper
├── train.py                   # canonical training entry point (env-driven)
├── Dockerfile                 # GPU training image for GCP
├── .dockerignore
├── requirements-docker.txt    # pinned runtime dependencies
├── kaggle_v4..v7.py           # experiment history (v4 = proven best, v7 = final recipe)
└── LICENSE
```

## Installation (Python 3.12)
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r requirements-docker.txt
```

## Dataset
Uses the **BraTS** dataset (one folder per patient with the modality + seg
volumes). Tested on **BraTS 2024** (`*-t1n / -t1c / -t2w / -t2f / -seg .nii`);
the loader also handles the older `t1 / t1ce / t2 / flair` naming.

## Run (canonical: `train.py`)
`train.py` is the professional entry point (the full v7 recipe, single clean
inference — no ensemble/TTA). Paths come from the environment, so the same code
runs on a laptop, Kaggle or a GCP VM without edits:

```bash
DATA_ROOT=/path/to/BraTS OUT_DIR=./out python train.py
```

Optional overrides (env vars):

| Var | Default | Meaning |
|-----|---------|---------|
| `EPOCHS` | 150 | training epochs |
| `N_PATIENTS` | 0 (=all) | cap patients (use a small value for a cheap Kaggle test) |
| `PATCH` | 96 | high-res patch: **64** (laptop/Kaggle), **96** (balanced), **128** (max context, GCP-class GPU only — can OOM 6 GB) |
| `AUG_LEVEL` | heavy | `light` (flips/rot/intensity) or `heavy` (+ elastic/gamma/noise/blur) |
| `SEED`, `BATCH_SIZE`, `NUM_WORKERS` | 42 / 1 / 4 | reproducibility & loader |

It writes `best.pth`, `results.json` and `curves.png` to `OUT_DIR`, prints
**Dice / mIoU / HD95** per region (at 0.5 and at val-tuned thresholds), and ends
with a **diagnostic report** — one GOOD/OK/WATCH verdict per signal (overfitting,
threshold gain, convergence, weakest region, best-epoch position, VRAM headroom)
so a cheap run tells you what to change before a full GCP run.

**Cheap Kaggle test first:** `N_PATIENTS=40 EPOCHS=20 PATCH=64 python train.py`
— confirms it trains and reads the diagnostics, then scale up on GCP.

The `kaggle_v4..v7.py` scripts are kept as the experiment history that led to
this recipe; `train.py` supersedes them.

## Run on GCP (Docker)
```bash
docker build -t glo-nca .

docker run --gpus all \
    -e DATA_ROOT=/data -e OUT_DIR=/out -e EPOCHS=150 \
    -v /mnt/brats:/data:ro \
    -v /mnt/checkpoints:/out \
    glo-nca
```
Mount the BraTS dataset at `/data` (read-only) and a writable checkpoint dir at
`/out` — on GCP these can be a persistent disk or a GCS bucket via `gcsfuse`.
Nothing about the data is baked into the image.

## Ablation (baseline vs GLO-NCA)
Set `use_attention: False` to get the plain multi-level NCA baseline, and
`True` for the global-context model — same code, one flag — to measure exactly
what the global-context block contributes.

---

## Acknowledgements
Built on the open-source **Med-NCA / M3D-NCA** framework by John Kalkhof et al.
(MIT-licensed). The global-context design, BraTS multi-modal/multi-class
pipeline, and evaluation are the thesis contributions.

- Kalkhof et al., *Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata*, IPMI 2023.
- Kalkhof & Mukhopadhyay, *M3D-NCA: Robust 3D Segmentation with Built-In Quality Control*, MICCAI 2023.

## License
MIT, for **academic/educational use only** — see [LICENSE](LICENSE). Not for
clinical use.

