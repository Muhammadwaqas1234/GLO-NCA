# GLO-NCA: Global Context-Aware Neural Cellular Automata for Brain Tumor Segmentation

> **MS Thesis — Muhammad Waqas, Air University, Islamabad.**
> *For academic and educational use only — not a medical device (see [LICENSE](LICENSE)).*

GLO-NCA segments brain tumors from **multi-modal MRI (BraTS)** using a
lightweight **Neural Cellular Automata** with a **global-context mechanism**.
A plain NCA only communicates locally; GLO-NCA adds a cheap Squeeze-and-
Excitation (SE) block that injects whole-volume context, so the model captures
both **local boundaries** and **global tumor location** while staying tiny
(~13k parameters, fits a 6 GB GPU).

---

## Why GLO-NCA
- **Lightweight** — ~13k parameters vs millions in U-Net / Transformers; runs on
  low-resource hardware (laptop GPU, edge devices).
- **Global-context aware** — the SE block (`use_attention=True`) adds whole-
  volume context for only ~4% extra parameters; this is the thesis novelty.
- **Multi-modal** — fuses the four BraTS modalities (T1, T1ce, T2, FLAIR).
- **Multi-class** — predicts the three standard nested tumor regions
  **WT** (Whole Tumor), **TC** (Tumor Core), **ET** (Enhancing Tumor).
- **Built-in quality control** — variance over stochastic inferences (NQM).

## Method (coarse-to-fine)
1. Downscale the volume and run an NCA on the **full low-res** image → global context.
2. Upscale its features, concatenate with the higher-res image, run a second NCA
   on a **random patch** (keeps VRAM low during training; full volume at inference).
3. Read out 3 channels → sigmoid → WT / TC / ET.

Loss: **Dice + Cross-Entropy** (`DiceCELoss`). Optimizer: **AdamW**.
Metrics: **Dice**, **mIoU**, **HD95** per region.

---

## Repository layout
```
.
├── src/                         # GLO-NCA pipeline (the only code you need to run it)
│   ├── agents/                  #   Agent, Agent_NCA, Agent_Multi_NCA, Agent_GLO_NCA
│   ├── datasets/                #   BraTS loader (Nii_Gz_Dataset_3D: Dataset_NiiGz_3D_BraTS)
│   ├── models/                  #   Model_BasicNCA3D (+ SE global-context block)
│   ├── losses/                  #   DiceCELoss
│   ├── utils/                   #   Experiment, helper
│   └── examples/train_GLO_NCA.py
├── train_GLO_NCA.ipynb          # main notebook (edit paths, run)
├── extra/                       # non-GLO code kept for reference (UNet/Med-NCA/tutorials)
├── requirements.txt
└── LICENSE
```

## Installation (Python 3.12)
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r requirements.txt
```

## Dataset
Uses the **BraTS** dataset (one folder per patient with the modality + seg
volumes). Tested on **BraTS 2024** (`*-t1n / -t1c / -t2w / -t2f / -seg .nii`);
the loader also handles the older `t1 / t1ce / t2 / flair` naming.

## Run
Edit the paths in `train_GLO_NCA.ipynb` (or `src/examples/train_GLO_NCA.py`) —
set **both** `img_path` and `label_path` to the BraTS dataset root — then run.

```python
# in the config
'img_path':   r"/path/to/BraTS2024_small_dataset",
'label_path': r"/path/to/BraTS2024_small_dataset",
'input_channels': 4,    # T1, T1ce, T2, FLAIR
'output_channels': 3,   # WT, TC, ET
'use_attention': True,  # global-context SE block (the novelty)
```

Evaluation prints **Dice / mIoU / HD95** per region on the test split.

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
