# GLO-NCA Cascade

**GLO-NCA** (Global Context-Aware Neural Cellular Automata) for brain-tumour segmentation
on BraTS: a two-level NCA cascade with channel (SE) and spatial gating blocks and
multi-modal MRI fusion, in about **30,000 parameters**. This repository is the training
package (configs, resumable training, evaluation, figures, CI) and its GCP tooling.

The model's context blocks are channel recalibration (SE, pooled over the whole volume) and a local
7×7×7 spatial gate in every cell. An explicit attention-pooled global-context block is implemented as
an optional variant (`model.use_global`); it is not part of the reported model or of this study.

## Contributions of this work

The package started from an existing neural cellular automata framework (see `LICENSE`). The following
are contributions of this work:

| Area | This work |
|---|---|
| Model | The GLO-NCA cell: squeeze-and-excitation channel recalibration and a spatial gate in every NCA cell, an optional attention-pooled global-context block, and the 48³ → 96³ cascade configuration used here |
| Multi-modal input | Fusion of T1, contrast-enhanced T1, T2 and FLAIR as input channels of the cell state |
| Training recipe | BraTS 2021 pipeline: foreground crop and resampling, brain-mask normalisation, focal Tversky + BCE loss over every region, ET-centred training crops, EMA weights with matching selection, augmentation |
| Evaluation | Full-resolution scoring with a shared nested decoder, surface HD95 in named conventions, validation-tuned post-processing, an ET relabelling rule, and submission to the official BraTS 2023 Synapse evaluation (mean Dice 0.854) |
| Analysis | Ablation design (global context: SE, spatial gate and coarse level removed together; multi-modal fusion with kept input slots), paired patient-level statistics, inference decomposition, step budget, receptive-field and perturbation analyses, cost benchmark |
| Engineering | Installable package, resumable cloud training, reproducibility safeguards (hashes, provenance, preflight), 45 unit and 22 pipeline checks, an equivalence test against the original script, and the published weights |

## Research questions

1. **Accuracy at a small size.** How close does a two-level NCA cascade with about 30,000 parameters
   come to much larger models in adult glioma segmentation (WT, TC, ET) on multi-parametric MRI,
   measured as the gap in mean Dice to published results on the same official validation set? Internal
   test (189 patients) and the official BraTS 2023 validation on Synapse (219 cases); the published
   results are context, not a controlled comparison.
2. **Global context.** Do the model's sources of context beyond the local neighbourhood (SE, the
   spatial gate and the coarse cascade level) improve segmentation over the fine NCA level alone?
   `brats2021_res96_fixed_noglobal.yaml`.
3. **Multi-modal fusion.** Do the four MRI modalities help compared with contrast-enhanced T1 alone?
   `brats2021_res96_fixed_single.yaml` and `glo-analyze modalities`.
4. **Information transport.** How far does information travel in the model, and how do accuracy and run
   time depend on the number of NCA steps? `glo-analyze erf`, `perturb` and `steps`.
5. **Cost and inference choices.** What does the model cost to run, and how much accuracy comes from the
   stochastic ensemble, flips and post-processing rather than the architecture? `benchmark.py` and
   `glo-analyze decompose`.

Each of the three runs is trained once (one seed) with the same split and frozen protocol, so the
comparisons are exploratory. Larger models are compared through their published results on the
BraTS 2023 adult glioma validation set, not retrained. Data: BraTS 2021 adult glioma for training and
the internal test; BraTS 2023 adult glioma validation for the external evaluation.

## Results

BraTS 2021: 1,251 glioma cases, seeded random split of 875 training, 187 validation and
189 test patients. Configuration `configs/experiments/brats2021_res96_long.yaml`
(96³ working resolution, augmentation, 150 epochs); best validation epoch 137 (mean Dice 0.830).
The test patients were never used for training, model selection or tuning.

| Test setting (189 patients) | WT Dice | TC Dice | ET Dice | Mean Dice |
|---|---|---|---|---|
| Single pass, 96³ | 0.893 | 0.863 | 0.754 | 0.837 |
| Ensemble ×4 + test-time flips, 96³ | 0.894 | 0.864 | 0.756 | 0.838 |
| **Ensemble ×4 + flips, post-processed, full resolution** | **0.921** | **0.895** | **0.817** | **0.878** |

| Region (post-processed, full resolution) | IoU | HD95 mean (mm) | HD95 median (mm) | Median Dice |
|---|---|---|---|---|
| WT | 0.860 | 4.93 | 2.24 | 0.944 |
| TC | 0.834 | 5.23 | 1.73 | 0.949 |
| ET | 0.723 | 4.49 (184 of 189 cases) | 2.00 | 0.872 |

- The first two rows are scored on the 96³ grid with the v7 Dice formula (an absent region scores 0).
- The last row and the second table come from one evaluation (`evaluate.py --hd95-surface`):
  thresholds WT/TC/ET 0.65/0.65/0.60 and small-lesion clean-up 400/25/400 voxels, both chosen
  on validation; predictions resampled to the original 240×240×155 scans; BraTS scoring, where an
  empty prediction of an absent region scores Dice 1.
- HD95 is surface-based, in mm, in the medpy convention (one 95th percentile over the surface
  distances of both directions, as `medpy.metric.binary.hd95`); for ET it is undefined in the 5 cases
  where exactly one of the two masks is empty.
- The ensemble passes are stochastic: the training run's own final test of the same setting gave
  0.880 mean Dice (0.921 / 0.896 / 0.823).
- Training: 55.8 h on one NVIDIA L4 (GCP Spot, no preemptions); peak GPU memory 10.4 GB.
- Run history: the run was paused and resumed with changed settings (`train.log`). It validated
  every 5 epochs until epoch 38 and every epoch from epoch 39, which is how epoch 137 was selected;
  early stopping was 20, then 40, then off from epoch 42.

| Figure | Content |
|---|---|
| Training curves | training loss, learning rate and validation Dice per epoch |
| Test summary | the three test settings above and the official HD95 |
| Per-case Dice | every test patient, post-processed at full resolution |
| Example segmentations | best, median and worst test patient on the 96³ grid (ensemble ×4 + flips) |

![Training curves](docs/figures/training_curves.png)
![Test summary](docs/figures/test_summary.png)
![Per-case Dice](docs/figures/test_per_case.png)
![Example segmentations](docs/figures/test_examples.png)

## Official BraTS 2023 validation (Synapse leaderboard)

The model above was submitted to the **BraTS 2023 Adult Glioma continuous evaluation** on Synapse
(219 validation cases, labels withheld, scored by the organisers; submission 9783550), with the same
setting as the post-processed test row: ensemble ×4 + flips, thresholds 0.65/0.65/0.60, clean-up
400/25/400 voxels, full resolution (`predict.py`).

| Metric | WT | TC | ET | Mean |
|---|---|---|---|---|
| Dice | 0.909 | 0.855 | 0.784 | 0.849 |
| HD95 (mm) | 6.37 | 9.41 | 22.98 | 12.92 |
| Lesion-wise Dice | 0.879 | 0.831 | 0.775 | 0.828 |
| Lesion-wise HD95 (mm) | 17.51 | 20.31 | 30.54 | 22.79 |
| Sensitivity | 0.921 | 0.869 | 0.840 | |
| Specificity | 0.999 | 0.999 | 0.999 | |

With ET post-processing (`predict.py --et-to-ncr 0.25,5000`: a predicted ET that is below 5,000 voxels
and below 25% of the tumour core is relabelled as necrotic core; chosen from the validation results),
submission 9783820:

| Metric | WT | TC | ET | Mean |
|---|---|---|---|---|
| Dice | 0.909 | 0.855 | 0.797 | 0.854 |
| HD95 (mm) | 6.37 | 9.41 | 24.29 | 13.36 |
| Lesion-wise Dice | 0.879 | 0.831 | 0.794 | 0.834 |
| Lesion-wise HD95 (mm) | 17.51 | 20.31 | 27.73 | 21.85 |

On the same validation set, the BraTS 2021 and 2022 winners (nnU-Net based ensembles) report Dice
0.928/0.878/0.845 and 0.927/0.875/0.844 (WT/TC/ET).

Per-case Dice below 0.50 (standard Dice): WT 1, TC 9, ET 22 of 219 cases; Dice 0: TC 2, ET 13.
Of the 13 ET zeros, 7 are cases without enhancing tumour where ET was predicted, 3 are missed ET,
and 3 are misplaced ET; an ET-volume relabelling rule alone changes the ET mean by at most +0.005,
because these false ET regions are as large as true small ones. Per-case scores with predicted
volumes: `docs/results/brats2023_validation_per_case.csv`.

## Model

| | |
|---|---|
| Cascade | 2 NCA levels, coarse then fine (48³ → 96³ for the results above; 32×32×24 → 64×64×48 in v7) |
| NCA cell | 24 channels, hidden 128, 20 + 20 steps, update probability 0.4 (config `fire_rate`/`skip_prob` 0.6 is the skip probability), dropout 0.1, kernels 7 / 3 |
| Context blocks | squeeze-and-excitation (global over channels) + local 7×7×7 spatial gate in every cell; optional attention-pooled global-context block (`model.use_global`), added to the hidden layer after normalisation |
| Fusion | T1, T1ce, T2 and FLAIR stacked as input channels; coarse features fused into the fine level |
| Data | foreground crop, cubic resize, nonzero z-norm, random 48³ training crops (ET-centred in the corrected recipe), augmentation |
| Loss | focal Tversky + BCE (β 0.75, γ 1.33, CE weight 0.5), optional per-region weights |
| Optimiser | AdamW, LR 1.6e-3 → 1e-5 per-step cosine, element-wise gradient clip ±1 |
| Weights | EMA 0.999; the best validation epoch is kept as `best.pth` |

The base configs reproduce the original Kaggle v7 script exactly (`legacy/kaggle/kaggle_v7.py`);
`tests/test_equivalence.py` runs both and compares losses, validation and test metrics.
The results above use `configs/experiments/brats2021_res96_long.yaml`.

## Layout

```
glo_nca/                  the package (pip install -e .)
  model.py                GLO-NCA cell: perception, SE, spatial gate, optional global block
  agent.py                cascade agent: coarse-to-fine inference, loss, optimiser step
  losses.py               focal Tversky + BCE loss
  metrics.py              IoU and HD95 (voxel and surface)
  config.py               YAML to settings
  data.py                 BraTS dataset (crop, resize, normalise, cache, patches, augmentation), split
  experiment.py           experiment state: config, split and train/val/test mode
  trainer.py              training loop, EMA, best-model selection, early stopping, final test
  evaluation.py           Dice / IoU / HD95, final decoder, threshold tuning, full-resolution scoring
  checkpoint.py           full checkpoints for exact resume
  reporting.py, plots.py  results JSON, per-case CSV, publication figures
  cli/                    command-line tools (also installed as glo-* commands)
train.py, evaluate.py     thin wrappers so the tools also run from a checkout without installing:
predict.py, ...           train, evaluate, predict, make_figures, compare_runs, benchmark, analyze
configs/                  base recipes; configs/experiments/ inherit and extend them
reproduce.sh              reproduce the reported test table from the released weights
MODEL_CARD.md             data, results, intended use and limitations of the released model
legacy/kaggle/            the original Kaggle scripts (v4-v7); they run against tag kaggle-v7-reference
tests/                    unit checks, pipeline test, v7 equivalence test, synthetic data generator
docker/, cloud/           training image; GCP scripts and systemd units
docs/                     figures, per-case results and analyses of the reported run; thesis text
.github/workflows/        CI: unit and pipeline tests on every push, v7 equivalence weekly
```

## Install

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # or the CPU index
pip install -e .                    # the package and the glo-train, glo-evaluate, glo-predict,
                                    # glo-figures, glo-compare, glo-benchmark and glo-analyze commands
pip install -r requirements.lock    # optional: the exact versions behind the reported results
```

## Reproduce the reported results

The weights, resolved configuration and patient split of the reported run are published as release
[v1.1.0](https://github.com/Muhammadwaqas1234/GLO-NCA/releases/tag/v1.1.0), with their SHA-256 hashes in `SHA256SUMS`:

| File | SHA-256 |
|---|---|
| `best.pth` | `94aff4cc8d19ac3b7cd04d22c81f4394d704f861a4c26107ab755974e4a1dc25` |
| `config.yaml` | `bd1c3ae1b6850ffcf6211eed8c262cee817c41b9bd1a4f606129fc02871b911e` |
| `split.json` | `330f554a569710951e21b95394a458a8e03607d517001be916cd63a689ee8027` |

```bash
./reproduce.sh /path/to/BraTS2021
```

downloads the three files and `SHA256SUMS` from the release (or uses them if present in
`release/glo-nca-brats2021-res96/`), refuses files whose hashes do not match, scores the 189 test
patients with the reported setting (ensemble x4 + flips, thresholds 0.65/0.65/0.60, clean-up
400/25/400, full resolution, BraTS empty-region rule, independent region masks, surface HD95 in the
medpy convention) and checks Dice within ±0.01 and HD95 within ±0.5 mm of the table above, with all
189 cases scored; the historical ensemble passes were not seeded. See `MODEL_CARD.md` for the model's
data, intended use and limitations.

## Train locally

```bash
pip install -e .                    # install torch first from the CUDA index
python train.py --config configs/experiments/brats2021_res96_long.yaml --data-root /path/to/BraTS2021
python train.py --resume experiments/<run-id> --data-root /path/to/BraTS2021
python train.py --config configs/glo_nca_cascade.yaml --data-root /path/to/BraTS --stop-after-epoch 10
```

Each run writes to `experiments/<run-id>/`: `config.yaml`, `split.json`, `train.log`,
`history.csv`, `status.json`, `last.pth` (every epoch, for resume) and `best.pth`
(EMA weights of the best epoch). The final test adds `results.json`, `test_per_case.csv`
and `figures/` (training curves, test summary, per-case Dice, example segmentations).

## Configuration options

All are off by default, so the base configs reproduce the v7 recipe; experiment configs
switch them on through `base:` inheritance, and `--override section.key=value` sets any value.

| Setting | Effect |
|---|---|
| `model.input_size` | working resolution, e.g. `[[48,48,48],[96,96,96]]` |
| `data.augment` | training-only flips, in-plane 90-degree rotations and intensity jitter |
| `data.cache` | `memory`, `disk` (low RAM, shared by workers) or `none` |
| `loss.tversky_beta`, `loss.region_weights` | recall/precision balance and loss weight per region |
| `training.val_every` | validate every N epochs (always on the last and the pause epoch) |
| `training.early_stop_patience`, `early_stop_min_delta` | stop when validation stops improving |
| `evaluation.tune_thresholds` | per-region threshold and clean-up size, chosen on validation |
| `evaluation.full_resolution` | score in the original scan space |
| `evaluation.brats_empty` | BraTS scoring of absent regions |
| `experiment.split_seed` | fixed split, so runs with other seeds can be ensembled |
| `model.use_se`, `model.use_spatial` | channel SE block and local spatial gate, each on or off (ablations) |
| `model.levels` | 2: coarse-to-fine cascade (default); 1: the fine level alone, trained on the same patches |
| `model.use_global` | attention-pooled global context (GCNet form), added to every voxel's hidden layer after normalisation |
| `model.update_prob` | probability that a cell updates per step; `fire_rate` (synonym `skip_prob`) is the skip probability (0.6 = 40% update) |
| `loss.all_regions` | score every region, also when its target is empty, so false positives are penalised |
| `training.validate_ema` | select the best epoch on the EMA weights that `best.pth` saves |
| `data.crop_et_prob` | share of training crops centred on enhancing tumour, at the crop that is trained on |
| `evaluation.nested_regions` | score the final nested masks (TC includes ET, WT includes TC), as written by `predict.py` |
| `evaluation.surface_hd95`, `evaluation.hd95_convention` | surface HD95 in mm from each scan's spacing; convention `medpy` (pooled percentile) or `max_directed` (larger directed percentile) |
| `model.norm` | `batch` (BatchNorm over the volume) or `layer` (per-voxel, strictly local update) |
| `loss.bce_logits` | BCE computed on logits instead of clamped probabilities |
| `data.brain_mask` | one brain mask (any modality non-zero) for resizing and normalisation |
| `data.et_labels` | labels counted as enhancing tumour (default 3 and 4); unknown labels stop loading |
| `training.amp` | `bf16`: autocast the training forward pass to bfloat16 (loss in float32) |
| `training.grad_checkpoint` | recompute each NCA step in the backward pass: less memory, identical gradients |
| `evaluation.case_seed` | seed the stochastic inference passes per case, so predictions repeat exactly |
| `data.legacy_patchify` | the old dataset-stage patch (working size, so it never moves); kept on only to reproduce earlier runs |
| `training.log_live_and_ema` | validate both the live and the EMA weights and record both |
| `training.persistent_workers` | keep data-loader workers alive across epochs (a resumed run is then not bit-identical to an uninterrupted one) |
| `data.modalities` | the input modalities; one modality gives the fusion ablation (`brats2021_res96_fixed_single.yaml`) |
| `data.split_by_patient` | keep all scans of one patient in the same split (identical split for BraTS 2021) |

`configs/experiments/brats2021_res96_fixed.yaml` switches the corrections above on and freezes one
evaluation protocol for every variant (ensemble ×4 + flips with case seeds, validation-tuned
thresholds, full resolution, BraTS empty rule, nested masks, surface HD95 in the medpy convention,
150 epochs without early stopping). The results above were produced before these corrections, with
all of them off: the loss skipped regions absent from a crop, validation scored the live weights
while `best.pth` holds the EMA weights, the training crop was uniformly random, and the spatial
block is a local 7×7×7 gate rather than a global operator.

## Ablation

Three runs of the corrected recipe, each trained once (seed 42, same patient split) with the frozen
evaluation protocol, so the comparisons are exploratory. Each ablation removes one mechanism: all
context beyond the local neighbourhood, or the multi-modal input. The run without global context also
has a third of the parameters, because the coarse level is a cell of its own; its result reflects the
missing context together with this capacity difference.

| Run | Config (`configs/experiments/`) | Removed | Parameters | Question |
|---|---|---|---|---|
| GLO-NCA | `brats2021_res96_fixed.yaml` | — | 30,138 | reference |
| Without global context | `brats2021_res96_fixed_noglobal.yaml` | SE, spatial gate and the coarse 48³ level (the fine level alone) | 10,272 | RQ2 |
| Without multi-modal fusion | `brats2021_res96_fixed_single.yaml` | FLAIR, T1 and T2 (T1ce only; the other three input slots stay empty) | 30,138 | RQ3 |

```bash
python train.py --config configs/experiments/brats2021_res96_fixed.yaml --data-root /path/to/BraTS2021
python train.py --config configs/experiments/brats2021_res96_fixed_noglobal.yaml --data-root /path/to/BraTS2021
python train.py --config configs/experiments/brats2021_res96_fixed_single.yaml --data-root /path/to/BraTS2021
python compare_runs.py --contrast context=experiments/<no-global>::experiments/<glo-nca> \
                       --contrast fusion=experiments/<t1ce-only>::experiments/<glo-nca> --output ablation.json
```

## Evaluate and ensemble

```bash
python evaluate.py --run experiments/<run-id> --data-root /path/to/BraTS2021 \
    --tune-thresholds --full-resolution --brats-empty
python evaluate.py --run experiments/<run-id> --data-root /path/to/BraTS2021 --splits test \
    --full-resolution --brats-empty --hd95-surface --thresholds WT=0.65,TC=0.65,ET=0.6
python evaluate.py --run experiments/<run-a> --run experiments/<run-b> --run experiments/<run-c> \
    --data-root /path/to/BraTS2021 --tune-thresholds --full-resolution
```

Scoring options, including threshold tuning and clean-up, default to the run's own `evaluation:`
section; flags override them and `--legacy` gives the original scoring (working grid, v7 empty rule,
independent masks, voxel HD95). The settings after all overrides form one specification that drives
the run and is written to the progress file and the output. Final scores report, per region, the
number of cases, the cases with a defined HD95 (the HD95 mean is conditional on them) and the missed
and false-positive cases; the per-case CSV has each case's HD95 and its status. Without tuning, cases are scored one at a time into a
progress file, so an interrupted evaluation resumes where it stopped; the file is tied to the
checkpoint, config, split, the data files (names, sizes, modification times), the metric version, the
code (including uncommitted changes) and the options, and a resume with any of them changed is refused. Ensemble members must share
`split.json` and every input setting; their predictions are aligned by case ID. `predict.py` keeps an
equivalent identity for its output folder.

`compare_runs.py` compares runs on the same test cases (difference B minus A, patient-level: the scans
of a patient are averaged first). It reports a 95% bootstrap interval over patients, a Wilcoxon test
and Holm correction for WT/TC/ET; several seeds per arm are given as comma-separated lists, and
`--contrast NAME=A::B` (repeatable) adds Holm correction across a family of contrasts. Inputs scored
under different rules, or without a recorded scoring specification, are refused unless
`--allow-mixed-specs` is given. Inference cost is measured with
`python benchmark.py --run experiments/<run-id> [--device cpu]`.

`analyze.py` adds the analyses of a trained run:

| Command | Purpose |
|---|---|
| `decompose` | the same seeded passes scored after each inference stage (single pass, ensemble, flips, thresholds, clean-up, ET rule) at full resolution, plus how well pass and flip spread flag poor cases |
| `steps` | accuracy and latency for other NCA step counts |
| `erf` | effective receptive field from input gradients, at a fixed tumour voxel per case, over several cases |
| `perturb` | effect of noise placed at a set distance from the predicted tumour boundary |
| `modalities` | leave-one-modality-out at inference |
| `manifest` | every case with its split, labels, region volumes and ET presence |
| `preflight` | data checks before training: files, shapes, co-registration, spacing, finite values, labels, duplicates |
| `overlap` | evaluated cases found in the run's training or validation split, from an official ID mapping table; fails when any case is unmapped |

Every start or resume of a run records its commit, uncommitted changes, Docker image, software
versions and GPU in `run_info.json`; a resume whose config changed is refused unless
`--allow-config-change` is given, and the changes are recorded.

## Train on GCP

Copy `cloud/config/gcp.env.example` to `cloud/config/gcp.env` and fill it in, then:

```bash
./cloud/scripts/start_vm.sh                      # from your machine
./cloud/scripts/setup_gcp.sh                     # on the VM: checkout and image build
./cloud/scripts/run_training.sh --config configs/experiments/brats2021_res96_long.yaml \
    --data-root /data/brats2021 --auto-stop --keep-alive
./cloud/scripts/status.sh                        # progress
./cloud/scripts/watchdog.sh                      # from your machine: restart after Spot preemptions
```

Runs are synced to `gs://<bucket>/experiments/<run-id>/` every 5 minutes. With
`--keep-alive` a restarted VM resumes the run from its last epoch; `watchdog.sh` restarts
the VM only after a preemption, and `--auto-stop` powers it off when the run ends.

## Verify

```bash
python tests/test_options.py                                   # unit checks, no data needed
python tests/test_pipeline.py                                  # train, predict, evaluate, compare, benchmark
python tests/make_synthetic_data.py /tmp/syn --cases 12         # synthetic BraTS-style cases
python tests/test_equivalence.py /tmp/syn                       # identical to the Kaggle v7 script
```

## Status

| Part | State |
|---|---|
| Reported model (BraTS 2021, internal test and BraTS 2023 validation) | done; weights published as release v1.1.0 |
| Code corrections from two external reviews and an independent audit | done; tested (`tests/`) and identical to the original recipe by default |
| The three runs (ablation table above) | configured; not yet trained |
| Analyses on BraTS 2021 (decomposition, steps, modalities, receptive field, GPU cost) | tools ready; to run with the ablation |
| Overlap of the BraTS 2023 validation cases with the training split | tool ready; needs the official BraTS 2023 glioma ID mapping |


