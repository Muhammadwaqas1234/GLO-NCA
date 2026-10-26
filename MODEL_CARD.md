# Model card: GLO-NCA (BraTS 2021, 96³)

## Model

- Two-level neural cellular automata cascade (48³ → 96³), 30,138 parameters, checkpoint 0.13 MB.
  These published weights are the first version, without the attention-pooled global-context block of
  the final GLO-NCA (`brats2021_res96_glonca.yaml`, 32,304 parameters).
- Each cell: learned depthwise perception, squeeze-and-excitation over channels, a local 7×7×7
  spatial gate, BatchNorm, and a stochastic update (update probability 0.4).
- Input: T1, T1ce, T2 and FLAIR, foreground-cropped, resized to 96³ and z-normalised per scan.
- Output: three overlapping regions, whole tumour (WT), tumour core (TC) and enhancing tumour (ET).
- Configuration: `configs/experiments/brats2021_res96_long.yaml`; best validation epoch 137.

## Training data

BraTS 2021 adult glioma training set (1,251 pre-operative multi-parametric MRI scans), seeded random
split by patient: 875 training, 187 validation, 189 test (`split.json`). Use of the data and of models
trained on it is subject to the BraTS data terms.

## Results

| Evaluation | WT | TC | ET | Mean Dice |
|---|---|---|---|---|
| Internal test, 189 patients (Dice) | 0.921 | 0.895 | 0.817 | 0.878 |
| Internal test, surface HD95 (mm), medpy convention* | 4.93 | 5.23 | 4.49 | |
| BraTS 2023 validation, 219 cases, Synapse (Dice) | 0.909 | 0.855 | 0.784 | 0.849 |
| Same, with ET relabelling rule (Dice) | 0.909 | 0.855 | 0.797 | 0.854 |

*One 95th percentile over the surface distances of both directions, as `medpy.metric.binary.hd95`;
ET is undefined (one mask empty) in 5 of 189 cases. The Synapse values are computed by the organisers.

Internal test: ensemble ×4 with flips, thresholds 0.65/0.65/0.60 and clean-up 400/25/400 voxels chosen
on validation, full resolution, BraTS scoring (`./reproduce.sh`). The ET relabelling rule was chosen
after seeing validation scores, so that row is development evidence, not an independent test.

## Intended use

Research on compact segmentation models. Not a medical device, not validated for clinical use, and not
to be used for diagnosis or treatment decisions.

## Limitations

- Trained on pre-operative adult glioma only; post-treatment scans, other tumour types, other sequences
  or non-skull-stripped images are out of scope.
- Fixed cubic resizing to 96³ loses small lesions; most remaining errors are missed small ET.
- The spatial block is local; only the SE block aggregates globally (over channels), and BatchNorm
  uses volume statistics.
- Results come from one training seed; the run was resumed with changed validation and
  early-stopping settings (see `train.log` and the README).
- Training used the original loss, which skips regions absent from a crop, and selected the best
  epoch on the live weights while saving the EMA weights; both are corrected in
  `configs/experiments/brats2021_res96_fixed.yaml`.

## Files

Release [v1.1.0](https://github.com/Muhammadwaqas1234/GLO-NCA/releases/tag/v1.1.0): `best.pth` (EMA weights of the best epoch, 0.12 MB), `config.yaml`
(resolved configuration), `split.json` (patient split 875/187/189) and `SHA256SUMS`.

| File | SHA-256 |
|---|---|
| `best.pth` | `94aff4cc8d19ac3b7cd04d22c81f4394d704f861a4c26107ab755974e4a1dc25` |
| `config.yaml` | `bd1c3ae1b6850ffcf6211eed8c262cee817c41b9bd1a4f606129fc02871b911e` |
| `split.json` | `330f554a569710951e21b95394a458a8e03607d517001be916cd63a689ee8027` |

Reproduce with `./reproduce.sh /path/to/BraTS2021`, which downloads the files, verifies the hashes and
checks every reported metric. The data itself is not redistributed; BraTS 2021 is obtained from the
challenge organisers under their data terms.
