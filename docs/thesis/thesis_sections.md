# GLO-NCA thesis sections: revised proposal, data, method, ethics and limitations

Text for the thesis chapters, matching the final implementation (repository `main`) and the reported run
`BRATS2021-RES96-LONG-20261001-071710`. Numbers in square brackets are to be filled in from the
corresponding analysis output.

## 1. Research question and objectives

**Research question.** How accurately can a compact two-level neural cellular automata (NCA) cascade
segment adult glioma sub-regions on multi-parametric MRI, and what do its channel recalibration (SE)
and spatial gating blocks contribute compared with an otherwise matched multiscale NCA?

**Objectives, each with an evaluable criterion.**

1. Build a two-level NCA cascade with four-modality input and SE and spatial-gating blocks in every cell.
   *Criterion:* trains stably on BraTS 2021 and is reproducible from the released configuration and
   weights (`reproduce.sh`).
2. Measure segmentation accuracy. *Primary endpoint:* patient-level mean of WT, TC and ET Dice on the
   held-out internal test set (189 patients). *Secondary endpoints:* surface HD95 (mm), lesion-wise
   Dice, false-positive and missed ET, and the official BraTS 2023 validation leaderboard scores.
3. Isolate the contribution of the context blocks. *Criterion:* paired per-patient difference in mean
   Dice between the model and a plain multiscale NCA (no SE, no spatial gate), trained with the same
   recipe, with a patient-level bootstrap 95% confidence interval and a Wilcoxon signed-rank test.
4. Measure computational cost. *Criterion:* parameters, checkpoint size, inference latency (median
   and 95th percentile) and peak memory on stated hardware, for the reported inference setting and for
   a single pass.

## 2. Corrections to the proposal

The final study differs from the proposal in the following points; the proposal's preliminary
experiment is a separate, earlier version and its scores are not compared with the final ones.

| Item | Proposal | Final study |
|---|---|---|
| Dataset | BraTS 2024, subset of 200 cases | BraTS 2021 adult glioma training set, all 1,251 cases |
| Split | 140 / 30 / 30 | 875 / 187 / 189 (70 / 15 / 15 %), random by patient, seed 42 |
| External evaluation | not planned | BraTS 2023 adult glioma validation set, 219 cases, official Synapse scoring |
| Working resolution | 64 × 64 × 48 | 96 × 96 × 96 (coarse level 48³) |
| Parameters | about 24,000 | 30,138 |
| Loss | Dice + cross-entropy | focal Tversky + binary cross-entropy per region |
| Spatial block | attention-pooled global vector | local 7 × 7 × 7 spatial gate (CBAM-style); an attention-pooled global block is implemented as an optional variant |
| Fusion | region-specific modality weighting | the four modalities are input channels; SE recalibrates the 24 perceived state channels |
| Evaluation | Dice and HD95 | Dice, surface HD95 in mm at full resolution, BraTS empty-region convention, lesion-wise scores on the leaderboard |

## 3. Data

**Training and internal test data.** The BraTS 2021 adult glioma training set contains 1,251
pre-operative multi-parametric MRI scans (T1, contrast-enhanced T1, T2, FLAIR), co-registered,
skull-stripped and resampled to 1 mm isotropic voxels (240 × 240 × 155), each with an expert
segmentation of necrotic tumour core (label 1), peritumoural oedema (label 2) and enhancing tumour
(label 4). The regions evaluated are whole tumour (WT: labels 1, 2, 4), tumour core (TC: 1, 4) and
enhancing tumour (ET: 4). Each scan belongs to a different patient. The cases were split at random by
patient into 875 training, 187 validation and 189 test cases (seed 42, saved as `split.json`). The
validation cases were used for model selection and post-processing choices; the test cases were used
once, for the final internal results.

**External evaluation.** The labels of the official BraTS 2021 validation cases are not released, and
the 2021 validation phase is closed; the same 219 cases are scored in the BraTS 2023 adult glioma
continuous evaluation on Synapse, where predictions are uploaded and the organisers compute the
metrics. This set was used for external evaluation (labels 1, 2 and 3 in the 2023 release, with 3 for
enhancing tumour). Its cases are disjoint from the BraTS 2021 training cases by construction of the
challenge; the official ID mapping table can confirm this (`glo-analyze overlap`).

**Preprocessing.** Each scan is cropped to the bounding box of non-zero voxels across modalities,
resized slice-wise to 96³ (cubic interpolation for images, nearest neighbour for labels) and
z-normalised per modality over its positive voxels. Predictions are resampled back into the cropped
region of the original scan (trilinear) before thresholding and scoring.

## 4. Method

**Cell state.** Each voxel *i* holds a state vector **h**ᵢ ∈ ℝ²⁴: channels 1–4 are the input
modalities (restored after every step), channels 5–7 are the logits of WT, TC and ET, and channels
8–24 are hidden.

**Update.** One step of a cell applies

  **p**ᵢ = G( S( (K ∗ **h**)ᵢ ) ),  **u**ᵢ = W₁ ReLU( BN( W₀ [**h**ᵢ, **p**ᵢ] ) ),
  **h**ᵢ ← **h**ᵢ + mᵢ **u**ᵢ,  mᵢ ~ Bernoulli(0.4),

where K is a learned depthwise convolution (kernel 7 on the coarse level, 3 on the fine level, reflect
padding); S is squeeze-and-excitation over the 24 channels (global average pooling, a 24 → 6 → 24
bottleneck and a sigmoid gate); G is a spatial gate A = σ(Conv₇ₓ₇ₓ₇[mean_c, max_c]) applied as
**p** ⊙ A; W₀ ∈ ℝ¹²⁸ˣ⁴⁸ and W₁ ∈ ℝ²⁴ˣ¹²⁸ (no bias, initialised to zero) form a per-voxel MLP with batch
normalisation over the volume and dropout 0.1; and mᵢ is a stochastic update mask (each voxel updates
with probability 0.4; in the code `fire_rate = 0.6` is the probability of skipping). After each step
the four input channels are reset to the image.

**Cascade.** The coarse cell runs 20 steps on the image max-pooled to 48³. Its state is upsampled to
96³ (nearest neighbour), its hidden and output channels are combined with the full-resolution image,
and the fine cell runs 20 more steps. During training the fine level runs on a random 48³ crop of the
96³ volume; at inference both levels run on the whole volume. The model has 30,138 parameters.

**Loss.** For each region r with sigmoid probability *p* and target *y*,

  TI = (TP + 1) / (TP + α·FP + β·FN + 1),  L_r = (1 − TI)^γ + 0.5 · BCE(p, y),

with α = 0.25, β = 0.75 and γ = 1.33, and L = Σ_r L_r. In the reported run a region's term was
included only when the region was present in the training crop; the corrected configuration scores
every region (Section 7).

**Training.** AdamW (β = 0.9, 0.99, weight decay 10⁻⁴), learning rate 1.6 × 10⁻³ with per-step cosine
decay to 10⁻⁵ over 150 epochs, batch size 1, element-wise gradient clipping at ±1, exponential moving
average of the weights (decay 0.999), and flips, in-plane 90° rotations and intensity jitter as
augmentation. The best epoch by validation mean Dice (epoch 137) was kept; its EMA weights are the
released model.

**Inference and post-processing.** Four stochastic passes and four flip orientations (none and one
per axis) are averaged. The regions are thresholded at 0.65 (WT), 0.65 (TC) and 0.60 (ET), connected
components smaller than 400, 25 and 400 voxels are removed, and the regions are nested (TC includes
ET, WT includes TC) to write the label map. For the leaderboard, an ET region smaller than 5,000 voxels
and smaller than 25% of the tumour core is relabelled as necrotic core.

## 5. Ethics and data use

This study used only publicly released, de-identified MRI data from the BraTS 2021 and BraTS 2023
challenges, obtained through the official channels under the challenges' data use terms, which permit
research use and require citation of the data descriptor publications. No new data were collected, no
patients were contacted and no attempt was made to re-identify individuals. The data were stored and
processed on the author's computer and a private cloud project, and were not redistributed.
[Institutional ethics determination: state here whether the university's ethics committee required a
review for secondary use of public de-identified data, and its reference if one was issued.] The model
is a research prototype, not a medical device; it has not been validated for clinical use and must
not be used for diagnosis or treatment decisions.

## 6. Limitations

- **Single split and single training run.** Results come from one patient-level split and one training
  seed; variability across splits and seeds was not estimated, so differences of about 0.01 Dice
  should be interpreted with care.
- **Post-processing chosen on the external set.** The ET relabelling rule was chosen after inspecting
  BraTS 2023 validation results. Leaderboard scores with the rule are therefore development results,
  not an independent test; the scores without the rule (mean Dice 0.849) are the pre-specified ones.
- **Local spatial block.** The spatial gate is a local 7³ operation; only SE aggregates over the whole
  volume (over channels), and batch normalisation uses volume statistics. The model is therefore not a
  global-attention model, and the contribution of the blocks is established only by the ablation.
- **Training defects in the reported run.** The reported run used a loss that skipped regions absent
  from a training crop, a training crop that was not biased towards enhancing tumour, and selected the
  best epoch on the live weights while saving the EMA weights. These are corrected in the code; the
  reported numbers are from the original run.
- **Run history.** The run was resumed several times with changed validation frequency (every 5
  epochs until epoch 38, then every epoch) and early-stopping settings, recorded in `train.log`.
- **Resolution.** Resizing each brain to 96³ discards detail; most remaining errors are missed or
  misplaced small enhancing tumour.
- **Scope.** Pre-operative adult glioma only; post-treatment scans, other tumour types and other
  acquisition protocols were not evaluated, and the internal and external sets both come from BraTS.
