# DATA PIPELINE AUDIT

Traced disk → model for one real case. Source-verified; no code executed against
real data (no GPU/torch in this environment). Every claim cites `file:line`.

## 1. Execution path

```
runner.run
 └─ datasource.resolve_data_root            -> dataset root
 └─ validate_dataset                        -> refuses to train on invalid data (good)
 └─ datasource.load_master_split            -> split/master_split.json (fail-hard if missing)
 └─ exp.data_split.images/labels[split]     runner.py:345-348
 └─ exp.set_model_state("train")            -> ds.setPaths + ds.setState
 └─ DataLoader(ds, ...)                     runner.py:434-437   [RECREATED EVERY EPOCH]
     └─ Dataset_NiiGz_3D_BraTS.__getitem__  Nii_Gz_Dataset_3D.py:135
         ├─ data.get_data(key)              cache lookup (Data_Instance.py:11)
         ├─ nib.load(...).get_fdata() × 5   :154-157  (4 modalities + seg)
         ├─ _foreground_bbox                :161      (np.any over 4-ch volume)
         ├─ rescale3d × 5                   :169-170  (2 × cv2 slice loops each)
         ├─ _labels_to_regions              :173
         ├─ data.set_data(key, ...)         :176      [WRITES INTO WORKER COPY]
         ├─ patchify_multimodal             :184
         ├─ _augment                        :192
         └─ nonzero z-norm                  :196-205
 └─ agent.prepare_data -> .to(device)       Agent_GLO_NCA_V3.py:60-61  (blocking transfer)
 └─ model forward
```

## 2. Verified-correct behaviour (PASS)

| Aspect | Verdict | Evidence |
|---|---|---|
| Modality ordering | PASS | `MODALITIES` fixed T1/T1ce/T2/FLAIR, config sets `[t1n,t1c,t2w,t2f]`; `raw` stacked in list order (`:154-156`) |
| Label mapping WT/TC/ET | PASS | `_labels_to_regions:125-132`. ET={3,4}, TC=NCR∪ET, WT=NCR∪ED∪ET. Correct nested BraTS regions, handles the Kaggle 4→3 remap |
| Multi-label (not softmax) | PASS | 3 independent sigmoid channels; loss is per-channel |
| Interpolation modes | PASS | images `INTER_LINEAR`, labels `INTER_NEAREST` (`rescale3d:230`); masks stay binary |
| Foreground crop | PASS | bbox across all modalities, applied identically to img and seg (`:160-165`) |
| Augmentation label-safety | PASS | flips/rot90 use the same axes for img and label; elastic shares one displacement field with order=0 for labels (`_elastic:318-332`); intensity ops touch image only and preserve zeros |
| Split leakage | PASS | subject-disjoint master split, loaded fail-hard, never regenerated when `split_file` is set (`runner.py:303-311`) |
| Shape/dtype consistency | PASS | `img_norm` explicitly `np.float32` (`:197`); label float32 (`:132`) |

## 3. FINDING D-01 (P1, CONFIRMED) — the preprocessing cache never works

**Current behaviour.** `Data_Container` (`Data_Instance.py`) is a plain in-memory
dict on the dataset object. `__getitem__` writes each fully preprocessed case
into it (`Nii_Gz_Dataset_3D.py:176`).

**Why it fails.** `runner.py:434-437` constructs a **new `DataLoader` inside the
epoch loop** with `num_workers=4` and **no `persistent_workers`**:

```python
for ep in range(start_epoch, epochs):
    loader = torch.utils.data.DataLoader(
        ds, shuffle=True, batch_size=batch_size, num_workers=workers,
        pin_memory=(device.type == "cuda"), worker_init_fn=_worker_init)
```

With `num_workers > 0` each worker gets its own copy of `ds`. `set_data` mutates
the **worker's** copy; nothing is sent back to the parent. Workers are destroyed
when the loader is exhausted at epoch end, and a brand-new loader spawns brand-new
workers next epoch. Consequence:

- The parent process cache is **permanently empty**.
- Each worker's cache survives only within one epoch, and with 4 workers each
  sees ~1/4 of the cases, so even the intra-epoch hit rate is ~0 at batch_size 1
  with shuffle (each case is visited once per epoch).
- **Every case is re-decompressed and re-resampled on every epoch, for all 300
  epochs.** The `.nii.gz` gzip decode of 5 volumes plus 10 `cv2` slice loops is
  repeated 898 × 300 times.

Confirmed absence of any mitigation: `persistent_workers`, `prefetch_factor` and
`non_blocking` appear **nowhere** in `src/` (grep). The docstring "so that this
only needs to be done once" (`Data_Instance.py:4-5`) is not true in production.

**Measured/source-verified cost.** Not measured here (no torch). Prior evidence in
`reports/validation/GLO_NCA_V3_PERFORMANCE_AUDIT.md` observed **100% GPU
utilisation** during a GCP epoch, which means the GPU is currently *not* starved —
so this is wasted CPU/IO work running concurrently with compute, not necessarily
wall-clock on the critical path today. It becomes the bottleneck the moment GPU
compute is reduced. Classification: **CONFIRMED defect, LIKELY-significant cost.**

**Methodology impact of fixing:** none on the model. But note any fix that changes
worker count or caching changes **which worker draws which RNG values**, so
bit-exact continuity with prior runs is not preserved. Flagged for Phase 2.

## 4. FINDING D-02 (P1, CONFIRMED) — patchify is a no-op that still does work

**Current behaviour.** For the production config, `patch_size: 128` and level 3
resolution is 128.

Trace of `self.size`: `runner._build_v3:134` sets
`"input_size": [[patch, patch, patch]]` = `[[128,128,128]]`.
`Experiment.set_size:124-126` sees `first` is a list, so calls
`dataset.set_size(input_size[-1])` → `self.size = (128,128,128)`.

That single `self.size` is then used by **both**:
- `rescale3d:228-240` — resizes every volume to exactly 128×128×128, and
- `patchify_multimodal:342` — `size = self.size` = `(128,128,128)`.

So in `patchify_multimodal` the image is already 128³ and the requested patch is
128³. Therefore:

```python
pos_x = random.randint(0, img.shape[0] - size[0])   # randint(0, 0) == 0
```

All three positions are **forced to 0**. The extracted "patch" is the whole
volume. **Patch extraction is a guaranteed no-op for the production config.**

**Why it still costs.** The no-op is not free:

1. `random.uniform(0,1) < 0.7` → ~70% of samples set `contains_mask=True`.
2. When true, the loop cannot `break` early unless the ET channel is non-empty at
   position (0,0,0) — i.e. unless the case has any ET at all.
3. For a case with **no ET voxels** (common — ET is the rarest region and absent
   in a meaningful fraction of cases), `patch.max()` is 0 every iteration, the
   fallback branch runs a **second** full-volume `label[...,0].max()` on the first
   iteration, and the loop runs the **full 50 iterations**.
4. Each iteration slices `label[0:128,0:128,0:128,2]` — a 2,097,152-element view —
   and calls `.max()` on it. That is **50 × 2.1M = ~105M element reductions per
   such sample**, all to re-derive the same answer, always ending at (0,0,0).

This is pure waste: the answer is structurally predetermined. Classification:
**CONFIRMED.**

**Possible safe optimization (Phase 2, do not apply yet).** Short-circuit when
`img.shape[:3] == size`. But `scripts/test_patchify_equivalence.py` (already in
the repo, untracked) correctly identifies the catch: the short-circuit consumes
**zero** `random.*` draws whereas the current path consumes 1 + 3×(1..50). Python
`random` is **shared with `_augment`** (`:266,270,278,289,290,294,298,305`), so
removing draws **shifts the entire augmentation RNG stream**. Output arrays would
be identical for a given call, but the *run* would not be bit-identical to a prior
run. Methodology impact: none (augmentation is random either way, and the
distribution is unchanged), but reproducibility-vs-previous-runs is affected. Must
be a deliberate, documented decision.

## 5. FINDING D-03 (P2, CONFIRMED) — `rescale3d` is a Python-level double slice loop

`rescale3d:232-240` resizes in two passes, each a Python `for` loop calling
`cv2.resize` per slice: first `img.shape[2]` iterations (~155 for a cropped BraTS
volume), then `self.size[1]` = 128 iterations. That is ~283 `cv2.resize` calls per
volume, × 5 volumes (4 modalities + seg) = **~1,415 Python-level cv2 calls per
case**, repeated every epoch because of D-01.

Correctness is fine (linear for images, nearest for labels, which is the right
choice). This is a pure performance observation. Classification: **CONFIRMED**
inefficiency, **POSSIBLE** significance (unmeasured, and currently overlapped with
GPU compute).

## 6. FINDING D-04 (P2, CONFIRMED) — redundant unused work in `__getitem__`

`Nii_Gz_Dataset_3D.py:142-143` constructs two `torchio` transform objects
(`RescaleIntensity`, `ZNormalization`) on **every single call**, before the cache
lookup. In the production config `nonzero_norm: true` (`runner.py:136`), so the
branch that uses them (`:207-214`) is **never taken** — they are constructed and
discarded 898×300 times. Harmless but dead.

## 7. FINDING D-05 (P2, CONFIRMED) — blocking host→device transfer

`Agent_GLO_NCA_V3.py:60-61` uses `.to(self.device)` without `non_blocking=True`,
while the loader does set `pin_memory=True` (`runner.py:437`). Pinned memory's
benefit — overlapping the copy with compute — is therefore not realised. Minor and
safe to change (no numerical effect), but unmeasured. Classification: **CONFIRMED**
code fact, **POSSIBLE** cost.

## 8. Data-leakage assessment

**PASS.** No leakage found.
- Split is subject-disjoint and loaded from a fingerprinted master file.
- Normalisation statistics are computed **per-case, per-channel** (`:198-203`),
  never across the dataset — so no train→test statistic bleed.
- `patchify` and `_augment` are gated on `self.state == "train"` (`:183`, `:191`),
  so validation and test see deterministic, un-augmented full volumes.
