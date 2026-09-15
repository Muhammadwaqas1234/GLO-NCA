# Master case split

`master_split.json` is the ONE canonical train/validation/test split shared by
every experiment (V2 A0/A1/A2/A3/Final and the V3 runs that require comparison).
It holds **case IDs only** — never image data — so it is safe and useful to
commit for reproducibility.

## Dataset scope (BraTS-MET 2025)
The training dataset is enumerated by **recursive, files-validated** case
discovery (`src/experiment/datasource.py: discover_cases`): a directory is a
case only if it directly contains all four modalities + a segmentation
(`{case}-t1n|t1c|t2w|t2f|seg.nii.gz`). Container directories are never treated
as cases.

- **Top-level cohort:** 650 `BraTS-MET-XXXXX-YYY` case directories.
- **Nested UCSD cohort:** 646 case directories inside `UCSD - Training/`
  (that folder is a container, not a case).
- **Total valid cases:** **1296** (discovered from the filesystem, not hardcoded;
  validation re-reports the real count if the dataset changes).

## Timepoint handling (each timepoint is an independent sample)
Case IDs carry a longitudinal timepoint suffix, e.g. `BraTS-MET-00559-000`,
`-001`, … `-004`. Each timepoint directory has its own multimodal MRI + seg and
is treated as an **independent segmentation sample** (`case_id`). Timepoints are
**not** collapsed into one sample — V3 is a single-volume segmentation model, not
a longitudinal model. So `BraTS-MET-00559-000` and `BraTS-MET-00559-001` are
distinct cases.

## Subject grouping & leakage rule (thesis-critical)
The **base subject** id is the case id minus the trailing `-<timepoint>`
(`src/experiment/datasource.py: subject_of`). The 1296 cases belong to **810
distinct subjects**; 287 subjects have >1 timepoint. To prevent temporal
leakage, the split is **SUBJECT-disjoint**: every timepoint of a subject is
assigned to the *same* partition. `build_master_split` enforces this as a hard
gate and refuses to write a leaky split; `check_split.py` re-verifies it.

The two cohorts have **no base-subject overlap and no case-id collisions**, so
merging them is unambiguous.

## Proportions, seed, files
- **Seed:** 42 (project canonical).
- **Proportions:** 70 / 15 / 15, applied **over subjects** (whole subjects are
  assigned), which for the current data yields ≈ 567 / 121 / 122 subjects →
  898 / 200 / 198 cases.
- **File:** `split/master_split.json`, created ONCE from the real dataset:

```bash
python scripts/create_master_split.py --data-root "$DATA_DIR"
python scripts/check_split.py --split split/master_split.json --data-root "$DATA_DIR"
```

It stores: `train` / `validation` / `test` (case IDs), `seed`, `grouping`,
`dataset_case_count`, `subject_count` (+ per-partition subject counts),
`split_sha256`, `patient_id_hash`, `split_version`, `created_at_utc`.

## Identity definitions
- **`split_sha256`** — sha256 over the canonical `{train,val,test}` case-ID sets
  (`datasource.split_fingerprint`); depends only on which case is in which
  partition, not on list order.
- **`patient_id_hash`** (a.k.a. dataset identity) — sha256 over the sorted set of
  all discovered **case IDs** (`datasource.patient_id_hash`).

All configs reference `split/master_split.json` via `data.split_file`. If the
file is missing, every training run **fails loudly** rather than silently
generating a fresh split — this prevents experimental drift. `.gitignore` allows
`master_split.json` (case-IDs only) to be committed while ignoring image data.
Commit the real split once generated so the exact thesis split is archived.
