# Reproducibility

- **`PHASE3_RUNBOOK.md`** — operational runbook for the experiment harness (split
  creation/verification, GCP pre-flight, launch order). Development-era document;
  where it names configs, the current names are in `configs/` (see
  `docs/thesis/FINAL_TRAINING_PROTOCOL.md` for the authoritative final config).
- **Master split:** `split/master_split.json` (frozen; SHA256
  `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`). Verify with
  `python scripts/check_split.py --split split/master_split.json`.
- **Software validation (no dataset):** `python scripts/validate_v3_local.py --device cpu`.
- Every experiment run records git commit, config/dataset/split identity, seed,
  environment and full checkpoint state in its manifest.
