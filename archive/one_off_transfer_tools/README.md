# One-off transfer / diagnostic tools (ARCHIVED — not production)

Moved here in Phase 2 after dependency analysis proved **zero references** from
any production script, config, Dockerfile, systemd unit or documentation.
Nothing was deleted: these are kept as engineering history and may still be
useful for a manual re-transfer.

| File | Was used for | References found | Why archived |
|---|---|---|---|
| `xfer2.sh` | one-off Google Drive → VM disk → GCS dataset transfer | **0** | dataset is already established in GCS; also lacked `set -e`, so a partial extract could reach the production GCS prefix |
| `drive_dl.py` | public-Drive downloader used only by `xfer2.sh:11` | **1** (only `xfer2.sh`) | its sole caller is archived; hard-codes a single Drive file id |
| `dataset_identity.py` | wrote a dataset identity manifest | **0** | superseded by the automatic `dataset_identity.json` the runner writes each run (`runner.py`); also used a flat `os.listdir` that could not see the nested `UCSD - Training/` cohort |

**Do not add these back to the production path.** The live equivalents are:

- dataset upload/verification → `cloud/scripts/upload_dataset.sh` (validates
  fail-closed *before* upload)
- dataset caching on the VM → `cloud/scripts/cache_dataset.sh`
- dataset identity/provenance → written automatically into every experiment
  directory as `config/dataset_identity.json`
