# Archive — historical / development artifacts

This directory retains development history for provenance and reproducibility.
**None of these files are part of the final GLO-NCA V3 pipeline** and none are
imported by `train.py`, `src/`, or the cloud scripts.

## `kaggle_experiment_history/`
`kaggle_v4.py … kaggle_v7.py` — the exploratory Kaggle notebooks/scripts that led
to the current recipe. `v4` was the strongest early config; `v7` was the final
Kaggle recipe that the production `train.py` + `src/experiment/` harness later
superseded. Kept only as a record of how the training recipe was arrived at.

They are **not maintained**, may reference older conventions, and must not be
used for the thesis experiments — use `configs/v3_multilevel_ckpt.yaml` with the
production runner instead.
