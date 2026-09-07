r"""GLO-NCA experiment tooling (Phase 1).

A thin, professional layer around the existing GLO-NCA V2 training code that
adds reproducibility and experiment management WITHOUT changing the research
methodology: YAML config, per-run experiment directories, dataset validation,
full checkpoints + resume, TensorBoard + CSV + graphs, logging, environment /
git / GPU capture, status + manifest.
"""
