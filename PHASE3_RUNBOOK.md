# Phase 3 Runbook — GLO-NCA V2 experimental execution

This is the exact, ordered procedure to produce the thesis-ready evidence. **You
run these on GCP** (a coding assistant cannot execute GPU training or use your
GCP credentials — any "results" produced without a real run would be fabricated
and are explicitly forbidden). The harness turns your real run outputs into
thesis tables and figures with **no hard-coded numbers**.

Everything below preserves the frozen-test discipline: thresholds and model
selection use validation only; the test set is evaluated once, frozen.

---

## 0. Preconditions
- Phase 2 cloud setup done (`cloud/config/gcp.env` filled, bucket + VM created).
- Dataset uploaded and validated (`cloud/scripts/upload_dataset.sh`).
- Branch `v2`, clean working tree. Record the commit you run with.

## 1. Cloud smoke test (Phase 3A) — MUST pass before anything expensive
```bash
./cloud/scripts/start_vm.sh
gcloud compute ssh "$VM_NAME" --zone "$GCP_ZONE"
cd /opt/glo-nca && ./cloud/scripts/setup_gcp.sh && ./cloud/scripts/verify_gcp.sh
./cloud/scripts/run_training.sh configs/smoke_test.yaml
./cloud/scripts/monitor.sh
# from local PC, verify round-trip:
./cloud/scripts/download_experiment.sh <smoke_experiment_id>
python cloud/scripts/validate_checkpoint.py experiments/<smoke_experiment_id>
```
If any step fails, STOP and fix infrastructure (do not use the final run as a test).

## 2. Dataset + split validation (Phase 3B)
```bash
# on the VM (dataset cached locally by run_training / cache_dataset)
python scripts/validate_dataset.py --root "$VM_DATA_DIR"          # PASS required
python scripts/dataset_identity.py --root "$VM_DATA_DIR" --out /out/dataset_identity.json
```
The split is created + persisted by the first training run and reused by all
others; verify integrity after the first ablation (step 3) with:
```bash
python scripts/check_split.py --experiment /out/<first_exp_id> --data-root "$VM_DATA_DIR"
```

## 3. Ablations (Phase 3C) — four runs, identical control vars, only switches differ
Run each with the same seed/epochs/patch/aug. Each writes its own experiment dir
and syncs to GCS automatically.
```bash
./cloud/scripts/run_training.sh configs/ablation_baseline.yaml   # A0: attn F, spatial F
./cloud/scripts/run_training.sh configs/ablation_se.yaml         # A1: attn T, spatial F
./cloud/scripts/run_training.sh configs/ablation_spatial.yaml    # A2: attn F, spatial T
./cloud/scripts/run_training.sh configs/ablation_full.yaml       # A3: attn T, spatial T
```
> Budget note: all four use the SAME 200-epoch budget so the comparison is fair
> (§17). If you first want a cheap trend check, copy a config and set
> `training.epochs` lower — but the REPORTED ablation must use equal budgets.

## 4. Final GLO-NCA run (Phase 3D)
`configs/gcp_full.yaml` is the final config (epochs 200, patch 96, aug light,
SE+spatial on, seed 42). A3(full) and the final share the architecture, but run
the final as its own clearly-named experiment:
```bash
./cloud/scripts/run_training.sh configs/gcp_full.yaml            # name: GLO-NCA-V2-final
```
Best-checkpoint selection (validation smoothed), validation threshold tuning,
frozen test @0.5 and @tuned, per-case metrics, statistical summary, diagnostic
report — all produced automatically by the run.

## 5. Aggregate into thesis tables + figures (Phase 3E/3F) — from REAL outputs
```bash
# download the finished experiments locally (or run these on the VM)
python scripts/make_ablation_table.py \
    --baseline experiments/GLO-NCA-V2-ablation-baseline-<ts> \
    --se       experiments/GLO-NCA-V2-ablation-se-<ts> \
    --spatial  experiments/GLO-NCA-V2-ablation-spatial-<ts> \
    --full     experiments/GLO-NCA-V2-ablation-full-<ts> \
    --out      reports/ablation_results.csv

python scripts/make_thesis_tables.py \
    --final experiments/GLO-NCA-V2-final-<ts> \
    --ablation-csv reports/ablation_results.csv

python scripts/make_figures.py \
    --ablation-csv reports/ablation_results.csv \
    --final experiments/GLO-NCA-V2-final-<ts> \
    --out experiments/GLO-NCA-V2-final-<ts>/reports/thesis/figures
```

## 6. Archive + verify + stop (Phase 3 §50-52, §71)
```bash
./cloud/scripts/sync_experiment.sh /out/GLO-NCA-V2-final-<ts>     # ensure in GCS
./cloud/scripts/status.sh                                        # list GCS experiments
./cloud/scripts/download_experiment.sh GLO-NCA-V2-final-<ts>     # local archive
./cloud/scripts/stop_vm.sh                                       # stop billing
```

## What each final experiment dir will contain (thesis archive)
```
reports/results.json, diagnostic_report.{json,txt}, thesis_results.csv,
        per_case_test.csv, statistical_summary.json, threshold_comparison.csv,
        thesis/{dataset_table,model_complexity,ablation_table,final_metrics,
                threshold_table,statistical_summary,training_summary}.csv,
        thesis/experiment_summary.json, thesis/final_summary.md,
        thesis/figures/*.png
checkpoints/{best,last}.pth + periodic/, metrics/*.csv, graphs/*.png,
tensorboard/, logs/training.log, config/*, split/*, cloud_metadata.json,
experiment_manifest.json, status.json
```

## Non-negotiables (enforced by the code + these steps)
- Test set evaluated ONCE, thresholds FROZEN from validation.
- No methodology change mid-experiment; a change ⇒ new experiment ID.
- HD95 stays in **voxels** (labelled `HD95(vox)`), never silently mm.
- No fabricated baselines; external numbers (if any) go in a clearly-labelled
  separate published-comparison table with sources.
- Ablations use equal training budgets; report absolute improvements (no
  significance claims unless a documented test is added).
