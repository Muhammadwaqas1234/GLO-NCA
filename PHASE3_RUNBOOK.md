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

## HARDENED SEQUENCE (Phase 3.1) — run in this exact order

### Step 0 — clean repo
```bash
git status && git rev-parse HEAD          # record the commit you run with
python scripts/verify_phase3_ready.py     # must print READY
```

### Step 1 — start VM
```bash
./cloud/scripts/start_vm.sh
gcloud compute ssh "$VM_NAME" --zone "$GCP_ZONE"
cd /opt/glo-nca && ./cloud/scripts/setup_gcp.sh
```

### Step 2 — infrastructure smoke test (synthetic split; MUST pass first)
```bash
./cloud/scripts/verify_gcp.sh
./cloud/scripts/run_training.sh configs/smoke_test.yaml
./cloud/scripts/monitor.sh
# round-trip from local PC:
./cloud/scripts/download_experiment.sh <smoke_experiment_id>
python cloud/scripts/validate_checkpoint.py experiments/<smoke_experiment_id>
```
STOP and fix infrastructure if anything fails. Never use the final run as a test.

### Step 3 — validate the REAL BraTS dataset
```bash
python scripts/validate_dataset.py --root "$VM_DATA_DIR"     # PASS required
```

### Step 4 — create the master split ONCE (shared by all 5 experiments)
```bash
python scripts/create_master_split.py --data-root "$VM_DATA_DIR"
# writes split/master_split.json (patient IDs only). Records the ACTUAL case
# count (do not assume 882). Refuses to overwrite without --force.
```

### Step 5 — validate the master split
```bash
python scripts/check_split.py --split split/master_split.json --data-root "$VM_DATA_DIR"
# must print SPLIT OK (disjoint partitions, full coverage, fingerprint matches)
```

### Step 6 — GCP preflight (cheap gate before GPU spend)
```bash
python scripts/preflight_gcp.py --data-root "$VM_DATA_DIR" --split split/master_split.json
# must print PREFLIGHT: PASS (GPU, CUDA, dataset, master split, configs, disk)
```

### Steps 7-11 — ablations A0-A3 then verify each used the master split
All five configs already point at `split/master_split.json`; each run FAILS if
that file is missing (no silent regeneration), guaranteeing an identical split.
```bash
./cloud/scripts/run_training.sh configs/ablation_baseline.yaml   # A0: F/F  (Step 7)
# Step 8 — verify A0 used the master split:
grep "MASTER split" /out/<A0_id>/logs/training.log
python scripts/check_split.py --experiment /out/<A0_id> --data-root "$VM_DATA_DIR"
./cloud/scripts/run_training.sh configs/ablation_se.yaml         # A1: T/F  (Step 9)
./cloud/scripts/run_training.sh configs/ablation_spatial.yaml    # A2: F/T  (Step 10)
./cloud/scripts/run_training.sh configs/ablation_full.yaml       # A3: T/T  (Step 11)
```
All four share the SAME 200-epoch budget + the SAME master split; only the two
switches differ. Confirm every experiment_manifest.json shows the SAME
`split.split_sha256`.

### Step 12 — final GLO-NCA run
```bash
./cloud/scripts/run_training.sh configs/gcp_full.yaml            # name: GLO-NCA-V2-final
```
Best-checkpoint (validation smoothed), validation threshold tuning, frozen test
@0.5 and @tuned, per-case metrics, statistical summary, diagnostic report — all
automatic. Uses the SAME master split fingerprint as A0-A3.

### Steps 13-15 — download archives, then aggregate thesis tables + figures
```bash
# download all five finished experiments locally (or run these on the VM)
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

### Steps 16-17 — archive, verify, STOP VM (cost control)
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
