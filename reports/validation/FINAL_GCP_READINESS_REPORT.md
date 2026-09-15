# GLO-NCA V3 — Final Pre-GCP Production Audit & 300-Epoch Handoff

**Date:** 2026-09-13 (audit) · 2026-09-14 (low-cost GCP deployment plan) · **Branch:** `v3-multilevel` · **Frozen V2 base:** `f9e5501`
**Dataset:** MICCAI-LH-BraTS2025-MET-Challenge-Training

> Repository-side audit only. No full training, no GCP usage, no dataset upload,
> no commit. Real-data results come from the actual production runner on real
> BraTS-MET cases; they are software-correctness proofs, **not** scientific
> metrics. GPU memory fit at 96³/128³ is **NOT** tested locally — it must run on
> the GCP GPU.

---

## Final facts (verified this audit)
| Item | Value | Status |
|---|---|---|
| V3 parameter count | **40,656** | PASS (== expected) |
| Production levels | 32³ / 96³ / 128³ | PASS |
| Unified model / single head | yes (not ensemble/averaging) | PASS |
| Dataset valid cases | **1296** (650 top-level + 646 UCSD) | PASS |
| Distinct subjects | **810** (287 multi-timepoint) | PASS |
| Duplicate case IDs | 0 | PASS |
| Master split | 898 / 200 / 198 cases; 567 / 121 / 122 subjects | PASS |
| Master split SHA256 | `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d` | PASS (matches) |
| Subject leakage | 0 straddling subjects | PASS |
| Split covers dataset | 1296 == 1296 (0 missing / 0 unknown) | PASS |
| Final epochs | **300** (locked) | PASS |
| V2 methodology core | byte-unchanged vs `f9e5501` | PASS |
| V2 parameter count | 30,138 | PASS |

---

## PASS — verified locally this audit
- **V2 frozen:** `git diff f9e5501` on V2 model, agents, losses, `metrics_eval.py`,
  and all V2 configs = empty. V2 params 30,138. V2 smoke runs end-to-end through
  the shared runner (prior task, COMPLETED).
- **Recursive discovery backward-compatible:** flat synthetic dataset → `rel==id`,
  V2 regression COMPLETED; nested BraTS-MET → 1296 cases, 646 UCSD found,
  containers skipped, 0 dups.
- **Master split:** SHA256 matches the canonical value; covers the dataset exactly;
  pairwise-disjoint; **subject-disjoint (0 leakage)**; not regenerated/modified.
- **Dataset identity:** case_count 1296, subject_count 810, patient_id_hash over
  case IDs; recorded in the experiment manifest + `dataset_identity.json`.
- **V3 architecture:** one unified `nn.Module`, single seg head; nested shape
  transitions verified (L1→proj→L2→proj→L3→learned concat fusion→(B,3,X,Y,Z));
  40,656 params; levels [32,96,128]; SE + spatial on; concat fusion.
- **V3 software suite (`validate_v3_local.py`, 18 checks):** build, shapes,
  forward, backward, finite grads, grad-clip, optimizer, scheduler, EMA,
  checkpoint, reload, resume (epoch 2, not 1), reproducibility (seed 42 identical),
  evaluation (Dice/mIoU/HD95), threshold discipline, frozen-test discipline,
  HD95-in-voxels — all PASS.
- **Real-data end-to-end (production `train.py`, 20 real cases, GPU):** discovery →
  validation PASS → subject-disjoint split → real preprocessing → V3 forward →
  loss → EMA → checkpoint → validation → validation-only threshold tuning →
  single-pass test → manifest → COMPLETED. WT/TC/ET regions present.
- **Real-data resume (production `--resume`):** prev_epoch=2 → next_epoch=3, reused
  saved split, restored optimizer/scheduler/EMA/RNG/best — no reset. This is the
  300-epoch restart semantics, proven on real data.
- **Scheduler for 300 epochs:** cosine `T_max = epochs × steps_per_epoch`; resume
  loop `range(start_epoch, epochs)` continues at N+1. PASS.
- **Extension safety:** `Workspace.create` always makes a fresh timestamped dir
  and never overwrites; `--resume` operates only on an explicit existing dir. A
  new launch of the final config creates `GLO-NCA-V3-MultiLevel-<timestamp>/`,
  never clobbering a prior run. PASS (see WARNING for the extend-past-300 policy).
- **Manifest completeness:** experiment_id, name, git, dataset (root/modalities/
  counts/**case_count/subject_count**/patient_id_hash), split (source/sha/file/
  version), model (version/arch/params/levels/fusion), training (epochs/batch/
  patch/aug), optimizer, scheduler, loss (β0.75/γ1.33), ema, seed, gpu, software,
  best_epoch, final_test, status. PASS.
- **Ablations (V3-A/B/C, + V3-D = full):** all share identical control variables
  (lr, wd, loss, EMA, grad-clip, seed 42, batch 1, light aug, **300 epochs**,
  fire_rate, SE, spatial, master split); differ only in enabled levels / fusion.
  Construct with 33,089 / 28,223 / 39,872 / 40,656 params. Not run locally.
- **Cloud scripts:** all 16 bash scripts syntactically valid; `pretrain_gate.sh`
  V2+V3-aware (adds V3 param report + 96³/128³ GPU gate + skips V2-only audit for
  V3); `upload_dataset.sh` recursive + verifies by discovered cases + preserves
  `UCSD - Training/`.
- **Config lock:** `configs/v3_multilevel.yaml` = v3, 300 epochs, batch 1, seed 42,
  32/96/128, SE+spatial, concat fusion, light aug, Focal-Tversky+BCE β0.75/γ1.33,
  AdamW, cosine, EMA, grad-clip, references master split. Parses.
- **compileall** (src, scripts, train.py, validate_checkpoint) clean.
- **Git/secret safety:** no dataset/NIfTI/checkpoints/experiments/`.env`/keys
  tracked; `.gitignore` + `.dockerignore` cover them; `gcp.env.example` is a
  template; real `gcp.env` absent and ignored.

## NOT TESTED LOCALLY — requires the GCP GPU
- **96³ and 128³ TRUE FIT** on the target GPU (`scripts/gpu_memory_gate_v3.py`).
  Local RTX 3050 6 GB OOMs at 96³ (expected, correctly reported — not a defect).
- **Real-data GCP smoke** inside the container on the GPU (`pretrain_gate.sh
  configs/v3_multilevel.yaml`): real fwd/bwd/opt/ckpt/reload/eval at production
  resolution.
- **GCS dataset upload + integrity** (all 1296 cases incl. nested UCSD).
- **Exhaustive per-voxel NIfTI validation of all 1296 cases**: DEFERRED locally
  (laptop I/O ≈ 40+ min for 6480 volume loads). Sampled cases + structural/coverage
  validation passed; the full `validate_dataset` runs in the GCP gate on faster
  storage. Marked: **LOCAL EXHAUSTIVE VALIDATION: DEFERRED**.

## BLOCKED
- None (repository-side). No known blockers preventing the GCP pre-flight.

## WARNINGS (non-blocking)
- **Master split created with `--skip-validate`.** Its *content* is fully
  validated (recursive discovery, coverage, disjointness, subject-leakage gate;
  fingerprint reproducible). Only the exhaustive per-voxel re-scan was skipped
  locally (see DEFERRED). Re-verified by `check_split.py` (SPLIT OK).
- **Extend-past-300 policy is documentation, not code-enforced.** Immutability of a
  finished `Final` holds as long as you do **not** `--resume` a completed 300-epoch
  experiment with a raised epoch count (that would extend in place). To extend,
  launch a **new** experiment (fresh timestamped dir) — e.g. name it
  `…-extension-001` — so the original 300-epoch artifact stays immutable. A fresh
  300-epoch and a 300→600 continuation are **not** scientifically equivalent.
- **GPU reproducibility is not bit-exact** (cudnn default mode) — consistent with
  V2's honest stance; RNG state is checkpointed/restored.

## FINAL STATUS
```
READY FOR GCP PRE-FLIGHT
```

---

## Files changed (vs `f9e5501`)
```
PHASE3_RUNBOOK.md                     (V3 docs)
cloud/scripts/upload_dataset.sh       (recursive verify; preserves UCSD)
scripts/check_split.py                (subject-disjointness check)
split/README.md                       (scope/timepoint/subject/leakage policy)
src/datasets/Nii_Gz_Dataset_3D.py     (getFilesInPath: recursive discovery — DISCOVERY ONLY)
src/experiment/config.py              (v3 patch-table bypass)
src/experiment/dataset_validation.py  (recursive, cohort counts)
src/experiment/datasource.py          (discover_cases/subject_of; subject-disjoint split + leakage gate)
src/experiment/runner.py              (V2/V3 dispatch; nested-path map; subject_count in identity/manifest)
configs/v3_multilevel.yaml            (epochs 200 -> 300)      [tracked-new this branch]
configs/v3_ablation_{A,B,C}*.yaml     (epochs 200 -> 300)      [tracked-new this branch]
```
V2 **methodology core** (model, agents, losses, metrics, all V2 configs) = unchanged.

## Files added (this audit)
```
FINAL_GCP_READINESS_REPORT.md   (this report)
```
(Plus, from prior phases and still untracked: `split/master_split.json`,
`scripts/gpu_memory_gate_v3.py`, `scripts/validate_v3_local.py`,
`scripts/local_gpu_smoke_test.py`, `cloud/scripts/pretrain_gate.sh`,
`configs/v3_*.yaml`, `src/models/Model_GLO_NCA_V3.py`, `src/agents/Agent_GLO_NCA_V3.py`,
`REAL_DATA_V3_VALIDATION_REPORT.md`.)

## Tests executed → results
| Test | Result |
|---|---|
| compileall (src/scripts/train.py) | PASS |
| V2 methodology-core diff vs f9e5501 | PASS (empty) |
| V2 regression (flat synth, runner) | PASS (30,138 params, COMPLETED) |
| Recursive discovery (nested BraTS-MET) | PASS (1296/810, 646 UCSD, 0 dups) |
| Master split fingerprint + coverage + leakage | PASS (sha matches; 0 leak; exact coverage) |
| V3 build / params / levels | PASS (40,656; 32/96/128) |
| V3 shape transitions (nested) | PASS |
| V3 software suite (18 checks) | PASS |
| Real-data end-to-end (20 cases, GPU) | PASS (COMPLETED) |
| Real-data resume (2→3) | PASS (no reset) |
| Manifest completeness | PASS |
| Ablation control-variable identity | PASS |
| All configs parse; 300-epoch lock | PASS |
| Cloud scripts bash syntax (16) | PASS |
| GPU memory gate script (local) | ran; 96³ OOM on 6 GB (expected) — GCP required |
| Git/secret safety | PASS |

## Final artifacts
- **Dataset identity:** case_count = 1296, subject_count = 810.
- **Master split fingerprint:** `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`.
- **V3 parameter count:** 40,656.
- **300-epoch config:** `configs/v3_multilevel.yaml` (v3; 32/96/128; SE+spatial;
  concat fusion; batch 1; seed 42; light aug; Focal-Tversky+BCE β0.75/γ1.33; AdamW;
  cosine; EMA; grad-clip; master split).

---

## Final GCP handoff checklist
```
[x] Repository reviewed
[x] V2 frozen
[x] V3 architecture verified
[x] 40,656 parameters verified
[x] 1,296 cases verified
[x] 810 subjects verified
[x] Subject-disjoint split verified
[x] Master split SHA256 verified
[x] Dataset identity verified
[x] Recursive UCSD discovery verified
[x] V3 production config verified
[x] 300 epochs locked
[x] Checkpoint/resume verified (local + real data)
[x] Frozen-test discipline verified
[x] Ablations verified (control vars identical)
[x] Cloud transfer verified (syntax + recursive; UCSD preserved)
[x] GCP pretrain gate ready (V2+V3 aware)
[x] GPU memory gate ready (script; runs on GCP)
[ ] Real-data GCP smoke        -> GCP only
[ ] Spot/resume safety         -> resume proven locally; GCS-cadence to confirm on GCP
[x] Monitoring available (status.json + monitor.sh + metrics CSVs + tensorboard)
[x] No dataset tracked
[x] No credentials tracked
[x] No experiment outputs tracked
[x] No commit created
```

## Exact next commands
Review (this repo), then commit **manually** (I created no commit):
```bash
git status
git diff --stat
git diff
# review, then (your decision) commit incl. split/master_split.json
```
After you commit + push, on the GCP GPU VM:
```bash
cd /opt/glo-nca        # or your VM workspace
python scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel.yaml
./cloud/scripts/pretrain_gate.sh configs/v3_multilevel.yaml
```
Proceed to the 300-epoch campaign (and V3-A/B/C ablations) **only** after:
```
FINAL PRE-TRAINING GATE: PASS
GLO-NCA TRAINING: READY
```
and only if the memory gate reports **96³ + 128³ TRUE FIT** on the real GPU. If it
does not, STOP and pick a larger-VRAM GPU — do NOT reduce the V3 architecture.
```bash
./cloud/scripts/run_training.sh configs/v3_multilevel.yaml   # 300-epoch final (user-initiated)
```
```
=============================================
FINAL STATUS: READY FOR GCP PRE-FLIGHT
=============================================
```

---

# Low-Cost GCP Deployment Plan (added 2026-09-14)

Cost principle enforced end-to-end: **PREPARE → VALIDATE → TRAIN → SAVE → VERIFY
→ DELETE expensive compute → KEEP scientific results.** Spend GPU money only
while GPU computation is actually required.

## Recommended region / zone
- **`REQUIRES GCP EXECUTION` — verify live availability/quota/price before use.**
- Recommended: a single region that has the chosen GPU AND colocates the GCS
  bucket, to avoid cross-region egress on the 1,296-case dataset. `us-central1`
  (zones `-a/-b/-c/-f`) is a reasonable low-cost default that offers T4/L4/A100;
  fall back to another zone on a capacity/quota error. Set `GCP_REGION`/`GCP_ZONE`
  and the bucket's location to the **same region**.

## GPU selection strategy (decided by the real gate, not by price)
- Production requirement: **96³ TRUE FIT AND 128³ TRUE FIT** (host-memory spill is
  NOT a fit). V3's 128³ fine level + projections + fusion pushes the activation
  peak above a plain 128³ NCA.
- **Start with L4 24 GB (G2, `nvidia-l4`, `g2-standard-8`)** — lowest-cost GPU with
  genuine 128³ headroom. Confirm with `scripts/gpu_memory_gate_v3.py`.
- T4 16 GB is cheaper but **tight** at V3 128³ — only use it if the gate reports a
  true fit (it may not). If L4 fails 128³, step up to A100 40 GB. **Never** reduce
  the V3 architecture to fit a smaller GPU.
- Estimated GPU list prices (**ESTIMATED**, verify on GCP pricing): T4 ≈ $0.35,
  L4 ≈ $0.71, A100 40 GB ≈ $3.67 per GPU-hour, plus machine type + disk.

## GCS storage strategy (durable = cheap; compute = expensive)
- **Dataset (once):** `gs://<bucket>/datasets/brats/` via `upload_dataset.sh`
  (recursive rsync; **preserves `UCSD - Training/`**; verifies by discovered
  cases). Uploaded ONCE and reused by V3-A/B/C/D + Final — **no per-experiment
  dataset copies**.
- **Results (durable):** `gs://<bucket>/experiments/<experiment_id>/` — synced
  during training (background watcher, every `SYNC_INTERVAL_SECONDS`, `.tmp`
  excluded) and finally on completion. Retained forever.
- **Master split:** travels **in the repo** (`split/master_split.json`), never
  regenerated on GCP; its SHA256 is recorded in every manifest.

## Checkpoint strategy (300 epochs, restart-safe)
- `logging.checkpoint_frequency: 10` in the production config → periodic snapshots
  every 10 epochs, plus `last.pth` every epoch and `best.pth` on improvement.
  Each checkpoint carries model/optimizer/scheduler/EMA/epoch/best/history/config/
  RNG (verified locally). Raise the frequency for Spot if desired.
- The background watcher pushes checkpoints to GCS so a lost VM loses at most one
  sync interval.

## Spot / preemption strategy
- Supported and reliable via the existing design: interrupt → new VM →
  `resume_training.sh <exp_id>` (GCS restore → `validate_checkpoint.py` → Phase-1
  `--resume`, original ID + config preserved, no epoch reset). Proven locally
  (real-data resume 2→3). **Recommendation:** Spot is acceptable for ablations and
  for the final run *given* frequent checkpoints + verified resume; if in doubt
  for the single thesis Final run, use a standard VM to avoid churn.

## Cleanup / resource-deletion strategy
- **`stop_vm.sh`** — halt compute billing, keep VM+disk (between sessions).
- **`verify_results.sh <exp_id>`** — confirms manifest/checkpoints/metrics/graphs
  exist and are non-empty in GCS and that a downloaded checkpoint reloads.
- **`delete_vm.sh <exp_id>`** — runs verification first, then deletes the VM **and
  its boot disk** (all VM billing stops). `--force` only for throwaway runs.
- **`cost_report.sh` / `--audit`** — estimate hourly cost; list VM/disks/snapshots/
  static IPs so nothing costly is left overnight.
- **Always retained:** bucket, dataset, results, master split. Only GPU compute is
  destroyed. No storage is ever auto-deleted; no VM auto-created.

## Result-preservation procedure (before any deletion)
```
train → final checkpoint → GCS sync (auto) → verify_results.sh PASS
      → status == completed → delete_vm.sh (verifies again, then deletes)
```
Never `DELETE → hope`. Always `COMPUTE → SAVE → VERIFY → DELETE`.

## VERIFIED (repository-side, this task)
- New cost scripts added and syntax-valid: `verify_results.sh`, `delete_vm.sh`,
  `cost_report.sh`. All 19 cloud scripts pass `bash -n`; `compileall` clean.
- Existing durability confirmed: background + final GCS sync in
  `_train_entrypoint.sh`; `resume_training.sh` GCS→restore→validate→resume;
  `sync_experiment.sh` `.tmp`-safe; monitoring via `monitor.sh` + status.json +
  metrics CSVs + tensorboard.
- Dataset reuse: single GCS copy for all experiments; `upload_dataset.sh`
  preserves nested UCSD and verifies by discovered cases.
- Config template + README updated with the V3 GPU/gate/cost guidance.
- V2 methodology core still byte-frozen; no dataset/checkpoints/credentials
  tracked; no commit created.

## REQUIRES GCP EXECUTION (not done here; do not claim as passed)
- Live region/zone/GPU availability, quota, and current pricing.
- `gpu_memory_gate_v3.py` → 96³ + 128³ TRUE FIT on the real GPU.
- Dataset upload + exhaustive NIfTI validation of all 1,296 cases on GCP.
- Real-data GCP smoke, pre-training gate, checkpoint/resume on a real VM.
- Actual Spot preemption→resume on GCP.
- Real billing figures (report only ESTIMATED until billing data exists).

---

```
REPOSITORY STATUS:
READY FOR GCP

GCP STATUS:
READY FOR PRE-FLIGHT — NOT YET EXECUTED

TRAINING STATUS:
NOT STARTED

FINAL TRAINING TARGET:
GLO-NCA V3 — 300 EPOCHS

COST STRATEGY:
MINIMIZE GPU RUNTIME
REUSE ONE DATASET COPY
STORE RESULTS DURABLY
VERIFY RESULTS
DELETE/STOP EXPENSIVE COMPUTE AFTER COMPLETION
```

## Tomorrow's commands (after `git commit` + `git push`)
```bash
# 0) One-time GCP setup (fill cloud/config/gcp.env from the example first)
./cloud/scripts/create_bucket.sh          # durable bucket (same region as VM)
./cloud/scripts/upload_dataset.sh "C:/path/to/MICCAI-LH-BraTS2025-MET-Challenge-Training"
./cloud/scripts/start_vm.sh               # creates the L4 GPU VM (billing starts)
./cloud/scripts/setup_gcp.sh              # build image / prepare VM
./cloud/scripts/cache_dataset.sh          # GCS -> VM local SSD cache (+ validate)

# 1) On the VM: memory gate + pre-training gate (NO training yet)
cd /opt/glo-nca
python scripts/gpu_memory_gate_v3.py --config configs/v3_multilevel.yaml
./cloud/scripts/pretrain_gate.sh configs/v3_multilevel.yaml
# proceed ONLY if: 96^3 TRUE FIT + 128^3 TRUE FIT  AND
#                  FINAL PRE-TRAINING GATE: PASS / GLO-NCA TRAINING: READY

# 2) Ablations then the final 300-epoch run (reuse the one dataset copy)
./cloud/scripts/run_training.sh configs/v3_ablation_A_global96.yaml
./cloud/scripts/run_training.sh configs/v3_ablation_B_global128.yaml
./cloud/scripts/run_training.sh configs/v3_ablation_C_add_fusion.yaml
./cloud/scripts/run_training.sh configs/v3_multilevel.yaml       # V3-D / FINAL 300ep

# 3) SAVE -> VERIFY -> DELETE (per experiment id)
./cloud/scripts/verify_results.sh <experiment_id>     # must PASS
./cloud/scripts/cost_report.sh --audit                # nothing left billing?
./cloud/scripts/delete_vm.sh <final_experiment_id>    # verifies, then deletes VM+disk
```
