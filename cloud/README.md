# GLO-NCA V2 — GCP Cloud Operations Manual (Phase 2)

This is the operational manual for running GLO-NCA V2 on Google Cloud. It is an
**infrastructure layer** around the Phase 1 experiment system — it does **not**
change any research methodology. The GPU VM is disposable; **Google Cloud
Storage (GCS) is the durable source of truth**, so a lost VM never loses the
experiment.

> **Honesty note.** These scripts were authored and locally verified (shell
> syntax, Python compile, config validation, sync-logic against a local fake
> bucket). They were **NOT** run against a real GCP project / GPU here — no
> credentials or GPU were available. Every step below is standard `gcloud` /
> Docker usage; treat "not cloud-verified" items as needing a real run.

---

## Architecture

```
Local PC ──upload──▶ GCS bucket ──sync──▶ GPU VM (local SSD cache)
                        ▲                      │
                        │                   Docker: GLO-NCA V2
                        │                      │
                   experiment  ◀──periodic sync──┘
                     outputs
                        │
Local PC ◀──download────┘
```

- **GCS = durable master.** Dataset and all experiment outputs live here.
- **VM local SSD = training cache.** GLO-NCA does heavy random-patch access to
  many small NIfTI files; a local SSD is much faster and more reliable than a
  network filesystem (gcsfuse). The dataset stays recoverable from GCS.
- **systemd + Docker** run training so it survives SSH disconnect.
- **Periodic GCS sync** (default every 5 min) means a VM failure loses at most
  one interval of progress; resume continues from the last valid checkpoint.

---

## Prerequisites

- A Google Cloud account with **billing enabled**.
- **`gcloud` CLI** installed and authenticated (`gcloud auth login`).
- **GPU quota** in your chosen region (request via IAM & Admin → Quotas if you
  have none — new projects often start at 0 GPUs).
- Docker (installed on the VM by `setup_gcp.sh`; the Deep Learning VM image
  already includes it).
- The local BraTS dataset (one folder per patient).

No service-account JSON keys are needed: the VM uses its **attached service
account** (Application Default Credentials) with `cloud-platform` scope. **Never
commit `cloud/config/gcp.env` or any credentials.**

---

## First-time setup

```bash
# 1. Authenticate locally
gcloud auth login

# 2. Configure (copy the example, then edit with your values)
cp cloud/config/gcp.env.example cloud/config/gcp.env
$EDITOR cloud/config/gcp.env          # set project, zone, bucket, VM, GPU

# 3. Create the bucket (idempotent)
./cloud/scripts/create_bucket.sh

# 4. Create + start the GPU VM (asks for confirmation; billing starts)
./cloud/scripts/start_vm.sh

# 5. On the VM: run setup (Docker, toolkit, repo checkout, image build)
gcloud compute ssh "$VM_NAME" --zone "$GCP_ZONE"
cd /opt/glo-nca && ./cloud/scripts/setup_gcp.sh

# 6. Verify the whole stack (PASS/WARN/FAIL)
./cloud/scripts/verify_gcp.sh

# 7. Upload the dataset (validates BEFORE upload; refuses invalid data)
#    run this from the machine that holds the dataset (local PC or VM)
./cloud/scripts/upload_dataset.sh /path/to/BraTS
```

---

## Cloud smoke test (do this before any expensive run)

```bash
# on the VM
./cloud/scripts/run_training.sh configs/smoke_test.yaml
./cloud/scripts/monitor.sh                       # watch it
# when it completes, from your local PC:
./cloud/scripts/download_experiment.sh <experiment_id>
```
The smoke test exercises GPU → Docker → dataset → validator → training →
checkpoint → metrics → TensorBoard → GCS sync → download → resume, cheaply.

---

## Normal training (full run)

```bash
./cloud/scripts/start_vm.sh                       # 1. start VM
./cloud/scripts/verify_gcp.sh                     # 2. verify (on the VM)
./cloud/scripts/run_training.sh configs/gcp_full.yaml   # 3. ONE command
./cloud/scripts/monitor.sh                        # 4. monitor
#   training runs under systemd -> safe to disconnect SSH now
sudo journalctl -u glo-nca-training -f            #    live logs
./cloud/scripts/sync_experiment.sh /out/<experiment_id>  # 5. (optional manual sync)
./cloud/scripts/stop_vm.sh                        # 6. stop VM when done (saves $)
```

Full training **never starts automatically** — you run step 3 yourself.

---

## Resume (after a VM was stopped or lost)

```bash
./cloud/scripts/start_vm.sh                       # new or same VM
cd /opt/glo-nca && ./cloud/scripts/setup_gcp.sh   # if a fresh VM
./cloud/scripts/resume_training.sh <experiment_id>
#   -> downloads the experiment from GCS
#   -> validates the checkpoint (refuses to run on a corrupt one)
#   -> reuses the ORIGINAL config + experiment ID (no config drift, no new run)
#   -> resumes from last.pth (or newest valid periodic checkpoint)
./cloud/scripts/monitor.sh
./cloud/scripts/stop_vm.sh                         # when done
```

---

## Download final results

```bash
./cloud/scripts/download_experiment.sh <experiment_id>
# lands in ./experiments/<experiment_id>/ with the full Phase 1 package:
# best.pth, last.pth, metrics/, graphs/, tensorboard/, logs/, config/, split/,
# reports/ (results.json, diagnostic_report.*, thesis_results.csv), manifest,
# and cloud_metadata.json.
```

## TensorBoard (private, via SSH tunnel)

```bash
gcloud compute ssh "$VM_NAME" --zone "$GCP_ZONE" -- -L 6006:localhost:6006
# on the VM:
tensorboard --logdir /out/<experiment_id>/tensorboard --port 6006
# then open http://localhost:6006 on your local machine
```
Do **not** expose TensorBoard on a public IP.

---

## GPU selection guide

GLO-NCA is tiny (~30K params); the memory cost is the **3D NCA activation
volumes**, driven by `patch_size` (Phase 1 hardware guide): 64³ ≈ 2.4 GB,
96³ ≈ 5–6 GB, 128³ ≈ 10–19 GB (batch size 1).

| GPU (GCP) | VRAM | Fits 96³ | Fits 128³ | Notes |
|---|---|---|---|---|
| **T4** (`nvidia-tesla-t4`) | 16 GB | ✅ | ⚠️ tight | cheapest; good default for 96³ |
| **L4** (`nvidia-l4`, G2) | 24 GB | ✅ | ✅ | modern, efficient, headroom for 128³ |
| **V100** (`nvidia-tesla-v100`) | 16 GB | ✅ | ⚠️ | older; fine but pricier than T4/L4 |
| **A100 40GB** (A2) | 40 GB | ✅ | ✅ | overkill for this model; most expensive |

**Recommendation:** **T4 for `patch_size: 96`** (the `gcp_full.yaml` default) —
lowest cost, ample VRAM. Use **L4** only if you switch to `patch_size: 128`.
A100 is not needed for a 30K-parameter model.

> GPU availability varies by region/zone — if creation fails with a capacity or
> quota error, try another zone or request quota.

---

## Cost control

```
GPU VM RUNNING  = you are paying (per-second, per-GPU + machine)
GPU VM STOPPED  = compute cost stops (you still pay a little for disk + GCS)
```

Workflow: **start → train → sync → stop.** `stop_vm.sh` halts billing without
deleting anything. Nothing here ever auto-deletes a VM, disk, or bucket, and
full training never auto-starts.

### Cost estimate (reported prices — NOT verified by me)

On-demand, US regions, third-party-reported for 2026 (**verify on the official
page before relying on them**): **T4 ≈ $0.54/GPU-hr**, **L4 ≈ $0.70/GPU-hr**,
A100 40GB ≈ $3.67/GPU-hr — plus the VM machine type (e.g. `n1-standard-8`) and
disk. A full 200-epoch run's wall-clock depends on the GPU and dataset size, so
I will not fabricate a total; estimate it as `hours × (GPU + machine + disk)`
once you know the per-epoch time from the smoke/early epochs. **Spot/preemptible
instances** cut GPU cost substantially for interruptible work (resume covers the
interruptions), but confirm current rates yourself.

Official pricing: https://cloud.google.com/products/compute/gpus-pricing

Sources (reported pricing, third-party):
- https://www.thundercompute.com/blog/google-cloud-gpu-instances
- https://www.thundercompute.com/blog/nvidia-t4-pricing
- https://cloud.google.com/products/compute/gpus-pricing

---

## Storage design

Recommend a **200 GB pd-ssd** boot/data disk (`DISK_SIZE_GB=200`):
- BraTS 2024 small dataset ≈ a few GB unzipped (882-case full set is larger —
  size the disk to your dataset + margin).
- Experiment outputs: checkpoints (`last.pth`, `best.pth`, periodic snapshots)
  are small for a 30K-param model, but periodic snapshots + logs + TensorBoard
  accumulate; 200 GB is comfortable headroom.
- The dataset is **never** stored only inside the Docker image — it lives in GCS
  and is cached to the disk.

---

## Failure recovery (what survives what)

| Event | Outcome |
|---|---|
| SSH disconnect / terminal closed | training continues (systemd) |
| Docker process dies | systemd unit exits with its code; **no auto-loop** (Restart=no). Resume manually. |
| VM restart | experiment is in GCS; resume on the VM |
| VM lost entirely | new VM → `setup_gcp.sh` → `resume_training.sh <id>` |
| Checkpoint corrupt | `validate_checkpoint.py` refuses it, falls back to newest valid periodic; corrupt file left untouched |
| GCS sync fails | training keeps running; local data kept; sync retries |
| Disk nearly full | `verify_gcp.sh` warns; monitor shows free space |

---

## Scripts reference

| Script | Purpose |
|---|---|
| `create_bucket.sh` | create GCS bucket + layout (idempotent) |
| `start_vm.sh` / `stop_vm.sh` / `status.sh` | VM lifecycle + cost control |
| `setup_gcp.sh` | prepare the VM (Docker, toolkit, repo, image) |
| `verify_gcp.sh` | PASS/WARN/FAIL environment check |
| `upload_dataset.sh` | validate-then-upload dataset to GCS |
| `cache_dataset.sh` | sync + validate dataset onto the VM's local disk |
| `run_training.sh` | start training (systemd + Docker), one command |
| `resume_training.sh` | restore from GCS + validate + resume |
| `sync_experiment.sh` | push an experiment dir to GCS (`--watch` to loop) |
| `download_experiment.sh` | pull a finished experiment to the local PC |
| `monitor.sh` | live status (experiment, GPU, disk) |
| `validate_checkpoint.py` | checkpoint integrity gate for resume |

See also `cloud/docker/README.md` for container details.
