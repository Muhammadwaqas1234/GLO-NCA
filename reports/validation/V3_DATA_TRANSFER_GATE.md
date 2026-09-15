# GLO-NCA V3 — Real Dataset Transfer & Integrity Gate

**Date:** 2026-09-15 (UTC) · **No training run.** · **All VMs TERMINATED ($0 compute).**

## Result
```
DATA TRANSFER GATE: BLOCKED — Google Drive blocks automated download of the
dataset ZIP from the GCP datacenter IP (public-file quota/anti-abuse). The file
is public and downloads correctly from the user's laptop IP, but NOT from a GCP
VM, and no authenticated Drive access is available on the VM.
```

## Method attempted (per request: Drive → GCP directly, no Windows→GCS)
- **Source:** Google Drive folder `.../folders/1EkdfcEHGID2e7LRA3QhH2rxmrBbb5LAh`,
  single file `MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip`
  (id `1gkz4kM89PUdqI4rbpi1KN5TYGfBGYe5k`), **~33.47 GB** (33,467,863,309 bytes,
  confirmed via HTTP `content-range`).
- **Transfer VM:** `glo-nca-xfer` (e2-standard-4, us-central1-b, 150 GB disk, no GPU).
- **Download methods tried on the VM (auth-free, as required):**
  1. Custom confirm-token HTTP downloader (the exact flow that returns real ZIP
     bytes from the laptop). **On the VM it received a 2,009-byte HTML page**, not
     the file (`GOT_HTML_NOT_FILE`).
  2. `gdown` (upgraded to latest) by file id. **Failed:** *"You may still be able
     to access the file from the browser … but Gdown can't. Please check
     connections and permissions."*

## Diagnosis (reproducible, not a transient)
The file is **PUBLIC** — from the laptop IP the confirm-token flow returns real
ZIP bytes (verified: first bytes `PK\x03\x04`, `content-range … /33467863309`).
From the **GCP datacenter IP** the same requests return Google's HTML
quota/confirm page instead of the file. This is Google Drive's well-known
anti-abuse throttling of large public-file downloads originating from cloud IP
ranges. It is an **access/policy limitation at the source**, not a repository,
tooling, disk, or GCP-permission problem.

Per the task's STOP condition, automated auth-free Drive→GCP download cannot be
performed; I did not fall back to Windows→GCS and did not repeatedly retry.

## What is verified / unchanged
- **Complete dataset exists and is intact on the laptop:** 1,296 cases / 810
  subjects / 0 duplicates / 646 nested UCSD cases; master split covers it exactly;
  patient_id_hash `7e9ff2cf…` (re-verified this session).
- **Frozen master split unchanged:** SHA256
  `d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`, counts
  898/200/198, subjects 567/121/122, 0 subject leakage. Not regenerated.
- **GCS bucket ready (empty):** `gs://glo-nca-v3-even-continuity-501915`
  (`datasets/brats-met-2025/`), us-central1.

## VM / cost / git
- `glo-nca-xfer` (CPU): **TERMINATED**. `glo-nca-v3-l4` (GPU): **TERMINATED**.
  No compute billing. Transfer cost: a few minutes of one e2 CPU VM (~cents);
  no bytes egressed from Drive.
- Git: no `.nii`/`.zip`/checkpoints/credentials tracked; helper scripts kept out
  of the repo. Supervisor-clean.

## To unblock (owner action required — pick one)
1. **Authenticated Drive download on the VM (recommended, keeps Drive→GCP):**
   configure `rclone` with a Google Drive remote using **your** OAuth. You run
   `rclone config` locally/in-browser once to authorize; provide the resulting
   token to the VM **securely** (not pasted into chat) — e.g. `rclone authorize
   "drive"` on your PC, then transfer the token to the VM via `gcloud compute scp`
   or the metadata server. Then `rclone copy` pulls the file at full datacenter
   speed. Authenticated requests are **not** subject to the anonymous
   cloud-IP quota block.
2. **Re-host the ZIP where cloud IPs can fetch it:** upload the ZIP once to the
   **GCS bucket directly from Drive via Google's own "Transfer"/Colab** (a Colab
   notebook runs as your authenticated Google account and can copy Drive→GCS), or
   to any HTTPS endpoint the VM may fetch (signed URL). Then the VM `gcloud
   storage cp` it internally.
3. **Windows → GCS** authenticated upload of the local copy (you asked to avoid
   this due to ~1.1 MB/s upstream ≈ ~9 h; listed only for completeness).

Recommended: **option 1 (rclone with your Drive OAuth)** — it satisfies the
"Drive → GCP directly, fast, no Windows upload" goal; the anonymous cloud-IP
block does not apply to authenticated requests. I can script the VM side and tell
you the exact one-time local `rclone authorize` step (no secrets in chat).

**Gate is NOT PASS. Do not proceed to the 3-epoch smoke test until the dataset is
in GCS and the frozen split is verified against it.**
