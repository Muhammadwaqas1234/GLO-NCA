# GLO-NCA V3 — Colab Drive → GCS Transfer

**Date:** 2026-09-15 (UTC) · No GPU VM started · No training · Master split untouched.

## What this task delivered
A runnable Colab notebook that streams the dataset ZIP **directly from your
Google Drive into your GCS bucket** (no Windows upload, no anonymous datacenter
download):

- **Notebook:** [`notebooks/glo_nca_drive_to_gcs.ipynb`](../../notebooks/glo_nca_drive_to_gcs.ipynb)
  (8 cells, validated: well-formed `.ipynb`, all code cells parse).

## Execution status: COMPLETE — GATE PASS (run by user in Colab as mo.waqas@techclomate.com)

The notebook ran successfully. Cell 7 report (actual values, not invented):

```
SOURCE            : Google Drive file 1gkz4kM89PUdqI4rbpi1KN5TYGfBGYe5k
SOURCE NAME       : MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip
DESTINATION       : gs://glo-nca-v3-even-continuity-501915/datasets/brats-met-2025/MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip
SIZE (src=dst)    : 33467863309 bytes (33.47 GB)
CONTENT TYPE      : application/zip
GENERATION        : 1789422304067291
AUTH              : Colab authenticate_user (no tokens exposed)
PROJECT           : even-continuity-501915-f9
GCS VERIFICATION  : PASS
DATA TRANSFER GATE: PASS
GPU               : NOT STARTED
TRAINING          : NOT STARTED
```

The 403 seen on the first attempt was an account mismatch (Colab signed in as
`raiwaqasabid2@gmail.com`; bucket/project owned by `mo.waqas@techclomate.com`).
Re-running Colab authenticated as the bucket owner resolved it with no IAM change.
`BUCKET LOCATION: None` in the print is a cosmetic client-attribute quirk (not
reloaded); the bucket is us-central1 and the object verified (size match + ZIP magic).

### (original) Execution status: AWAITING USER RUN (browser auth required)
The transfer itself must be run **by you** in Colab, because only you can perform
the Google browser authorization for Drive + Cloud (credentials must never leave
the authenticated Colab session or be pasted into chat). I prepared and validated
the notebook; I did **not** (and cannot) execute it or fabricate transfer numbers.

### How to run (≈ a few minutes at Colab↔Google bandwidth)
1. Go to **colab.research.google.com** → **File → Upload notebook** →
   `notebooks/glo_nca_drive_to_gcs.ipynb`.
2. **Runtime → Run all.**
3. When prompted, complete the **two authorizations** in the popups:
   Google account (Drive) and Google Cloud. Authorize the account that owns the
   Drive file and has access to project `even-continuity-501915-f9`.
4. Watch Cell 5 print live progress (%, GB, MB/s, ETA); Cell 6 verifies; Cell 7
   prints the final report. Copy Cell 7's output back here and I will record the
   actual size/duration/speed and flip this gate to PASS.

## Fixed parameters (verified from this environment where possible)
| Item | Value |
|---|---|
| Google Drive file id | `1gkz4kM89PUdqI4rbpi1KN5TYGfBGYe5k` |
| Expected filename | `MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip` |
| Expected size | ~33.47 GB (33,467,863,309 bytes, from HTTP content-range) |
| GCP project | `even-continuity-501915-f9` |
| Bucket | `gs://glo-nca-v3-even-continuity-501915` — **verified exists**, location US-CENTRAL1 |
| Destination | `datasets/brats-met-2025/MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip` |
| Prior partial object | **none** (destination verified clean this session) |

## How the notebook meets the requirements
- **Auth:** `google.colab.auth.authenticate_user()` (one Colab OAuth for Drive API
  + GCS). No tokens/cookies/keys handled outside Colab.
- **Source verify (Cell 3):** Drive API `files.get` returns name/size/mime;
  asserts size > 30 GB and `.zip` — does not trust the name alone.
- **Destination verify (Cell 4):** confirms bucket exists; deletes **only** a
  prior object at this exact path whose size ≠ source (never unrelated objects).
- **Streaming transfer (Cell 5):** `MediaIoBaseDownload` (64 MB chunks) → GCS
  resumable `blob.open('wb', chunk_size=...)`. **Never loads 33 GB into RAM and
  needs no 33 GB Colab disk.** Skips if a complete object already exists.
- **Upload verify (Cell 6):** asserts `dst_size == src_size` and reads the first
  4 bytes from GCS to confirm ZIP magic `PK`. Does **not** extract in Colab.
- **Report (Cell 7):** prints size / content-type / generation / PASS-FAIL.

## Values to be filled from the actual run
Transfer duration, average speed, and final object size/generation — **from Cell 7
output** (not invented). Paste them here after running.

## Untouched (per rules)
Master split `split/master_split.json` unchanged (SHA
`d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d`). No model,
config, architecture, epoch, or checkpointing changes. GPU VM `glo-nca-v3-l4` and
CPU VM `glo-nca-xfer` both TERMINATED. Local authoritative dataset not deleted.

## Final gate
```
DATA TRANSFER GATE: PASS — dataset streamed Drive→Colab→GCS and verified
(33,467,863,309 bytes src==dst, content-type application/zip, ZIP magic PK,
generation 1789422304067291). No GPU started, no training started.
```

Once you paste the Cell 7 report showing SIZE MATCH + ZIP MAGIC PASS, this gate
becomes **PASS** and the dataset is available at
`gs://glo-nca-v3-even-continuity-501915/datasets/brats-met-2025/…`. Do not start
the 3-epoch smoke test automatically — that is a separate task.
