#!/usr/bin/env bash
# GCP-side transfer: Drive(public,confirm-token) -> disk -> extract -> verify -> GCS
set -uo pipefail
BUCKET=gs://glo-nca-v3-even-continuity-501915
PREFIX=datasets/brats-met-2025
L=~/xfer2.log
echo "[x2] $(date -u) start" | tee "$L"
sudo mkdir -p /mnt/data && sudo chown "$USER":"$USER" /mnt/data
df -h /mnt/data | tee -a "$L"
echo "[x2] downloading ZIP (confirm-token, auth-free)..." | tee -a "$L"
python3 ~/drive_dl.py /mnt/data/batch1.zip 2>&1 | tee -a "$L"
[ -f /mnt/data/batch1.zip ] || { echo "[x2] DOWNLOAD_FAILED" | tee -a "$L"; exit 2; }
ls -la /mnt/data/batch1.zip | tee -a "$L"
echo "[x2] verifying ZIP is a real archive..." | tee -a "$L"
if ! unzip -l /mnt/data/batch1.zip >/dev/null 2>&1; then echo "[x2] BAD_ZIP" | tee -a "$L"; exit 3; fi
ZIP_ENTRIES=$(unzip -l /mnt/data/batch1.zip | tail -1 | awk '{print $2}')
echo "ZIP_ENTRIES=$ZIP_ENTRIES" | tee -a "$L"
echo "[x2] extracting..." | tee -a "$L"
mkdir -p /mnt/data/extracted
unzip -q -o /mnt/data/batch1.zip -d /mnt/data/extracted 2>&1 | tail -3 | tee -a "$L"
NSEG=$(find /mnt/data/extracted -name '*-seg.nii.gz' | wc -l)
echo "EXTRACTED_SEG_COUNT=$NSEG" | tee -a "$L"
# dataset root = dir whose subtree holds the cases
ROOT=$(find /mnt/data/extracted -maxdepth 3 -type d -name 'MICCAI-LH*' | head -1)
[ -z "$ROOT" ] && ROOT=$(dirname "$(find /mnt/data/extracted -name '*-seg.nii.gz' | head -1)" | xargs dirname)
[ -z "$ROOT" ] && ROOT=/mnt/data/extracted
echo "DATASET_ROOT=$ROOT" | tee -a "$L"
echo "[x2] syncing to GCS (VM service account)..." | tee -a "$L"
gcloud storage rsync -r "$ROOT" "$BUCKET/$PREFIX" 2>&1 | tail -4 | tee -a "$L"
GCS_SEG=$(gcloud storage ls -r "$BUCKET/$PREFIX/**" 2>/dev/null | grep -c 'seg.nii.gz')
echo "GCS_SEG_COUNT=$GCS_SEG" | tee -a "$L"
du -sh "$ROOT" | tee -a "$L"
echo "X2_COMPLETE $(date -u)" | tee -a "$L"
