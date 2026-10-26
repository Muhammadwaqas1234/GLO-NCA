#!/usr/bin/env bash
# Stage BraTS 2021 glioma training data (Kaggle) to GCS through a temporary CPU VM: stage_dataset.sh [--keep-vm]
# The VM runs the job as its startup script (no SSH); progress is written to <dataset>/_staging/log.txt.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

KAGGLE_DATASET="dschettler8845/brats-2021-task1"
KAGGLE_FILE="BraTS2021_Training_Data.tar"
EXPECTED_CASES=1251
STAGE_VM="${VM_NAME}-stage"
KEEP_VM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-vm) KEEP_VM=1; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

: "${DATA_PREFIX:?set DATA_PREFIX in gcp.env}"
[[ "${DATA_PREFIX}" == *brats2021* ]] || die "DATA_PREFIX must point to the BraTS 2021 glioma data: ${DATA_PREFIX}"
DEST="gs://${GCS_BUCKET}/${DATA_PREFIX}"
WORK="gs://${GCS_BUCKET}/_staging"
KAGGLE_KEY="${KAGGLE_CONFIG_DIR:-${HOME}/.kaggle}/kaggle.json"
[[ -f "${KAGGLE_KEY}" ]] || die "Kaggle API key not found at ${KAGGLE_KEY}"
need gcloud

if gcs_exists "${DEST}/MANIFEST.txt"; then
  pass "already staged: ${DEST} ($(gcs cat "${DEST}/MANIFEST.txt" | grep -m1 '^cases='))"
  exit 0
fi

log "staging ${KAGGLE_DATASET}/${KAGGLE_FILE} -> ${DEST} via temporary VM ${STAGE_VM} (${GCP_ZONE})"
confirm "Create temporary VM ${STAGE_VM} (e2-standard-4, 100 GB disk, about \$0.20/h; deleted afterwards)?"

cleanup() {
  gcs rm -q "${WORK}/kaggle.json" >/dev/null 2>&1 || true
  [[ "${KEEP_VM}" == "1" ]] && { warn "keeping ${STAGE_VM}; delete it yourself when done"; return; }
  gcloud compute instances delete "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" --quiet \
    >/dev/null 2>&1 && pass "temporary VM ${STAGE_VM} deleted" || warn "could not delete ${STAGE_VM}; check the console"
}
trap cleanup EXIT

# The key waits in the private bucket only until the VM has read it; the VM deletes it at once.
gcs rm -q "${WORK}/**" >/dev/null 2>&1 || true
gcs cp -q "${KAGGLE_KEY}" "${WORK}/kaggle.json"

STARTUP="$(mktemp)"
cat > "${STARTUP}" <<REMOTE
#!/bin/bash
set -uo pipefail
LOG=/var/log/glo-stage.log
exec > >(tee -a "\$LOG") 2>&1
push_log() { gcloud storage cp -q "\$LOG" "${WORK}/log.txt" >/dev/null 2>&1 || true; }
fail() { echo "FAILED: \$*"; echo "FAILED" > /tmp/state; gcloud storage cp -q /tmp/state "${WORK}/state.txt"; push_log; exit 1; }
[[ -f /var/lib/glo-stage-done ]] && exit 0
echo "[\$(date -u +%T)] start"
mkdir -p -m 700 /root/.kaggle
gcloud storage cp -q "${WORK}/kaggle.json" /root/.kaggle/kaggle.json || fail "key not readable"
gcloud storage rm -q "${WORK}/kaggle.json" || true
chmod 600 /root/.kaggle/kaggle.json
mkdir -p /stage/raw /stage/brats2021 && cd /stage/raw
# Credentials go to a private curl config file (not the command line); the download streams to disk.
python3 -c 'import json; k = json.load(open("/root/.kaggle/kaggle.json")); print("user = \"%s:%s\"" % (k["username"], k["key"]))' \
  > /root/.kaggle/curl.cfg || fail "key format"
chmod 600 /root/.kaggle/curl.cfg; shred -u /root/.kaggle/kaggle.json
echo "[\$(date -u +%T)] downloading"; push_log
curl -sfL --retry 5 --retry-delay 10 -K /root/.kaggle/curl.cfg -o download.bin \
  "https://www.kaggle.com/api/v1/datasets/download/${KAGGLE_DATASET}/${KAGGLE_FILE}" &
pid=\$!
while kill -0 \$pid 2>/dev/null; do
  sleep 60; echo "[\$(date -u +%T)] downloading: \$(du -sh download.bin 2>/dev/null | cut -f1)"; push_log
done
wait \$pid; rc=\$?; shred -u /root/.kaggle/curl.cfg; [[ \$rc == 0 ]] || fail "download (curl exit \$rc)"
# Kaggle may serve the file zipped.
if python3 -c 'import sys, zipfile; sys.exit(0 if zipfile.is_zipfile("download.bin") else 1)'; then
  python3 -m zipfile -e download.bin . && rm -f download.bin || fail "unzip"
else
  mv download.bin ${KAGGLE_FILE}
fi
[[ -s ${KAGGLE_FILE} ]] || fail "${KAGGLE_FILE} not in the download"
echo "[\$(date -u +%T)] extracting"; push_log
tar -xf ${KAGGLE_FILE} -C /stage/brats2021 && rm -f ${KAGGLE_FILE} || fail "tar"
cd /stage/brats2021
n=\$(find . -mindepth 1 -maxdepth 1 -type d -name 'BraTS2021_*' | wc -l)
[[ "\$n" == "${EXPECTED_CASES}" ]] || fail "expected ${EXPECTED_CASES} cases, found \$n"
for d in BraTS2021_*; do
  for m in flair t1 t1ce t2 seg; do [[ -s "\$d/\${d}_\${m}.nii.gz" ]] || fail "missing \$d/\${d}_\${m}.nii.gz"; done
done
echo "[\$(date -u +%T)] checksums"; push_log
find . -type f -name '*.nii.gz' | sort | xargs sha256sum > SHA256SUMS
{ echo "dataset=BraTS 2021 Task 1 (adult glioma) training data"
  echo "source=kaggle:${KAGGLE_DATASET}/${KAGGLE_FILE}"
  echo "cases=\$n"
  echo "files=\$(wc -l < SHA256SUMS)"
  echo "bytes=\$(du -sb . | cut -f1)"
  echo "staged_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > /stage/MANIFEST.txt
echo "[\$(date -u +%T)] uploading"; push_log
gcloud storage rsync -r . "${DEST}" --quiet || fail "upload"
gcloud storage cp -q /stage/MANIFEST.txt "${DEST}/MANIFEST.txt" || fail "manifest"
touch /var/lib/glo-stage-done
echo "DONE" > /tmp/state; gcloud storage cp -q /tmp/state "${WORK}/state.txt"
echo "[\$(date -u +%T)] done"; push_log
REMOTE

gcloud compute instances create "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" \
  --machine-type e2-standard-4 --boot-disk-size 100GB --boot-disk-type pd-balanced \
  --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud --scopes storage-rw \
  --metadata-from-file startup-script="${STARTUP}" >/dev/null
rm -f "${STARTUP}"
pass "VM ${STAGE_VM} created; the job runs on it (log: ${WORK}/log.txt)"

# Wait for the job (up to 3 hours), reporting its last step.
state="" last=""
for _ in $(seq 360); do
  sleep 30
  state="$(gcs cat "${WORK}/state.txt" 2>/dev/null || true)"
  step="$(gcs cat "${WORK}/log.txt" 2>/dev/null | grep '^\[' | tail -1 || true)"
  [[ -n "${step}" && "${step}" != "${last}" ]] && { log "VM: ${step}"; last="${step}"; }
  [[ -n "${state}" ]] && break
done
[[ "${state}" == "DONE" ]] || { gcs cat "${WORK}/log.txt" 2>/dev/null | tail -20; die "staging failed (${state:-timeout})"; }
gcs cat "${DEST}/MANIFEST.txt" | grep -q "^cases=${EXPECTED_CASES}$" || die "manifest missing or incomplete at ${DEST}"
gcs rm -q "${WORK}/**" >/dev/null 2>&1 || true
pass "staged ${EXPECTED_CASES} glioma cases at ${DEST}"
