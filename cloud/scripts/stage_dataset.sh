#!/usr/bin/env bash
# Stage BraTS 2021 glioma training data (Kaggle) to GCS through a temporary CPU VM: stage_dataset.sh [--keep-vm]
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
  [[ "${KEEP_VM}" == "1" ]] && { warn "keeping ${STAGE_VM}; delete it yourself when done"; return; }
  gcloud compute instances delete "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" --quiet \
    >/dev/null 2>&1 && pass "temporary VM ${STAGE_VM} deleted" || warn "could not delete ${STAGE_VM}; check the console"
}
trap cleanup EXIT

gcloud compute instances create "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" \
  --machine-type e2-standard-4 --boot-disk-size 100GB --boot-disk-type pd-balanced \
  --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud --scopes storage-rw >/dev/null
for _ in $(seq 30); do
  gcloud compute ssh "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" --command true \
    >/dev/null 2>&1 && break
  sleep 10
done

# The key travels by scp into a private directory and is shredded on the VM after the download.
gcloud compute ssh "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" \
  --command 'mkdir -p -m 700 ~/.kaggle' >/dev/null
gcloud compute scp "${KAGGLE_KEY}" "${STAGE_VM}:~/.kaggle/kaggle.json" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" >/dev/null

gcloud compute ssh "${STAGE_VM}" --project "${GCP_PROJECT_ID}" --zone "${GCP_ZONE}" --command "bash -s" <<REMOTE
set -euo pipefail
chmod 600 ~/.kaggle/kaggle.json
trap 'shred -u ~/.kaggle/kaggle.json 2>/dev/null || rm -f ~/.kaggle/kaggle.json' EXIT
sudo apt-get -qq update >/dev/null && sudo apt-get -qq install -y python3-pip >/dev/null
python3 -m pip install -q --user kaggle
mkdir -p ~/stage/raw ~/stage/brats2021 && cd ~/stage/raw
python3 -m kaggle datasets download -d ${KAGGLE_DATASET} -f ${KAGGLE_FILE} -p . --quiet
shred -u ~/.kaggle/kaggle.json
for z in *.zip; do [[ -e "\$z" ]] && python3 -m zipfile -e "\$z" . && rm -f "\$z"; done
tar -xf ${KAGGLE_FILE} -C ~/stage/brats2021 && rm -f ${KAGGLE_FILE}
cd ~/stage/brats2021
# Case folders must sit directly under the dataset root.
n=\$(find . -mindepth 1 -maxdepth 1 -type d -name 'BraTS2021_*' | wc -l)
[[ "\$n" == "${EXPECTED_CASES}" ]] || { echo "expected ${EXPECTED_CASES} cases, found \$n" >&2; exit 3; }
bad=0
for d in BraTS2021_*; do
  for m in flair t1 t1ce t2 seg; do [[ -s "\$d/\${d}_\${m}.nii.gz" ]] || { echo "missing \$d/\${d}_\${m}.nii.gz" >&2; bad=1; }; done
done
[[ "\$bad" == 0 ]] || exit 4
find . -type f -name '*.nii.gz' | sort | xargs sha256sum > SHA256SUMS
{ echo "dataset=BraTS 2021 Task 1 (adult glioma) training data"
  echo "source=kaggle:${KAGGLE_DATASET}/${KAGGLE_FILE}"
  echo "cases=\$n"
  echo "files=\$(wc -l < SHA256SUMS)"
  echo "bytes=\$(du -sb . | cut -f1)"
  echo "staged_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > MANIFEST.txt
gcloud storage rsync -r . "${DEST}" --quiet
gcloud storage cp MANIFEST.txt "${DEST}/MANIFEST.txt" --quiet
cat MANIFEST.txt
REMOTE

gcs cat "${DEST}/MANIFEST.txt" | grep -q "^cases=${EXPECTED_CASES}$" || die "staging did not complete: ${DEST}"
pass "staged ${EXPECTED_CASES} glioma cases at ${DEST}"
