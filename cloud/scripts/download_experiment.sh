#!/usr/bin/env bash
# Download a complete experiment directory from GCS to the local machine,
# preserving the exact Phase 1 structure. Read-only against GCS.
#
# Usage:
#   ./cloud/scripts/download_experiment.sh GLO-NCA-V2-YYYYMMDD-HHMMSS [dest_base]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

EXP_ID="${1:-}"
[[ -n "${EXP_ID}" ]] || die "usage: download_experiment.sh <experiment_id> [dest_base]"
DEST_BASE="${2:-${REPO_DIR}/experiments}"
SRC="${GCS_EXPERIMENTS}/${EXP_ID}"
DEST="${DEST_BASE}/${EXP_ID}"

gcs_exists "${SRC}" || die "experiment not found in GCS: ${SRC}"
mkdir -p "${DEST}"
log "downloading ${SRC} -> ${DEST}"
gcs_rsync -r "${SRC}" "${DEST}"

# Report what arrived so the user can confirm a complete package.
log "downloaded contents:"
for f in checkpoints/best.pth checkpoints/last.pth reports/results.json \
         reports/thesis_results.csv experiment_manifest.json status.json; do
  [[ -e "${DEST}/${f}" ]] && echo "  ok  ${f}" || echo "  --  ${f} (absent)"
done
pass "experiment downloaded to ${DEST}"
