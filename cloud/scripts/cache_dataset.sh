#!/usr/bin/env bash
# Ensure a valid dataset on the VM's local disk, synced from the GCS master copy.
# Local disk is much faster than gcsfuse for many small NIfTI reads.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

log "syncing dataset ${GCS_DATA} -> ${VM_DATA_DIR} (local training cache)"
mkdir -p "${VM_DATA_DIR}"
gcs_rsync -r "${GCS_DATA}" "${VM_DATA_DIR}"

log "validating local dataset cache (runs in the training image; the host has no nibabel)..."
if repo_python --ro "${VM_DATA_DIR}" -- scripts/validate_dataset.py --root "${VM_DATA_DIR}"; then
  pass "local dataset cache valid at ${VM_DATA_DIR}"
else
  die "local dataset cache FAILED validation -- refusing to start GPU training."
fi
