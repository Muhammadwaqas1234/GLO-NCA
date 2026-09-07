#!/usr/bin/env bash
# Ensure the dataset is present and VALID on the VM's local disk (the training
# cache), synced from the durable GCS master copy.
#
# Design: GCS = durable master copy; VM local disk = training cache. GLO-NCA does
# heavy random-patch access to many small NIfTI files, so a local SSD cache is
# far faster and more reliable than a network filesystem (gcsfuse). The dataset
# stays recoverable from GCS at all times.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

log "syncing dataset ${GCS_DATA} -> ${VM_DATA_DIR} (local training cache)"
mkdir -p "${VM_DATA_DIR}"
gcs_rsync -r "${GCS_DATA}" "${VM_DATA_DIR}"

log "validating local dataset cache (Phase 1 validator)..."
PYBIN="$(command -v python3 || command -v python)"
if "${PYBIN}" "${REPO_DIR}/scripts/validate_dataset.py" --root "${VM_DATA_DIR}"; then
  pass "local dataset cache valid at ${VM_DATA_DIR}"
else
  die "local dataset cache FAILED validation -- refusing to start GPU training."
fi
