#!/usr/bin/env bash
# Upload the local BraTS dataset to GCS -- but ONLY after the Phase 1 validator
# passes. Never uploads an invalid dataset. Uses rsync so re-runs are cheap
# (no blind re-upload of unchanged files).
#
# Usage: ./cloud/scripts/upload_dataset.sh /path/to/BraTS
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

LOCAL_DATA="${1:-}"
[[ -n "${LOCAL_DATA}" ]] || die "usage: upload_dataset.sh /path/to/BraTS"
[[ -d "${LOCAL_DATA}" ]] || die "dataset not found: ${LOCAL_DATA}"

log "validating dataset BEFORE upload (Phase 1 validator)..."
PYBIN="$(command -v python3 || command -v python)"
if "${PYBIN}" "${REPO_DIR}/scripts/validate_dataset.py" --root "${LOCAL_DATA}"; then
  pass "dataset validation PASS"
else
  die "dataset validation FAILED -- refusing to upload invalid data."
fi

log "uploading ${LOCAL_DATA} -> ${GCS_DATA} (rsync, structure preserved)"
gcs_rsync -r "${LOCAL_DATA}" "${GCS_DATA}"

log "verifying upload..."
n_local=$(find "${LOCAL_DATA}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
n_remote=$(gcs ls "${GCS_DATA}/" 2>/dev/null | grep -c '/$' || echo 0)
log "local patient folders: ${n_local} | remote prefixes: ${n_remote}"
if [[ "${n_remote}" -ge "${n_local}" && "${n_local}" -gt 0 ]]; then
  pass "dataset uploaded to ${GCS_DATA}"
else
  warn "remote count (${n_remote}) < local (${n_local}); re-run to complete."
fi
