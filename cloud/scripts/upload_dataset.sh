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

log "uploading ${LOCAL_DATA} -> ${GCS_DATA} (recursive rsync, nested cohorts + "
log "  'UCSD - Training/' structure preserved; never flattened/renamed)"
gcs_rsync -r "${LOCAL_DATA}" "${GCS_DATA}"

log "verifying upload (by discovered VALID cases, recursively -- not just top-level)"
# Local: count valid case dirs the loader will actually see (recursive, files-
# validated), so the count matches the 1296-case nested layout.
n_local=$(REPO_DIR="${REPO_DIR}" "${PYBIN}" - "${LOCAL_DATA}" <<'PY'
import sys, os; sys.path.insert(0, os.environ.get("REPO_DIR","."))
from src.experiment.datasource import discover_case_ids
print(len(discover_case_ids(sys.argv[1])))
PY
)
# Remote: count uploaded seg files (one per case) recursively.
n_remote=$(gcs ls -r "${GCS_DATA}/**" 2>/dev/null | grep -c 'seg.nii' || echo 0)
log "local valid cases: ${n_local} | remote seg files: ${n_remote}"
if [[ "${n_remote}" -ge "${n_local}" && "${n_local}" -gt 0 ]]; then
  pass "dataset uploaded to ${GCS_DATA} (${n_local} cases)"
else
  warn "remote cases (${n_remote}) < local (${n_local}); re-run to complete."
fi
