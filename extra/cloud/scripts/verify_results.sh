#!/usr/bin/env bash
# =============================================================================
# GLO-NCA -- verify a finished experiment is SAFELY stored in GCS before any
# expensive resource is deleted. This is the "VERIFY" in COMPUTE -> SAVE ->
# VERIFY -> DELETE. It NEVER deletes anything and NEVER starts compute.
#
# Checks, for one experiment id:
#   * the experiment prefix exists in GCS;
#   * required artifacts exist AND are non-empty in GCS
#       (manifest, status, best.pth, last.pth, results.json, metrics CSVs,
#        graphs, config);
#   * status.json state == "completed";
#   * downloads the best + last checkpoints and validates they RELOAD with the
#     required resume state (reuses validate_checkpoint.py);
# Exit 0 only if everything passes -> safe to delete the GPU VM.
#
# Usage (on the VM or any machine with gcloud + repo):
#   ./extra/cloud/scripts/verify_results.sh <experiment_id> [--keep-download]
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../cloud/scripts" && pwd)/lib.sh"
load_config
require_gcloud_auth

EXP_ID="${1:-}"
[[ -n "${EXP_ID}" ]] || die "usage: verify_results.sh <experiment_id> [--keep-download]"
KEEP="${2:-}"
SRC="${GCS_EXPERIMENTS}/${EXP_ID}"
PYBIN="$(command -v python3 || command -v python)"
FAILS=0
mark() { if [ "$1" -eq 0 ]; then pass "$2"; else fail "$2"; FAILS=$((FAILS+1)); fi; }

log "verifying GCS artifacts for ${EXP_ID}"
gcs_exists "${SRC}"; mark $? "experiment prefix exists in GCS (${SRC})"

# Required artifacts must exist AND be non-zero in GCS. `gcs ls -l` prints a size
# column; we assert the size is > 0. Checkpoints validated by reload below.
REQUIRED=(
  "experiment_manifest.json"
  "status.json"
  "checkpoints/best.pth"
  "checkpoints/last.pth"
  "reports/results.json"
  "reports/thesis_results.csv"
  "config/config.yaml"
)
for rel in "${REQUIRED[@]}"; do
  line="$(gcs ls -l "${SRC}/${rel}" 2>/dev/null | head -1 || true)"
  size="$(echo "${line}" | awk '{print $1}' | grep -E '^[0-9]+$' || echo 0)"
  if [ -n "${line}" ] && [ "${size:-0}" -gt 0 ] 2>/dev/null; then
    pass "GCS artifact non-empty: ${rel} (${size} bytes)"
  else
    fail "GCS artifact missing/empty: ${rel}"; FAILS=$((FAILS+1))
  fi
done

# At least one metrics CSV and one graph should be present.
gcs ls "${SRC}/metrics/" >/dev/null 2>&1; mark $? "metrics/ present in GCS"
gcs ls "${SRC}/graphs/"  >/dev/null 2>&1; mark $? "graphs/ present in GCS"

# status must be completed -- UNLESS we are verifying a failed run purely to
# confirm its artifacts are durably in GCS before releasing the GPU.
#
# Phase 2 (P1, cost safety): previously this check hard-required "completed", and
# delete_vm.sh refuses to delete when verification fails. A CRASHED run could
# therefore never be cleaned up through the normal path, so the GPU VM kept
# billing until someone remembered the undocumented --force. ALLOW_FAILED makes
# the legitimate case explicit and still verifies that everything reachable has
# been preserved first.
state="$(gcs cat "${SRC}/status.json" 2>/dev/null | grep -o '"state":[^,]*' || true)"
if [[ "${ALLOW_FAILED:-0}" == "1" ]]; then
  if echo "${state}" | grep -qiE 'completed|failed'; then
    pass "status.json state is terminal (${state}) [ALLOW_FAILED=1]"
  else
    fail "status.json state is not terminal (${state}) -- run may still be active"
    FAILS=$((FAILS+1))
  fi
else
  set +e
  echo "${state}" | grep -qi 'completed'
  STATE_RC=$?
  set -e
  mark "${STATE_RC}" "status.json state == completed (${state})"
  if [[ "${STATE_RC}" -ne 0 ]]; then
    log "NOTE: to release the GPU after a FAILED run (artifacts still verified),"
    log "      re-run with: ALLOW_FAILED=1 ${BASH_SOURCE[0]##*/} ${EXP_ID}"
  fi
fi

# Download best + last and validate they reload with resume state.
TMP="$(mktemp -d)"
log "downloading checkpoints to ${TMP} for reload validation"
mkdir -p "${TMP}/${EXP_ID}/checkpoints"
gcs_cp "${SRC}/checkpoints/best.pth" "${TMP}/${EXP_ID}/checkpoints/best.pth" 2>/dev/null || true
gcs_cp "${SRC}/checkpoints/last.pth" "${TMP}/${EXP_ID}/checkpoints/last.pth" 2>/dev/null || true
gcs_cp "${SRC}/experiment_manifest.json" "${TMP}/${EXP_ID}/experiment_manifest.json" 2>/dev/null || true
# Absolute path: this script is invoked by delete_vm.sh from an arbitrary CWD,
# where the old repo-relative path silently failed and blocked VM cleanup.
"${PYBIN}" "${CLOUD_DIR}/scripts/validate_checkpoint.py" "${TMP}/${EXP_ID}"
mark $? "downloaded checkpoint reloads with resume state"

if [ "${KEEP}" != "--keep-download" ]; then rm -rf "${TMP}"; else log "kept ${TMP}"; fi

echo; echo "========================================"
if [ "${FAILS}" -eq 0 ]; then
  echo "RESULTS VERIFIED: PASS"
  echo "Safe to delete the GPU compute for ${EXP_ID} (results are durable in GCS)."
  echo "  -> ./extra/cloud/scripts/delete_vm.sh"
  echo "========================================"
  exit 0
else
  echo "RESULTS VERIFICATION: FAIL (${FAILS} issue(s))"
  echo "DO NOT delete compute. Re-sync and re-verify:"
  echo "  -> ./cloud/scripts/sync_experiment.sh <local_experiment_dir>"
  echo "========================================"
  exit 1
fi
