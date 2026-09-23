#!/usr/bin/env bash
# =============================================================================
# GLO-NCA -- DELETE the GPU VM (and its boot disk) to STOP ALL COMPUTE + DISK
# billing after results are safely stored. This is the final cost-cleanup step.
#
# Safety: refuses to delete unless results have been verified in GCS, UNLESS you
# pass --force (e.g. deleting a VM from a failed/aborted run with nothing to
# keep). It ONLY touches the VM named in gcp.env -- never other resources or
# other projects. The GCS bucket, dataset, master split and experiment results
# are NOT touched (durable storage is intentionally retained).
#
# Order enforced:  COMPUTE -> SAVE -> VERIFY -> DELETE
#
# Usage:
#   ./extra/cloud/scripts/verify_results.sh <experiment_id>   # must PASS first
#   ./extra/cloud/scripts/delete_vm.sh <experiment_id>        # verifies, then deletes
#   ./extra/cloud/scripts/delete_vm.sh --force                # delete without verify
# =============================================================================
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../cloud/scripts" && pwd)/lib.sh"
load_config
require_gcloud_auth

ARG="${1:-}"
if [[ "${ARG}" == "--force" ]]; then
  warn "FORCE delete requested -- skipping results verification."
  warn "Only do this for a run whose results you do NOT need to keep."
else
  EXP_ID="${ARG}"
  [[ -n "${EXP_ID}" ]] || die "usage: delete_vm.sh <experiment_id>   (or --force)
       Verify results are safe in GCS first; this deletes the GPU VM + disk."
  log "verifying results for ${EXP_ID} BEFORE deleting compute..."
  if ! bash "$(dirname "${BASH_SOURCE[0]}")/verify_results.sh" "${EXP_ID}"; then
    die "results verification FAILED -- refusing to delete the VM.
         Re-sync (sync_experiment.sh) and re-verify before cleanup."
  fi
  pass "results verified in GCS -- safe to delete compute."
fi

# shellcheck disable=SC2086
gcloud compute instances describe "${VM_NAME}" $(vm_flags) >/dev/null 2>&1 \
  || { pass "VM ${VM_NAME} does not exist -- nothing to delete."; exit 0; }

warn "About to DELETE VM ${VM_NAME} and its boot disk in ${GCP_ZONE}."
warn "This stops ALL compute + attached-disk billing for it. Irreversible."
warn "GCS bucket, dataset, master split and experiment results are KEPT."
confirm "Delete GPU VM ${VM_NAME} now?"
# --delete-disks=all removes the boot disk too (no lingering disk billing).
# shellcheck disable=SC2086
gcloud compute instances delete "${VM_NAME}" $(vm_flags) --delete-disks=all --quiet
pass "VM ${VM_NAME} + boot disk DELETED -- GPU/disk billing for it has stopped."
echo
log "Retained (durable, low-cost): ${GCS_DATA} (dataset), ${GCS_EXPERIMENTS} (results),"
log "  split/master_split.json (in the repo). Nothing else should be billing GPU now."
log "Sanity-check for stray costly resources with: ./extra/cloud/scripts/cost_report.sh --audit"
