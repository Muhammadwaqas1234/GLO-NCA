#!/usr/bin/env bash
# Show VM state, GCS experiment list, and cost reminder. Read-only.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

echo "=== VM ==="
# shellcheck disable=SC2086
if gcloud compute instances describe "${VM_NAME}" $(vm_flags) >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  gcloud compute instances describe "${VM_NAME}" $(vm_flags) \
    --format="table(name, status, machineType.basename(), zone.basename())"
  # shellcheck disable=SC2086
  st=$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) --format="value(status)")
  if [[ "${st}" == "RUNNING" ]]; then
    warn "VM is RUNNING -> incurring compute cost. Stop when idle."
  else
    pass "VM is ${st} -> no compute cost."
  fi
else
  echo "VM ${VM_NAME}: does not exist"
fi

echo
echo "=== GCS experiments (${GCS_EXPERIMENTS}) ==="
gcs ls "${GCS_EXPERIMENTS}/" 2>/dev/null || echo "  (none yet)"

echo
echo "=== GCS dataset (${GCS_DATA}) ==="
gcs ls "${GCS_DATA}/" 2>/dev/null | head || echo "  (not uploaded yet)"
