#!/usr/bin/env bash
# Stop (NOT delete) the GPU VM to halt compute billing. The VM, its disk and the
# bucket are all preserved. This never deletes anything.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

# shellcheck disable=SC2086
gcloud compute instances describe "${VM_NAME}" $(vm_flags) >/dev/null 2>&1 \
  || die "VM ${VM_NAME} does not exist."

# shellcheck disable=SC2086
state=$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) --format="value(status)")
if [[ "${state}" == "TERMINATED" || "${state}" == "STOPPED" ]]; then
  pass "VM ${VM_NAME} already stopped (${state})"
  exit 0
fi

warn "Stopping VM ${VM_NAME}. In-VM training that is NOT persisted (systemd/tmux)
     will be interrupted. Ensure a recent GCS sync first (sync_experiment.sh)."
confirm "Stop VM ${VM_NAME} now?"
# shellcheck disable=SC2086
gcloud compute instances stop "${VM_NAME}" $(vm_flags)
pass "VM ${VM_NAME} stopped -- compute billing halted (disk storage still billed)."
