#!/usr/bin/env bash
# Start the GPU VM, creating it on first use. Idempotent and non-destructive.
# Attaches the VM's default service account with cloud-platform scope so it can
# reach GCS via Application Default Credentials (no JSON keys).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

# shellcheck disable=SC2086
if gcloud compute instances describe "${VM_NAME}" $(vm_flags) >/dev/null 2>&1; then
  state=$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) \
          --format="value(status)")
  if [[ "${state}" == "RUNNING" ]]; then
    pass "VM ${VM_NAME} already RUNNING"
  else
    log "starting existing VM ${VM_NAME} (was ${state})"
    gcloud compute instances start "${VM_NAME}" $(vm_flags)
    pass "VM ${VM_NAME} started"
  fi
else
  log "VM ${VM_NAME} does not exist -- creating it"
  log "  machine=${MACHINE_TYPE} gpu=${GPU_TYPE}x${GPU_COUNT} disk=${DISK_SIZE_GB}GB"
  log "  image=${IMAGE_FAMILY}/${IMAGE_PROJECT} zone=${GCP_ZONE}"
  confirm "Creating a GPU VM starts billing while it is RUNNING. Continue?"
  # shellcheck disable=SC2086
  gcloud compute instances create "${VM_NAME}" $(vm_flags) \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${DISK_SIZE_GB}GB" \
    --boot-disk-type=pd-ssd \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True"
  pass "VM ${VM_NAME} created and starting"
fi

echo
log "reminder: a RUNNING GPU VM costs money. Stop it when idle:"
log "  ./cloud/scripts/stop_vm.sh"
