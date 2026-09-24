#!/usr/bin/env bash
# =============================================================================
# GLO-NCA -- GCP cost helper. Two modes, both READ-ONLY (never create/delete):
#
#   (default)  Estimate the cost of the configured GPU VM and, if it is running,
#              its elapsed runtime. Costs are ESTIMATES from a small built-in
#              table (public list prices change -- verify against GCP pricing).
#
#   --audit    List potentially-billing GLO-NCA resources so nothing is left on
#              overnight: the VM (+state), its disks, snapshots, static IPs, and
#              the bucket. Only inspects resources tied to this project/config.
#
# Usage:
#   ./cloud/scripts/cost_report.sh
#   ./cloud/scripts/cost_report.sh --audit
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

# ESTIMATED public on-demand GPU list prices (USD/GPU-hour). VERIFY before use.
gpu_price() {
  case "$1" in
    nvidia-tesla-t4)   echo "0.35 (T4 16GB, ESTIMATED)";;
    nvidia-l4)         echo "0.71 (L4 24GB, ESTIMATED)";;
    nvidia-tesla-v100) echo "2.48 (V100 16GB, ESTIMATED)";;
    nvidia-tesla-a100) echo "3.67 (A100 40GB, ESTIMATED)";;
    nvidia-a100-80gb)  echo "5.07 (A100 80GB, ESTIMATED)";;
    *)                 echo "unknown (verify on GCP pricing)";;
  esac
}

if [[ "${1:-}" == "--audit" ]]; then
  echo "=== GLO-NCA GCP cost audit (read-only) ==="
  echo "-- GPU VM ${VM_NAME} --"
  # shellcheck disable=SC2086
  gcloud compute instances describe "${VM_NAME}" $(vm_flags) \
    --format="table(name,status,machineType.basename(),
      guestAccelerators[0].acceleratorType.basename(),
      guestAccelerators[0].acceleratorCount)" 2>/dev/null \
    || echo "  (VM does not exist -- no GPU compute billing)"
  echo "-- Disks in ${GCP_ZONE} (attached disks bill even when the VM is stopped) --"
  gcloud compute disks list --zones="${GCP_ZONE}" --project="${GCP_PROJECT_ID}" \
    --format="table(name,sizeGb,type.basename(),status)" 2>/dev/null || true
  echo "-- Snapshots --"
  gcloud compute snapshots list --project="${GCP_PROJECT_ID}" \
    --format="table(name,diskSizeGb,storageBytes)" 2>/dev/null || true
  echo "-- Static external IPs (reserved but unused IPs bill) --"
  gcloud compute addresses list --project="${GCP_PROJECT_ID}" \
    --format="table(name,address,status,region.basename())" 2>/dev/null || true
  echo "-- Bucket (durable storage -- intentionally KEPT) --"
  echo "   ${GCS_ROOT}  (dataset: ${GCS_DATA}, results: ${GCS_EXPERIMENTS})"
  echo
  echo "After a campaign, the GPU VM + its disk are the expensive items to DELETE"
  echo "(./cloud/scripts/delete_vm.sh <exp_id>). Keep the bucket."
  exit 0
fi

echo "=== GLO-NCA GPU VM cost estimate (ESTIMATED) ==="
echo "region/zone : ${GCP_REGION:-?} / ${GCP_ZONE}"
echo "machine     : ${MACHINE_TYPE:-?}"
echo "GPU         : ${GPU_TYPE:-?} x ${GPU_COUNT:-1}"
echo "GPU price   : \$$(gpu_price "${GPU_TYPE:-}") /GPU-hour (+ machine + disk, extra)"
# shellcheck disable=SC2086
if gcloud compute instances describe "${VM_NAME}" $(vm_flags) >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  state=$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) --format="value(status)")
  # shellcheck disable=SC2086
  started=$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) \
            --format="value(lastStartTimestamp)" 2>/dev/null || echo "")
  echo "VM state    : ${state}"
  [[ -n "${started}" ]] && echo "last started: ${started}"
  if [[ "${state}" == "RUNNING" ]]; then
    warn "VM is RUNNING -> GPU billing is active. Stop or delete when idle."
  fi
else
  echo "VM state    : (does not exist -- no GPU billing)"
fi
echo
echo "NOTE: prices are ESTIMATES; confirm on https://cloud.google.com/compute/gpus-pricing"
echo "Record per experiment: GPU, count, machine, region, start, end, runtime, est. cost."
