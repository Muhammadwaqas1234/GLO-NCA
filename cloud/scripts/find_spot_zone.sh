#!/usr/bin/env bash
# Find a zone where an L4 Spot VM can actually be created.
#
# READ-ONLY by default: it lists which zones offer the GPU and reports the
# configured zone first. Capacity is not queryable through any GCP API -- the
# only definitive test is attempting creation -- so --probe optionally creates
# a VM and deletes it immediately.
#
# Usage:
#   ./cloud/scripts/find_spot_zone.sh              # list candidate zones
#   ./cloud/scripts/find_spot_zone.sh --probe      # also attempt a real create
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

GPU="${GPU_TYPE:-nvidia-l4}"
MACHINE="${MACHINE_TYPE:-g2-standard-8}"
PROBE=0
[ "${1:-}" = "--probe" ] && PROBE=1

log "GPU ${GPU} / machine ${MACHINE}"
log "configured zone: ${GCP_ZONE:-<unset>}"
echo

log "zones offering ${GPU}:"
mapfile -t ZONES < <(
  gcloud compute accelerator-types list \
    --filter="name=${GPU}" \
    --format="value(zone)" \
    --project="${GCP_PROJECT_ID}" 2>/dev/null | sort -u
)

if [ "${#ZONES[@]}" -eq 0 ]; then
  fail "no zones report ${GPU}. Check the GPU name and project quota."
  exit 1
fi

for z in "${ZONES[@]}"; do
  mark=" "
  [ "${z}" = "${GCP_ZONE:-}" ] && mark="*"
  printf "  %s %s\n" "${mark}" "${z}"
done
echo
log "* = the zone currently configured in cloud/config/gcp.env"
log "This list means the GPU is OFFERED there. It does NOT prove Spot capacity"
log "is available right now: GCP exposes no capacity API. The only definitive"
log "test is attempting to create the instance."
echo

if [ "${PROBE}" -eq 0 ]; then
  log "re-run with --probe to attempt a real (immediately deleted) create"
  exit 0
fi

PROBE_NAME="glo-nca-spot-probe-$$"
log "probing Spot capacity by creating and deleting ${PROBE_NAME}"
log "a successful probe bills a few seconds of Spot GPU time"
confirm "Attempt a real Spot create/delete probe?"

for z in "${ZONES[@]}"; do
  printf "  %-22s " "${z}"
  if gcloud compute instances create "${PROBE_NAME}" \
        --zone="${z}" --project="${GCP_PROJECT_ID}" \
        --machine-type="${MACHINE}" \
        --accelerator="type=${GPU},count=1" \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size=50GB \
        --no-restart-on-failure \
        >/dev/null 2>&1; then
    echo "SPOT AVAILABLE"
    gcloud compute instances delete "${PROBE_NAME}" \
      --zone="${z}" --project="${GCP_PROJECT_ID}" --quiet >/dev/null 2>&1 || true
    pass "use GCP_ZONE=${z}"
    exit 0
  fi
  echo "no capacity / not permitted"
done

fail "no probed zone had L4 Spot capacity. Wait and retry, or use STANDARD."
exit 1
