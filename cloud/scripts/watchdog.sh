#!/usr/bin/env bash
# Restart the VM after a Spot preemption: watchdog.sh [--interval SECONDS]  (runs on any machine with gcloud)
# Only a preemption triggers a restart; a stop by the run itself or by you ends the watchdog.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

INTERVAL=300
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) [[ "${2:-}" =~ ^[0-9]+$ ]] || die "--interval needs seconds"; INTERVAL="$2"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

last_stop() {  # the most recent event that stopped the VM
  gcloud compute operations list --project="${GCP_PROJECT_ID}" \
    --filter="targetLink~/instances/${VM_NAME}\$ AND zone:${GCP_ZONE} AND operationType~(preempted|stop|guestTerminate)" \
    --sort-by=~insertTime --limit=1 --format="value(operationType)" 2>/dev/null
}

log "watching ${VM_NAME} (${GCP_ZONE}) every ${INTERVAL}s; Ctrl+C to stop"
while true; do
  # shellcheck disable=SC2086
  status=$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) --format="value(status)" 2>/dev/null)
  if [[ "${status}" == "TERMINATED" ]]; then
    reason=$(last_stop)
    if [[ "${reason}" == "compute.instances.preempted" ]]; then
      log "preempted; restarting ${VM_NAME}"
      # shellcheck disable=SC2086
      if gcloud compute instances start "${VM_NAME}" $(vm_flags) >/dev/null 2>&1; then
        pass "${VM_NAME} restarted; the boot hook resumes the run"
      else
        warn "start failed (likely no Spot capacity); retrying in ${INTERVAL}s"
      fi
    else
      pass "${VM_NAME} stopped normally (${reason:-unknown}); watchdog done"
      exit 0
    fi
  fi
  sleep "${INTERVAL}"
done
