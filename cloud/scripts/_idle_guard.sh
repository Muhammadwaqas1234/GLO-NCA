#!/usr/bin/env bash
# Boot guard for the Spot restarter: after the resume hooks ran, a VM with no keep-alive run left and no
# training running pauses the restarter and powers off, so a finished study does not keep billing.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

sleep 120                                   # let the resume hooks start their runs
compgen -G "/var/lib/glo-nca-cascade/keepalive*.env" >/dev/null && { log "keep-alive runs present"; exit 0; }
systemctl list-units --type=service --state=active --no-legend 'glo-nca-cascade-training*' | grep -q . \
  && { log "training running"; exit 0; }
GCP_REGION="${GCP_REGION:-${GCP_ZONE%-*}}"
gcloud scheduler jobs pause "${RESTARTER_JOB:-${VM_NAME}-restarter}" --location "${GCP_REGION}" \
  --project "${GCP_PROJECT_ID}" >/dev/null 2>&1 && log "restarter paused" || warn "could not pause the restarter"
log "nothing to resume; powering off"
shutdown -h +1
