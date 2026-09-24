#!/usr/bin/env bash
# Sync one experiment directory to GCS; safe to repeat. Checkpoints are written
# as *.tmp then renamed, and *.tmp is excluded, so the remote copy is always
# resume-usable.
#
# Usage:
#   ./cloud/scripts/sync_experiment.sh <experiment_dir>
#   ./cloud/scripts/sync_experiment.sh <experiment_dir> --watch   # loop forever
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

EXP_DIR="${1:-}"
[[ -n "${EXP_DIR}" && -d "${EXP_DIR}" ]] || die "usage: sync_experiment.sh <experiment_dir> [--watch]"
EXP_ID="$(basename "${EXP_DIR%/}")"
DEST="${GCS_EXPERIMENTS}/${EXP_ID}"

do_sync() {
  # Local -> GCS, excluding *.tmp; never deletes remote files.
  gcs_rsync -r -x '.*\.tmp$' "${EXP_DIR}" "${DEST}" \
    && log "synced ${EXP_ID} -> ${DEST}" \
    || warn "sync failed (will retry next cycle); local data untouched."
}

if [[ "${2:-}" == "--watch" ]]; then
  log "watching ${EXP_ID}, syncing every ${SYNC_INTERVAL_SECONDS}s (Ctrl-C to stop)"
  while true; do
    do_sync
    # stop watching once the experiment is finished, after a final sync.
    state=$(grep -o '"state":[^,]*' "${EXP_DIR}/status.json" 2>/dev/null || echo "")
    if echo "${state}" | grep -qiE 'completed|failed'; then
      log "experiment ${state}; final sync done, stopping watch."
      break
    fi
    sleep "${SYNC_INTERVAL_SECONDS}"
  done
else
  do_sync
fi
