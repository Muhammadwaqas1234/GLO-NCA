#!/usr/bin/env bash
# Boot hook: resume the keep-alive run after a restart (e.g. a Spot preemption); no-op otherwise.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
[[ -f "${KEEPALIVE_FILE}" ]] || { log "keep-alive off; nothing to resume"; exit 0; }
# shellcheck disable=SC1090
source "${KEEPALIVE_FILE}"
RUN_DIR="${VM_OUT_DIR}/${KEEP_RUN_ID}"
TRIES_FILE="${KEEPALIVE_FILE%.env}.tries"

read -r STATE EPOCH <<< "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1]));
print(d.get('state', 'none'), d.get('epoch', 0))" "${RUN_DIR}/status.json" 2>/dev/null || echo "none 0")"
if [[ "${STATE}" == "completed" ]] \
   || [[ "${STATE}" == "paused" && -n "${KEEP_STOP_AFTER}" && "${EPOCH}" -ge "${KEEP_STOP_AFTER}" ]]; then
  log "run ${KEEP_RUN_ID} is done (${STATE} at epoch ${EPOCH}); keep-alive off"
  rm -f "${KEEPALIVE_FILE}" "${TRIES_FILE}"; exit 0
fi

# Count only runs that failed on their own (the entrypoint records the exit code; a preemption
# records nothing); give up after three failures at the same epoch so a broken run cannot loop.
RC_FILE="$(dirname "${KEEPALIVE_FILE}")/last_rc"
LAST_RC=$(cat "${RC_FILE}" 2>/dev/null || echo none); rm -f "${RC_FILE}"
read -r PREV_EPOCH FAILS <<< "$(cat "${TRIES_FILE}" 2>/dev/null || echo "-1 0")"
[[ "${PREV_EPOCH}" == "${EPOCH}" ]] || FAILS=0
[[ "${LAST_RC}" != "none" && "${LAST_RC}" != "0" ]] && FAILS=$((FAILS + 1))
echo "${EPOCH} ${FAILS}" > "${TRIES_FILE}"
if (( FAILS >= 3 )); then
  fail "run failed 3 times at epoch ${EPOCH} (last rc ${LAST_RC}); keep-alive off, VM stopping"
  rm -f "${KEEPALIVE_FILE}"; shutdown -h +1; exit 1
fi

# The GPU driver can take a moment after boot.
for _ in $(seq 1 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done

STOP=(); [[ -n "${KEEP_STOP_AFTER}" ]] && STOP=(--stop-after-epoch "${KEEP_STOP_AFTER}")
export ASSUME_YES=1
cd "${GLO_WORKSPACE}" || die "no checkout at ${GLO_WORKSPACE}"
if [[ -f "${RUN_DIR}/last.pth" ]]; then
  log "resuming ${KEEP_RUN_ID} after epoch ${EPOCH} (failures so far ${FAILS})"
  bash cloud/scripts/resume_training.sh "${KEEP_RUN_ID}" --data-root "${KEEP_DATA_ROOT}" \
    "${STOP[@]}" --auto-stop --keep-alive
else
  # Stopped before the first checkpoint: start the same config again.
  [[ -n "${KEEP_CONFIG}" ]] || die "no checkpoint and no config recorded for ${KEEP_RUN_ID}"
  log "restarting ${KEEP_CONFIG} (no checkpoint yet)"
  rm -rf "${RUN_DIR}"
  bash cloud/scripts/run_training.sh --config "${KEEP_CONFIG}" --data-root "${KEEP_DATA_ROOT}" \
    "${STOP[@]}" --auto-stop --keep-alive
fi
