#!/usr/bin/env bash
# systemd entrypoint: runs training in Docker with GCS sync, then a final sync and optional VM stop.
set -euo pipefail
# shellcheck disable=SC1091
source /etc/glo-nca-cascade.env

OUT="${GLO_OUT}"; RUN_ID="${GLO_RUN_ID}"; RUN_DIR="${OUT}/${RUN_ID}"
GCS_DEST="gs://${GLO_BUCKET}/${GLO_EXP_PREFIX}/${RUN_ID}"
INTERVAL="${GLO_SYNC_INTERVAL:-300}"
read -r -a EXTRA_ARGS <<< "${GLO_EXTRA_ARGS:-}"
gcs() { if gcloud storage --help >/dev/null 2>&1; then gcloud storage "$@"; else gsutil "$@"; fi; }
sync_now() { gcs rsync -r -x '.*\.tmp$' "${RUN_DIR}" "${GCS_DEST}" >/dev/null 2>&1; }

echo "[entrypoint] run ${RUN_ID} mode=${GLO_MODE} extra=${GLO_EXTRA_ARGS:-none}"

# --- background GCS sync (a preemption loses at most one interval) -----------
( while true; do sleep "${INTERVAL}"; [[ -d "${RUN_DIR}" ]] || continue
    sync_now && echo "[sync] pushed ${RUN_ID}" || echo "[sync] WARNING: sync failed; will retry"
  done ) &
SYNC_PID=$!
trap 'kill "${SYNC_PID}" 2>/dev/null || true' EXIT

# Training in Docker; --shm-size=8g because DataLoader workers pass volumes through /dev/shm.
COMMON=(--rm --gpus all --shm-size=8g -v "${GLO_DATA}:${GLO_DATA}:ro" -v "${OUT}:/out"
        -e MPLBACKEND=Agg glo-nca-cascade:latest)
set +e
if [[ "${GLO_MODE}" == "resume" ]]; then
  docker run "${COMMON[@]}" --resume "/out/${RUN_ID}" --data-root "${GLO_DATA}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
  docker run "${COMMON[@]}" --config configs/glo_nca_cascade.yaml --data-root "${GLO_DATA}" \
    --output /out --experiment-id "${RUN_ID}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
RC=$?
set -e
kill "${SYNC_PID}" 2>/dev/null || true

# --- final sync, attempted on every exit path --------------------------------
if [[ -d "${RUN_DIR}" ]]; then
  sync_now && echo "[entrypoint] final GCS sync ok -> ${GCS_DEST}" \
    || echo "[entrypoint] WARNING: final GCS sync failed; data kept at ${RUN_DIR}"
fi
rm -f /tmp/glo-nca-cascade-training.lock
echo "[entrypoint] training finished rc=${RC}"

# Stop the VM when the run ends (completed, paused or failed) to halt GPU billing.
if [[ "${GLO_AUTO_STOP:-0}" == "1" ]]; then
  echo "[entrypoint] auto-stop: shutting the VM down"
  shutdown -h +1 || true
fi
exit "${RC}"
