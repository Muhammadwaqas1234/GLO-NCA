#!/usr/bin/env bash
# Run by the systemd unit: reads /etc/glo-nca.env, runs Docker training with a
# background GCS sync, and exits with the container's code (no restart loop).
set -euo pipefail
# shellcheck disable=SC1091
source /etc/glo-nca.env

REPO="${GLO_REPO}"; DATA="${GLO_DATA}"; OUT="${GLO_OUT}"; CONFIG="${GLO_CONFIG}"
BUCKET="${GLO_BUCKET}"; EXP_PREFIX="${GLO_EXP_PREFIX}"
CACHE="${GLO_CACHE:-}"
VALIDATE_WORKERS="${GLO_VALIDATE_WORKERS:-4}"
# Optional extra train.py arguments (e.g. --stop-after-epoch 30), set by the launcher.
read -r -a EXTRA_ARGS <<< "${GLO_EXTRA_ARGS:-}"
INTERVAL="${GLO_SYNC_INTERVAL:-300}"
gcs() { if gcloud storage --help >/dev/null 2>&1; then gcloud storage "$@"; else gsutil "$@"; fi; }

# The preprocessing cache must be a persistent host dir mounted at /app/.cache;
# without it every container start would rebuild the cache.
if [[ -z "${CACHE}" || ! -d "${CACHE}" ]]; then
  echo "[entrypoint] FATAL: persistent cache dir missing (GLO_CACHE=${CACHE:-unset})." >&2
  echo "[entrypoint] Relaunch with run_training.sh or resume_training.sh." >&2
  rm -f /tmp/glo-nca-training.lock
  exit 1
fi
echo "[entrypoint] launching training container: config=${CONFIG} cache=${CACHE} extra=${GLO_EXTRA_ARGS:-none}"

# --- cloud metadata (best-effort) ---------------------------------------------
META_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_DIGEST="$(docker image inspect glo-nca:latest --format '{{index .RepoDigests 0}}' 2>/dev/null || echo 'local-build:no-digest')"
IMAGE_ID="$(docker image inspect glo-nca:latest --format '{{.Id}}' 2>/dev/null || echo 'unknown')"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
VM_NAME_META="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || echo N/A)"
MACHINE_META="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | sed 's#.*/##' || echo N/A)"
ZONE_META="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | sed 's#.*/##' || echo N/A)"

# --- experiment id, fixed up front and passed via --experiment-id --------------
# so the sync watcher never guesses the newest directory in /out.
if [[ -n "${GLO_RESUME_DIR:-}" ]]; then
  EXP_ID="$(basename "${GLO_RESUME_DIR}")"
else
  # experiment.name from the config, read with the image's Python (the host has no PyYAML).
  EXP_NAME="$(docker run --rm --entrypoint python glo-nca:latest -c \
                "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment']['name'])" \
                "${CONFIG}" 2>/dev/null || true)"
  [[ -n "${EXP_NAME}" ]] || EXP_NAME="GLO-NCA"
  EXP_ID="${EXP_NAME}-$(date +%Y%m%d-%H%M%S)"
fi
EXP_DIR="${OUT}/${EXP_ID}"
echo "[entrypoint] experiment id: ${EXP_ID}"

# --- background GCS sync watcher (runs during training) ---------------------
# Pushes this experiment dir (excluding *.tmp) every interval.
(
  while true; do
    sleep "${INTERVAL}"
    [[ -d "${EXP_DIR}" ]] || continue
    gcs rsync -r -x '.*\.tmp$' "${EXP_DIR}" \
      "gs://${BUCKET}/${EXP_PREFIX}/${EXP_ID}" \
      >/dev/null 2>&1 && echo "[sync] pushed ${EXP_ID}" \
      || echo "[sync] WARNING: sync failed; will retry (local data kept)."
  done
) &
SYNC_PID=$!
trap 'kill "${SYNC_PID}" 2>/dev/null || true' EXIT

# --- training in Docker; writes into $OUT ------------------------------------
# Resume mode when GLO_RESUME_DIR is set, else a fresh --config run.
# --shm-size=8g is required: DataLoader workers pass 128^3 volumes through /dev/shm
# (Docker default 64 MB crashes the workers). Both modes need it, and both mount
# the persistent cache.
set +e
if [[ -n "${GLO_RESUME_DIR:-}" ]]; then
  echo "[entrypoint] RESUME mode for experiment ${EXP_ID}"
  docker run --rm --gpus all --shm-size=8g \
    -v "${DATA}:/data:ro" -v "${OUT}:/out" -v "${CACHE}:/app/.cache" -e DATA_ROOT=/data \
    -e GLO_VALIDATE_WORKERS="${VALIDATE_WORKERS}" \
    glo-nca:latest --resume "/out/${EXP_ID}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
  docker run --rm --gpus all --shm-size=8g \
    -v "${DATA}:/data:ro" -v "${OUT}:/out" -v "${CACHE}:/app/.cache" -e DATA_ROOT=/data \
    -e GLO_VALIDATE_WORKERS="${VALIDATE_WORKERS}" \
    glo-nca:latest --config "${CONFIG}" --output /out \
    --experiment-id "${EXP_ID}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
TRAIN_RC=$?
set -e
kill "${SYNC_PID}" 2>/dev/null || true

# --- cloud metadata into the experiment dir ----------------------------------
if [[ -d "${EXP_DIR}" ]]; then
  cat > "${EXP_DIR}/cloud_metadata.json" <<JSON
{
  "gcp_vm_name": "${VM_NAME_META}",
  "machine_type": "${MACHINE_META}",
  "zone": "${ZONE_META}",
  "gpu_name": "${GPU_NAME}",
  "nvidia_driver": "${DRIVER}",
  "docker_image_id": "${IMAGE_ID}",
  "docker_image_digest": "${IMAGE_DIGEST}",
  "meta_captured": "${META_TAG}",
  "training_exit_code": ${TRAIN_RC}
}
JSON
else
  echo "[entrypoint] WARNING: experiment dir ${EXP_DIR} not found; skipping metadata."
fi

# --- final GCS sync, attempted on every exit path ----------------------------
if [[ -d "${EXP_DIR}" ]]; then
  echo "[entrypoint] final GCS sync (rc=${TRAIN_RC}) -> ${EXP_PREFIX}/${EXP_ID}"
  gcs rsync -r -x '.*\.tmp$' "${EXP_DIR}" "gs://${BUCKET}/${EXP_PREFIX}/${EXP_ID}" || \
    echo "[entrypoint] WARNING: final GCS sync failed; local data preserved at ${EXP_DIR}."
else
  echo "[entrypoint] WARNING: nothing to sync (no ${EXP_DIR}); local disk unchanged."
fi

# Release the training lock (see run_training.sh) regardless of outcome.
rm -f /tmp/glo-nca-training.lock
echo "[entrypoint] training finished rc=${TRAIN_RC}, experiment=${EXP_ID}"
exit "${TRAIN_RC}"
