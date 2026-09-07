#!/usr/bin/env bash
# Invoked BY the systemd unit (not by the user directly). Reads /etc/glo-nca.env,
# captures cloud metadata, runs the Docker training to completion, and keeps a
# background GCS sync watcher alive so checkpoints reach GCS during training.
#
# Exit code is that of the training container, so systemd sees real success or
# failure (and does NOT restart a completed/failed experiment into a loop).
set -euo pipefail
# shellcheck disable=SC1091
source /etc/glo-nca.env

REPO="${GLO_REPO}"; DATA="${GLO_DATA}"; OUT="${GLO_OUT}"; CONFIG="${GLO_CONFIG}"
BUCKET="${GLO_BUCKET}"; EXP_PREFIX="${GLO_EXP_PREFIX}"
INTERVAL="${GLO_SYNC_INTERVAL:-300}"
gcs() { if gcloud storage --help >/dev/null 2>&1; then gcloud storage "$@"; else gsutil "$@"; fi; }

echo "[entrypoint] launching training container: config=${CONFIG}"

# --- cloud metadata (best-effort; extends, never replaces, Phase 1 manifest) --
META_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_DIGEST="$(docker image inspect glo-nca:latest --format '{{index .RepoDigests 0}}' 2>/dev/null || echo 'local-build:no-digest')"
IMAGE_ID="$(docker image inspect glo-nca:latest --format '{{.Id}}' 2>/dev/null || echo 'unknown')"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
VM_NAME_META="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || echo N/A)"
MACHINE_META="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | sed 's#.*/##' || echo N/A)"
ZONE_META="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | sed 's#.*/##' || echo N/A)"

# --- background GCS sync watcher (runs during training) ---------------------
# Periodically pushes the whole experiment dir to GCS (excluding *.tmp), so a
# VM failure loses at most one sync interval of progress.
(
  while true; do
    sleep "${INTERVAL}"
    latest="$(ls -t "${OUT}" 2>/dev/null | head -1 || true)"
    [[ -n "${latest}" && -d "${OUT}/${latest}" ]] || continue
    gcs rsync -r -x '.*\.tmp$' "${OUT}/${latest}" \
      "gs://${BUCKET}/${EXP_PREFIX}/${latest}" \
      >/dev/null 2>&1 && echo "[sync] pushed ${latest}" \
      || echo "[sync] WARNING: sync failed; will retry (local data kept)."
  done
) &
SYNC_PID=$!
trap 'kill "${SYNC_PID}" 2>/dev/null || true' EXIT

# --- run training in Docker; container writes into $OUT (mounted) -----------
# Resume mode (GLO_RESUME_DIR set) reuses the existing experiment dir and the
# Phase 1 --resume path; otherwise a fresh --config run.
before="$(ls "${OUT}" 2>/dev/null || true)"
set +e
if [[ -n "${GLO_RESUME_DIR:-}" ]]; then
  RES_ID="$(basename "${GLO_RESUME_DIR}")"
  echo "[entrypoint] RESUME mode for experiment ${RES_ID}"
  docker run --rm --gpus all \
    -v "${DATA}:/data:ro" -v "${OUT}:/out" -e DATA_ROOT=/data \
    glo-nca:latest --resume "/out/${RES_ID}"
else
  docker run --rm --gpus all \
    -v "${DATA}:/data:ro" -v "${OUT}:/out" -e DATA_ROOT=/data \
    glo-nca:latest --config "${CONFIG}" --output /out
fi
TRAIN_RC=$?
set -e
after="$(ls "${OUT}" 2>/dev/null || true)"
kill "${SYNC_PID}" 2>/dev/null || true

# Identify the experiment dir. In resume mode it is the restored dir; otherwise
# the newest directory that did not exist before the run.
if [[ -n "${GLO_RESUME_DIR:-}" ]]; then
  EXP_ID="$(basename "${GLO_RESUME_DIR}")"
else
  EXP_ID="$(comm -13 <(echo "${before}" | sort) <(echo "${after}" | sort) | tail -1)"
  [[ -z "${EXP_ID}" ]] && EXP_ID="$(ls -t "${OUT}" | head -1)"
fi
EXP_DIR="${OUT}/${EXP_ID}"

# --- write cloud metadata into the experiment dir (extends manifest) --------
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
  # --- final sync to GCS (exclude in-progress *.tmp) ---
  gcs rsync -r -x '.*\.tmp$' "${EXP_DIR}" "gs://${BUCKET}/${EXP_PREFIX}/${EXP_ID}" || \
    echo "[entrypoint] WARNING: final GCS sync failed; local data preserved."
fi

rm -f /tmp/glo-nca-training.lock
echo "[entrypoint] training finished rc=${TRAIN_RC}, experiment=${EXP_ID}"
exit "${TRAIN_RC}"
