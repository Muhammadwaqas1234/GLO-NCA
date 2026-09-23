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

# --- resolve the experiment id UP FRONT (Phase 2, P1) ------------------------
# Previously the sync watcher used `ls -t "${OUT}" | head -1`, i.e. the most
# recently MODIFIED directory in /out. Any other directory touched later (an
# earlier experiment, a gate smoke run) silently became the sync target and the
# LIVE experiment was never pushed to GCS. The id is now decided here and passed
# explicitly to train.py via --experiment-id, so every consumer agrees on it.
if [[ -n "${GLO_RESUME_DIR:-}" ]]; then
  EXP_ID="$(basename "${GLO_RESUME_DIR}")"
else
  # Read experiment.name from the config on the HOST (the image ENTRYPOINT is
  # `python train.py`, so it cannot be used as a generic interpreter).
  EXP_NAME="$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment']['name'])" \
                "${REPO}/${CONFIG}" 2>/dev/null || true)"
  [[ -n "${EXP_NAME}" ]] || EXP_NAME="GLO-NCA"
  EXP_ID="${EXP_NAME}-$(date +%Y%m%d-%H%M%S)"
fi
EXP_DIR="${OUT}/${EXP_ID}"
echo "[entrypoint] experiment id: ${EXP_ID}"

# --- background GCS sync watcher (runs during training) ---------------------
# Pushes THIS experiment dir to GCS (excluding *.tmp), so a VM failure loses at
# most one sync interval of progress. No directory guessing.
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

# --- run training in Docker; container writes into $OUT (mounted) -----------
# Resume mode (GLO_RESUME_DIR set) reuses the existing experiment dir and the
# Phase 1 --resume path; otherwise a fresh --config run.
#
# --shm-size=8g is LOAD-BEARING, not a tuning knob. Docker defaults
# /dev/shm to 64 MB. PyTorch DataLoader workers pass whole 128^3 volumes
# between processes through shared memory, so with `training.workers: 2`
# that default is exhausted and a worker dies with:
#     ERROR: Unexpected bus error ... insufficient shared memory (shm)
#     RuntimeError: DataLoader worker (pid ...) exited unexpectedly
# Observed on the L4 during Phase 2 validation: the run crashed at first
# data load, AFTER the full dataset-validation scan had already been paid
# for. BOTH invocations need it -- a resumed run loads data identically.
set +e
if [[ -n "${GLO_RESUME_DIR:-}" ]]; then
  echo "[entrypoint] RESUME mode for experiment ${EXP_ID}"
  docker run --rm --gpus all --shm-size=8g \
    -v "${DATA}:/data:ro" -v "${OUT}:/out" -e DATA_ROOT=/data \
    glo-nca:latest --resume "/out/${EXP_ID}"
else
  docker run --rm --gpus all --shm-size=8g \
    -v "${DATA}:/data:ro" -v "${OUT}:/out" -e DATA_ROOT=/data \
    glo-nca:latest --config "${CONFIG}" --output /out \
    --experiment-id "${EXP_ID}"
fi
TRAIN_RC=$?
set -e
kill "${SYNC_PID}" 2>/dev/null || true

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
else
  echo "[entrypoint] WARNING: experiment dir ${EXP_DIR} not found; skipping metadata."
fi

# --- final sync to GCS (ALWAYS attempted; Phase 2, P0-E) ---------------------
# Previously both the metadata write AND the final sync sat inside
# `if [[ -d "${EXP_DIR}" ]]`, so a training crash that left the directory
# unresolved skipped the final sync entirely -- losing up to a full sync
# interval of GPU progress at exactly the moment it mattered most. The sync is
# now attempted on EVERY exit path, success or failure, and only its own result
# is allowed to fail softly (local data is always preserved on the VM disk).
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
