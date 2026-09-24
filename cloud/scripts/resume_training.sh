#!/usr/bin/env bash
# Resume an experiment on any VM: pick the newest valid copy (local run dir or
# GCS) -> validate checkpoint -> train.py --resume, keeping the original
# experiment id and saved config.
#
# Usage: ./cloud/scripts/resume_training.sh GLO-NCA-PRODUCTION-YYYYMMDD-HHMMSS [--stop-after-epoch N]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

EXP_ID="${1:-}"
[[ -n "${EXP_ID}" ]] || die "usage: resume_training.sh <experiment_id> [--stop-after-epoch N]"
# Optional pause: --stop-after-epoch N checkpoints epoch N and exits before any
# end-of-run evaluation; the schedule is unchanged and resume_training.sh continues.
EXTRA_ARGS=""
if [[ "${2:-}" == "--stop-after-epoch" ]]; then
  [[ "${3:-}" =~ ^[1-9][0-9]*$ ]] || die "--stop-after-epoch needs a positive epoch number"
  EXTRA_ARGS="--stop-after-epoch ${3}"
elif [[ -n "${2:-}" ]]; then
  die "unknown option: ${2} (only --stop-after-epoch N is supported)"
fi
require_gcloud_auth

# Same systemd-backed concurrency guard as run_training.sh.
assert_no_training_running

# The image must match the commit: checkpoint checks and dataset validation run in it.
assert_image_matches_repo "${VM_WORKSPACE}"

SRC="${GCS_EXPERIMENTS}/${EXP_ID}"
EXP_DIR="${VM_OUT_DIR}/${EXP_ID}"
STAGE="${VM_OUT_DIR}/.resume-staging-${EXP_ID}"
mkdir -p "${VM_OUT_DIR}"

# --- choose the source: never let an older GCS copy overwrite a newer local one ---
# GCS is downloaded into a staging dir, never over the live run directory.
rm -rf "${STAGE}"; mkdir -p "${STAGE}"
if gcs_exists "${SRC}"; then
  log "staging ${SRC} -> ${STAGE}"
  gcs_rsync -r "${SRC}" "${STAGE}"
else
  warn "experiment not in GCS (${SRC}); only the local copy can be used"
fi
set +e
DECISION="$(repo_python --ro "${VM_OUT_DIR}" -- \
  cloud/scripts/select_resume_source.py "${EXP_DIR}" "${STAGE}")"
RC=$?
set -e
echo "${DECISION}"
[[ ${RC} -eq 0 ]] || { rm -rf "${STAGE}"; die "no valid checkpoint locally or in GCS for ${EXP_ID}. Nothing modified."; }
SOURCE="$(sed -n 's/^SOURCE=//p' <<< "${DECISION}")"
case "${SOURCE}" in
  local)
    rm -rf "${STAGE}"
    pass "resuming from the local run directory ${EXP_DIR}" ;;
  gcs)
    if [[ -e "${EXP_DIR}" ]]; then
      BACKUP="${EXP_DIR}.local-$(date +%Y%m%d-%H%M%S)"
      mv "${EXP_DIR}" "${BACKUP}"
      warn "older local copy kept (not deleted) at ${BACKUP}"
    fi
    mv "${STAGE}" "${EXP_DIR}"
    pass "resuming from the GCS copy, restored to ${EXP_DIR}" ;;
  *) rm -rf "${STAGE}"; die "unexpected resume decision: '${SOURCE}'" ;;
esac

# --- checkpoint must exist and load ------------------------------------------
log "validating checkpoint..."
repo_python --ro "${EXP_DIR}" -- cloud/scripts/validate_checkpoint.py "${EXP_DIR}" \
  || die "checkpoint validation FAILED -- see message above. Nothing modified."

# --- resume reuses config/config.yaml from the experiment dir; require it -------
[[ -f "${EXP_DIR}/config/config.yaml" ]] \
  || die "saved config missing (${EXP_DIR}/config/config.yaml); cannot safely resume."
pass "config present; resume will reuse the original experiment configuration."

# --- dataset cache valid, persistent preprocessing cache present ---
bash "${CLOUD_DIR}/scripts/cache_dataset.sh"
ensure_cache_dir

# --- relaunch under systemd in resume mode ---
UNIT=glo-nca-training
sudo cp "${CLOUD_DIR}/systemd/glo-nca-training.service" "/etc/systemd/system/${UNIT}.service"
sudo systemctl daemon-reload
# Same env file as a fresh run, plus GLO_RESUME_DIR to select resume mode.
sudo tee "/etc/glo-nca.env" >/dev/null <<EOF
GLO_REPO=${VM_WORKSPACE}
GLO_DATA=${VM_DATA_DIR}
GLO_OUT=${VM_OUT_DIR}
GLO_CACHE=${VM_CACHE_DIR}
GLO_VALIDATE_WORKERS=${VALIDATE_WORKERS}
GLO_EXTRA_ARGS="${EXTRA_ARGS}"
GLO_CONFIG=${EXP_DIR}/config/config.yaml
GLO_RESUME_DIR=${EXP_DIR}
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
EOF
# Confirm BEFORE writing the lock, so declining leaves no stale lock behind.
confirm "Resume ${EXP_ID} ${EXTRA_ARGS:-(to the planned budget)} on this GPU VM (billing continues while running)?"
write_training_lock "${EXP_ID}"
sudo systemctl start "${UNIT}"
pass "resume started under systemd unit '${UNIT}' for ${EXP_ID}."
log "logs: sudo journalctl -u ${UNIT} -f   |   status: ./cloud/scripts/monitor.sh"
