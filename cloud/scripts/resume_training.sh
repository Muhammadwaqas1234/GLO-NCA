#!/usr/bin/env bash
# Resume an experiment from GCS on a (possibly brand-new) VM:
#   GCS experiment -> local restore -> validate checkpoint -> Phase 1 --resume.
# Preserves the ORIGINAL experiment ID and config (no new experiment created,
# no accidental config change).
#
# Usage: ./cloud/scripts/resume_training.sh GLO-NCA-V2-YYYYMMDD-HHMMSS
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

EXP_ID="${1:-}"
[[ -n "${EXP_ID}" ]] || die "usage: resume_training.sh <experiment_id>"
require_gcloud_auth

LOCK="/tmp/glo-nca-training.lock"
if [[ -e "${LOCK}" ]] && kill -0 "$(cat "${LOCK}" 2>/dev/null)" 2>/dev/null; then
  die "training already running (pid $(cat "${LOCK}")). Refusing to start a second."
fi

SRC="${GCS_EXPERIMENTS}/${EXP_ID}"
gcs_exists "${SRC}" || die "experiment not in GCS: ${SRC}"
EXP_DIR="${VM_OUT_DIR}/${EXP_ID}"
mkdir -p "${EXP_DIR}"
log "restoring ${SRC} -> ${EXP_DIR}"
gcs_rsync -r "${SRC}" "${EXP_DIR}"

# --- validate a checkpoint exists and loads (never overwrite a corrupt one) --
PYBIN="$(command -v python3 || command -v python)"
log "validating checkpoint..."
"${PYBIN}" "${CLOUD_DIR}/scripts/validate_checkpoint.py" "${EXP_DIR}" \
  || die "checkpoint validation FAILED -- see message above. Nothing modified."

# --- config-consistency guard: the resumed run MUST reuse the saved config ----
# Phase 1 --resume already reads config/config.yaml from inside the experiment
# dir, so architecture/patch/seed cannot drift. We assert the file is present.
[[ -f "${EXP_DIR}/config/config.yaml" ]] \
  || die "saved config missing (${EXP_DIR}/config/config.yaml); cannot safely resume."
pass "config present; resume will reuse the original experiment configuration."

# --- dataset cache must be valid before GPU spend ---
bash "${CLOUD_DIR}/scripts/cache_dataset.sh"

# --- relaunch under systemd, in Phase 1 --resume mode ---
UNIT=glo-nca-training
sudo cp "${CLOUD_DIR}/systemd/glo-nca-training.service" "/etc/systemd/system/${UNIT}.service"
sudo systemctl daemon-reload
# The entrypoint runs the full config path normally; for resume we point it at
# the experiment dir. We reuse the same env file but flag resume mode.
sudo tee "/etc/glo-nca.env" >/dev/null <<EOF
GLO_REPO=${VM_WORKSPACE}
GLO_DATA=${VM_DATA_DIR}
GLO_OUT=${VM_OUT_DIR}
GLO_CONFIG=${EXP_DIR}/config/config.yaml
GLO_RESUME_DIR=${EXP_DIR}
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
EOF
echo $$ > "${LOCK}"
confirm "Resume ${EXP_ID} on this GPU VM (billing continues while running)?"
sudo systemctl start "${UNIT}"
pass "resume started under systemd unit '${UNIT}' for ${EXP_ID}."
log "logs: sudo journalctl -u ${UNIT} -f   |   status: ./cloud/scripts/monitor.sh"
