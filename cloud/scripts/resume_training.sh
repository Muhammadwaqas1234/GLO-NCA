#!/usr/bin/env bash
# Resume a run from last.pth (restored from GCS if missing): resume_training.sh <run-id> [--data-root DIR] [--stop-after-epoch N] [--auto-stop] [--keep-alive]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

RUN_ID="${1:-}"; [[ -n "${RUN_ID}" ]] || die "usage: resume_training.sh <run_id> [--data-root DIR] [--stop-after-epoch N] [--auto-stop]"
shift
EXTRA_ARGS=""; AUTO_STOP=0; KEEP_ALIVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after-epoch) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--stop-after-epoch needs a positive number"
                        EXTRA_ARGS="--stop-after-epoch $2"; shift 2 ;;
    --auto-stop) AUTO_STOP=1; shift ;;
    --keep-alive) KEEP_ALIVE=1; shift ;;
    --data-root) GLO_DATA_ROOT="${2:?--data-root needs a path}"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_gcloud_auth
assert_no_training_running
assert_image_matches_repo "${GLO_WORKSPACE}"

RUN_DIR="${VM_OUT_DIR}/${RUN_ID}"
if [[ ! -f "${RUN_DIR}/last.pth" ]]; then
  gcs_exists "${GCS_EXPERIMENTS}/${RUN_ID}" || die "run ${RUN_ID} not found locally or in GCS."
  log "restoring ${RUN_ID} from GCS"
  mkdir -p "${RUN_DIR}"; gcs_rsync -r "${GCS_EXPERIMENTS}/${RUN_ID}" "${RUN_DIR}"
fi
[[ -f "${RUN_DIR}/last.pth" && -f "${RUN_DIR}/config.yaml" ]] \
  || die "${RUN_DIR} lacks last.pth or config.yaml; cannot resume."
pass "resuming from ${RUN_DIR} ($(grep -o '"epoch": [0-9]*' "${RUN_DIR}/status.json" 2>/dev/null || echo 'epoch ?'))"

sudo cp "${GLO_WORKSPACE}/cloud/systemd/glo-nca-cascade-training.service" "/etc/systemd/system/${TRAIN_UNIT}.service"
sudo systemctl daemon-reload
sudo tee /etc/glo-nca-cascade.env >/dev/null <<EOF
GLO_REPO=${GLO_WORKSPACE}
GLO_DATA=${GLO_DATA_ROOT}
GLO_OUT=${VM_OUT_DIR}
GLO_RUN_ID=${RUN_ID}
GLO_MODE=resume
GLO_EXTRA_ARGS="${EXTRA_ARGS}"
GLO_AUTO_STOP=${AUTO_STOP}
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
EOF

confirm "Resume cascade run ${RUN_ID} ${EXTRA_ARGS:-(to the end)} (billing continues while running)?"
write_training_lock "${RUN_ID}"
sudo systemctl reset-failed "${TRAIN_UNIT}" 2>/dev/null || true
sudo systemctl start "${TRAIN_UNIT}"
pass "cascade run ${RUN_ID} resumed under systemd unit '${TRAIN_UNIT}'."
if [[ "${KEEP_ALIVE}" == "1" ]]; then
  # Keep the original config, needed only if the run must restart before its first checkpoint.
  enable_keep_alive "${RUN_ID}" "$(sed -n 's/^KEEP_CONFIG=//p' "${KEEPALIVE_FILE}" 2>/dev/null)" "${GLO_DATA_ROOT}" "${EXTRA_ARGS#--stop-after-epoch }"
fi
log "logs: sudo journalctl -u ${TRAIN_UNIT} -f"
