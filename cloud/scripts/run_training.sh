#!/usr/bin/env bash
# Start a run under systemd: run_training.sh [--stop-after-epoch N] [--auto-stop]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

EXTRA_ARGS=""; AUTO_STOP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after-epoch) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--stop-after-epoch needs a positive number"
                        EXTRA_ARGS="--stop-after-epoch $2"; shift 2 ;;
    --auto-stop) AUTO_STOP=1; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

assert_no_training_running
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 || die "NVIDIA GPU not visible."
pass "GPU visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
assert_image_matches_repo "${GLO_WORKSPACE}"
[[ -d "${GLO_DATA_ROOT}" ]] || die "dataset not found at ${GLO_DATA_ROOT}"
mkdir -p "${VM_OUT_DIR}"

RUN_ID="GLO-NCA-CASCADE-$(date -u +%Y%m%d-%H%M%S)"
sudo cp "${GLO_WORKSPACE}/cloud/systemd/glo-nca-cascade-training.service" "/etc/systemd/system/${TRAIN_UNIT}.service"
sudo systemctl daemon-reload
sudo tee /etc/glo-nca-cascade.env >/dev/null <<EOF
GLO_REPO=${GLO_WORKSPACE}
GLO_DATA=${GLO_DATA_ROOT}
GLO_OUT=${VM_OUT_DIR}
GLO_RUN_ID=${RUN_ID}
GLO_MODE=new
GLO_EXTRA_ARGS="${EXTRA_ARGS}"
GLO_AUTO_STOP=${AUTO_STOP}
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
EOF

confirm "Start GLO-NCA cascade run ${RUN_ID} ${EXTRA_ARGS:-(full planned budget)} (billing continues while running)?"
write_training_lock "${RUN_ID}"
sudo systemctl reset-failed "${TRAIN_UNIT}" 2>/dev/null || true
sudo systemctl start "${TRAIN_UNIT}"
pass "cascade run ${RUN_ID} started under systemd unit '${TRAIN_UNIT}'."
log "logs:   sudo journalctl -u ${TRAIN_UNIT} -f"
log "status: ./cloud/scripts/status.sh ${RUN_ID}"
log "GCS:    ${GCS_EXPERIMENTS}/${RUN_ID}/"
