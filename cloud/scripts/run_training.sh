#!/usr/bin/env bash
# Start a run under systemd: run_training.sh [--config PATH] [--data-root DIR] [--stop-after-epoch N] [--auto-stop] [--keep-alive] [--slot NAME]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

EXTRA_ARGS=""; AUTO_STOP=0; KEEP_ALIVE=0; CONFIG="configs/glo_nca_cascade.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop-after-epoch) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--stop-after-epoch needs a positive number"
                        EXTRA_ARGS="--stop-after-epoch $2"; shift 2 ;;
    --auto-stop) AUTO_STOP=1; shift ;;
    --keep-alive) KEEP_ALIVE=1; shift ;;
    --slot) SLOT_ARG="${2:?--slot needs a name}"; shift 2 ;;
    --config) CONFIG="${2:?--config needs a path}"; shift 2 ;;
    --data-root) GLO_DATA_ROOT="${2:?--data-root needs a path}"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

use_slot "${SLOT_ARG:-}"
assert_no_training_running
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 || die "NVIDIA GPU not visible."
pass "GPU visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
assert_image_matches_repo "${GLO_WORKSPACE}"
[[ -d "${GLO_DATA_ROOT}" ]] || die "dataset not found at ${GLO_DATA_ROOT}"
[[ -f "${GLO_WORKSPACE}/${CONFIG}" ]] || die "config not found: ${CONFIG}"
mkdir -p "${VM_OUT_DIR}"

RUN_PREFIX="$(basename "${CONFIG}" .yaml | tr "a-z_" "A-Z-")"
RUN_ID="${RUN_PREFIX}-$(date -u +%Y%m%d-%H%M%S)"
install_train_unit
write_train_env "${RUN_ID}" new "${CONFIG}" "${EXTRA_ARGS}" "${AUTO_STOP}"

confirm "Start GLO-NCA cascade run ${RUN_ID} ${EXTRA_ARGS:-(full planned budget)} (billing continues while running)?"
write_training_lock "${RUN_ID}"
sudo systemctl reset-failed "${TRAIN_UNIT}" 2>/dev/null || true
sudo systemctl start "${TRAIN_UNIT}"
pass "cascade run ${RUN_ID} started under systemd unit '${TRAIN_UNIT}'."
if [[ "${KEEP_ALIVE}" == "1" ]]; then
  enable_keep_alive "${RUN_ID}" "${CONFIG:-}" "${GLO_DATA_ROOT}" "${EXTRA_ARGS#--stop-after-epoch }"
fi
log "logs:   sudo journalctl -u ${TRAIN_UNIT} -f"
log "status: ./cloud/scripts/status.sh ${RUN_ID}"
log "GCS:    ${GCS_EXPERIMENTS}/${RUN_ID}/"
