#!/usr/bin/env bash
# Start a two-level GLO-NCA training run on the VM:
#   checks GPU, dataset cache and image commit, then runs Docker training under
#   systemd (survives SSH disconnect) with GCS sync and a single-run lock.
#
# Usage: ./cloud/scripts/run_training.sh configs/glo_nca_production.yaml [--stop-after-epoch N]
# configs/glo_nca_production.yaml is the only production configuration.
# Never starts automatically.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

# No default config: a bare run must fail rather than train the wrong model.
CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "usage: run_training.sh <config>
       production : ./cloud/scripts/run_training.sh configs/glo_nca_production.yaml
       Verify first: python scripts/verify_glo_nca_production_config.py configs/glo_nca_production.yaml"
[[ -f "${REPO_DIR}/${CONFIG}" || -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
# Optional pause: --stop-after-epoch N checkpoints epoch N and exits before any
# end-of-run evaluation; the schedule is unchanged and resume_training.sh continues.
EXTRA_ARGS=""
if [[ "${2:-}" == "--stop-after-epoch" ]]; then
  [[ "${3:-}" =~ ^[1-9][0-9]*$ ]] || die "--stop-after-epoch needs a positive epoch number"
  EXTRA_ARGS="--stop-after-epoch ${3}"
elif [[ -n "${2:-}" ]]; then
  die "unknown option: ${2} (only --stop-after-epoch N is supported)"
fi

# Concurrency guard backed by systemd (the real job owner), not a launcher pid.
assert_no_training_running

# --- 1) GPU present? ---
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
  || die "NVIDIA GPU not visible. Run on the GPU VM after verify_gcp.sh."
pass "GPU visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- 2) image present AND built from the current commit? ---
# Checked first: dataset validation below runs inside this image.
assert_image_matches_repo "${VM_WORKSPACE}"

# --- 3) dataset cache valid, persistent preprocessing cache present? ---
bash "${CLOUD_DIR}/scripts/cache_dataset.sh"
ensure_cache_dir

# --- 4) launch under systemd (persistent) ---
mkdir -p "${VM_OUT_DIR}"
UNIT=glo-nca-training
log "installing systemd unit ${UNIT} (survives SSH disconnect)"
sudo cp "${CLOUD_DIR}/systemd/glo-nca-training.service" "/etc/systemd/system/${UNIT}.service"
sudo systemctl daemon-reload

# Pass runtime settings to the unit via an env file it reads.
sudo tee "/etc/glo-nca.env" >/dev/null <<EOF
GLO_REPO=${VM_WORKSPACE}
GLO_DATA=${VM_DATA_DIR}
GLO_OUT=${VM_OUT_DIR}
GLO_CACHE=${VM_CACHE_DIR}
GLO_VALIDATE_WORKERS=${VALIDATE_WORKERS}
GLO_EXTRA_ARGS="${EXTRA_ARGS}"
GLO_CONFIG=${CONFIG}
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
EOF

# Confirm BEFORE writing the lock, so declining leaves no stale lock behind.
confirm "Start GLO-NCA training with ${CONFIG} ${EXTRA_ARGS:-(full budget)} on this GPU VM (billing continues while running)?"
write_training_lock "${CONFIG}"
sudo systemctl start "${UNIT}"
pass "training started under systemd unit '${UNIT}'."

echo
log "It now runs independently of your SSH session. Useful commands:"
log "  status : ./cloud/scripts/monitor.sh"
log "  logs   : sudo journalctl -u ${UNIT} -f"
log "  stop   : sudo systemctl stop ${UNIT}   (then sync + stop_vm.sh)"
log "GCS sync watcher is launched by the unit; checkpoints land in"
log "  ${GCS_EXPERIMENTS}/<experiment_id>/"
