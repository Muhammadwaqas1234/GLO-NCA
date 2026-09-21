#!/usr/bin/env bash
# Start a GLO-NCA V2 training run on the VM, resiliently:
#   * validates GPU + dataset cache first (refuses to burn GPU on bad input)
#   * runs training inside Docker under a systemd unit -> survives SSH
#     disconnect / terminal close / laptop shutdown
#   * captures cloud metadata into the experiment dir
#   * starts a background GCS sync watcher (checkpoints -> GCS)
#   * a lock prevents two concurrent training processes on this VM
#
# Usage:
#   ./cloud/scripts/run_training.sh configs/glo_nca_production.yaml   # THE run
#
# configs/glo_nca_production.yaml is the ONLY production configuration. The
# other configs in configs/ are historical (V2-era baselines and the frozen
# V3 reference) and train a DIFFERENT architecture; they are kept because
# audit scripts and thesis evidence cite them, not because they are options.
#
# This NEVER starts automatically -- the user runs it explicitly.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

# Phase 2: V3 is the production architecture. There is deliberately NO default
# config -- the old default (configs/historical/gcp_full.yaml) was the V2 baseline, so a
# bare `run_training.sh` would have launched the WRONG architecture for days.
CONFIG="${1:-}"
[[ -n "${CONFIG}" ]] || die "usage: run_training.sh <config>
       production : ./cloud/scripts/run_training.sh configs/glo_nca_production.yaml
       reference  : ./cloud/scripts/run_training.sh configs/historical/v3_multilevel_ckpt.yaml   (FROZEN 32/96/128 -- historical)
       Verify first: python scripts/verify_glo_nca_production_config.py configs/glo_nca_production.yaml"
[[ -f "${REPO_DIR}/${CONFIG}" || -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"

# Concurrency guard backed by systemd (the real job owner), not a launcher pid.
assert_no_training_running

# --- 1) GPU present? ---
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
  || die "NVIDIA GPU not visible. Run on the GPU VM after verify_gcp.sh."
pass "GPU visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- 2) dataset cache valid? ---
bash "${CLOUD_DIR}/scripts/cache_dataset.sh"

# --- 3) image present? ---
docker image inspect glo-nca:latest >/dev/null 2>&1 \
  || die "docker image glo-nca:latest missing. Run setup_gcp.sh."

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
GLO_CONFIG=${CONFIG}
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
EOF

# Confirm BEFORE writing the lock, so declining leaves no stale lock behind.
confirm "Start GLO-NCA training with ${CONFIG} on this GPU VM (billing continues while running)?"
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
