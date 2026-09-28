#!/usr/bin/env bash
# Read-only run status: service state, progress, last epochs and GPU: status.sh [run-id]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

RUN_ID="${1:-$(ls -t "${VM_OUT_DIR}" 2>/dev/null | grep '^GLO-NCA-CASCADE-' | head -1)}"
[[ -n "${RUN_ID}" ]] || die "no cascade run under ${VM_OUT_DIR}"
RUN_DIR="${VM_OUT_DIR}/${RUN_ID}"
echo "run:     ${RUN_ID}"
echo "service: $(systemctl is-active "${TRAIN_UNIT}" 2>/dev/null || true)"
echo "status:  $(tr -d '\n ' < "${RUN_DIR}/status.json" 2>/dev/null || echo N/A)"
echo "--- last epochs"
grep -E "^ep [0-9]+/" "${RUN_DIR}/train.log" 2>/dev/null | tail -5 || echo "N/A"
echo "--- GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo N/A
