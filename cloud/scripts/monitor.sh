#!/usr/bin/env bash
# Live status of the current training run on the VM. Read-only. Prints N/A for
# anything unavailable rather than inventing values.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

OUT="${VM_OUT_DIR}"
EXP_DIR="${1:-$(ls -t "${OUT}" 2>/dev/null | head -1)}"
[[ -n "${EXP_DIR}" ]] && EXP_DIR="${OUT}/${EXP_DIR#"${OUT}/"}"

jqget() { # crude JSON field read without jq dependency
  grep -o "\"$2\"[^,}]*" "$1" 2>/dev/null | head -1 | sed 's/.*: *//; s/"//g' || true; }

echo "=== Experiment ==="
if [[ -n "${EXP_DIR}" && -f "${EXP_DIR}/status.json" ]]; then
  echo "Experiment:            $(basename "${EXP_DIR}")"
  echo "State:                 $(jqget "${EXP_DIR}/status.json" state)"
  echo "Epoch:                 $(jqget "${EXP_DIR}/status.json" current_epoch)/$(jqget "${EXP_DIR}/status.json" total_epochs)"
  echo "Best epoch:            $(jqget "${EXP_DIR}/status.json" best_epoch)"
  echo "Best validation score: $(jqget "${EXP_DIR}/status.json" best_score)"
  last_ck="${EXP_DIR}/checkpoints/last.pth"
  [[ -f "${last_ck}" ]] && echo "Last checkpoint:       $(date -r "${last_ck}" 2>/dev/null || stat -c %y "${last_ck}" 2>/dev/null)" \
                        || echo "Last checkpoint:       N/A"
else
  echo "Experiment:            N/A (no experiment dir under ${OUT})"
fi

echo
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "GPU:                   %s\nGPU utilization:       %s\nGPU memory:            %s / %s\nTemperature:           %s\n",$1,$2,$3,$4,$5}'
else
  echo "GPU:                   N/A (nvidia-smi not found)"
fi

echo
echo "=== System ==="
echo "Disk (${VM_DATA_DIR}):        $(df -h "${VM_DATA_DIR}" 2>/dev/null | awk 'NR==2{print $4" free of "$2}' || echo N/A)"
echo "Disk (${VM_OUT_DIR}):         $(df -h "${VM_OUT_DIR}" 2>/dev/null | awk 'NR==2{print $4" free of "$2}' || echo N/A)"
echo "systemd unit:          $(systemctl is-active glo-nca-training 2>/dev/null || echo N/A)"
echo
echo "logs:  sudo journalctl -u glo-nca-training -f"
