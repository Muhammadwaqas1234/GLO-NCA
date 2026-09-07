#!/usr/bin/env bash
# Environment verification with clear PASS / WARN / FAIL per check.
# Safe to run from a laptop (checks GCP + local tooling) OR on the GPU VM
# (additionally checks NVIDIA/Docker/container). Never changes anything.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

FAILS=0
chk() { if eval "$2" >/dev/null 2>&1; then pass "$1"; else fail "$1"; FAILS=$((FAILS+1)); fi; }
chkw(){ if eval "$2" >/dev/null 2>&1; then pass "$1"; else warn "$1 (skipped/not on this host)"; fi; }

echo "=== GCP ==="
chk  "gcloud installed"            "command -v gcloud"
chk  "gcloud authenticated"        "gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q ."
chk  "project reachable"           "gcloud projects describe ${GCP_PROJECT_ID}"
chk  "bucket accessible"           "gcs ls ${GCS_ROOT}"

echo
echo "=== Host (GPU only present on the VM) ==="
chkw "NVIDIA GPU visible"          "command -v nvidia-smi && nvidia-smi"
chkw "NVIDIA driver reports"       "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
if command -v df >/dev/null; then
  avail=$(df -BG --output=avail "${VM_DATA_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -n "${avail}" && "${avail}" -ge 50 ]]; then pass "disk >=50GB free at ${VM_DATA_DIR} (${avail}G)"
  elif [[ -n "${avail}" ]]; then warn "low disk at ${VM_DATA_DIR}: ${avail}G"
  else warn "disk check skipped (${VM_DATA_DIR} not present on this host)"; fi
fi

echo
echo "=== Docker ==="
chkw "docker installed"            "command -v docker"
chkw "docker daemon up"            "docker info"
chkw "NVIDIA container runtime"    "docker info | grep -qi nvidia || command -v nvidia-container-toolkit"
chkw "image glo-nca present"       "docker image inspect glo-nca:latest"

echo
echo "=== Container (only if the image + GPU exist) ==="
if docker image inspect glo-nca:latest >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  chkw "CUDA visible in container"   "docker run --rm --gpus all glo-nca:latest -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' --help 2>/dev/null || docker run --rm --gpus all --entrypoint python glo-nca:latest -c 'import torch; assert torch.cuda.is_available()'"
  chkw "GLO-NCA imports in container" "docker run --rm --entrypoint python glo-nca:latest -c 'import src.experiment.runner'"
else
  warn "container CUDA/import checks skipped (need image + GPU on this host)"
fi

echo
if [[ "${FAILS}" -eq 0 ]]; then pass "verify_gcp: no hard failures"; exit 0
else fail "verify_gcp: ${FAILS} hard failure(s) above"; exit 1; fi
