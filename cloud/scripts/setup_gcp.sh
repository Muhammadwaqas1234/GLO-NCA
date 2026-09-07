#!/usr/bin/env bash
# Runs ON THE GPU VM. Idempotent: prepares Docker + NVIDIA Container Toolkit,
# checks out the repo, builds the image, and creates the data/out dirs.
# Deep Learning VM images already ship the NVIDIA driver, Docker and the
# container toolkit; this script only fills gaps and never destroys state.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

log "GLO-NCA V2 VM setup (idempotent)"

# --- Docker ---
if command -v docker >/dev/null 2>&1; then pass "docker present"
else
  log "installing docker"
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
fi

# --- NVIDIA Container Toolkit (needed for --gpus all) ---
if docker info 2>/dev/null | grep -qi nvidia || command -v nvidia-ctk >/dev/null 2>&1; then
  pass "NVIDIA container runtime present"
else
  warn "NVIDIA container toolkit not detected."
  log "on Deep Learning VM images it is preinstalled; otherwise install per:"
  log "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
fi

# --- repo checkout ---
if [[ -d "${VM_WORKSPACE}/.git" ]]; then
  pass "repo already at ${VM_WORKSPACE}"
  git -C "${VM_WORKSPACE}" fetch --quiet origin v2 && \
    git -C "${VM_WORKSPACE}" checkout --quiet v2 && \
    git -C "${VM_WORKSPACE}" pull --quiet origin v2 || warn "git update skipped"
else
  log "cloning repo (branch v2) into ${VM_WORKSPACE}"
  sudo mkdir -p "${VM_WORKSPACE}"; sudo chown "$USER" "${VM_WORKSPACE}"
  git clone -b v2 https://github.com/Muhammadwaqas1234/GLO-NCA.git "${VM_WORKSPACE}"
fi

# --- data / out dirs ---
sudo mkdir -p "${VM_DATA_DIR}" "${VM_OUT_DIR}"
sudo chown "$USER" "${VM_DATA_DIR}" "${VM_OUT_DIR}" || true
pass "data dir ${VM_DATA_DIR}, out dir ${VM_OUT_DIR} ready"

# --- build image ---
log "building Docker image glo-nca:latest (uses the Phase 1 Dockerfile)"
docker build -t glo-nca:latest "${VM_WORKSPACE}"
pass "image built"

log "setup complete. Next: ./cloud/scripts/verify_gcp.sh"
