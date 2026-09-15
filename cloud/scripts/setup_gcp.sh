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
# Phase 2 (P1): the branch was hard-coded to 'v2', so a VM provisioned for the
# V3 campaign silently received the V2 baseline WITHOUT the V3 model, agent or
# configs. Default to the V3 branch; override with GLO_BRANCH for a V2 run.
GLO_BRANCH="${GLO_BRANCH:-v3-multilevel}"
if [[ -d "${VM_WORKSPACE}/.git" ]]; then
  pass "repo already at ${VM_WORKSPACE}"
  git -C "${VM_WORKSPACE}" fetch --quiet origin "${GLO_BRANCH}" && \
    git -C "${VM_WORKSPACE}" checkout --quiet "${GLO_BRANCH}" && \
    git -C "${VM_WORKSPACE}" pull --quiet origin "${GLO_BRANCH}" \
    || die "git update to branch '${GLO_BRANCH}' FAILED. Refusing to build an
       image from an unknown checkout -- fix the repo state and re-run."
else
  log "cloning repo (branch ${GLO_BRANCH}) into ${VM_WORKSPACE}"
  sudo mkdir -p "${VM_WORKSPACE}"; sudo chown "$USER" "${VM_WORKSPACE}"
  git clone -b "${GLO_BRANCH}" https://github.com/Muhammadwaqas1234/GLO-NCA.git "${VM_WORKSPACE}"
fi
# Prove the checkout actually contains V3 before building an image from it.
[[ -f "${VM_WORKSPACE}/src/models/Model_GLO_NCA_V3.py" \
   && -f "${VM_WORKSPACE}/split/master_split.json" ]] \
  || die "checkout at ${VM_WORKSPACE} (branch ${GLO_BRANCH}) is missing the V3
       model and/or the canonical split. Refusing to build a broken image."
pass "checkout verified: V3 model + canonical split present (branch ${GLO_BRANCH})"

# --- data / out dirs ---
sudo mkdir -p "${VM_DATA_DIR}" "${VM_OUT_DIR}"
sudo chown "$USER" "${VM_DATA_DIR}" "${VM_OUT_DIR}" || true
pass "data dir ${VM_DATA_DIR}, out dir ${VM_OUT_DIR} ready"

# --- build image ---
log "building Docker image glo-nca:latest (uses the Phase 1 Dockerfile)"
docker build -t glo-nca:latest "${VM_WORKSPACE}"
pass "image built"

log "setup complete. Next: ./cloud/scripts/verify_gcp.sh"
