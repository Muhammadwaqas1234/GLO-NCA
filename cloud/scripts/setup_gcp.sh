#!/usr/bin/env bash
# Runs on the GPU VM; idempotent. Fills Docker / NVIDIA toolkit gaps, checks out
# the repo, creates data/out dirs and builds the commit-stamped image.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

log "GLO-NCA VM setup (idempotent)"

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
# Production branch by default; override with GLO_BRANCH.
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
# The checkout must contain the model and the canonical split before building.
[[ -f "${VM_WORKSPACE}/src/models/Model_GLO_NCA_V3.py" \
   && -f "${VM_WORKSPACE}/split/master_split.json" ]] \
  || die "checkout at ${VM_WORKSPACE} (branch ${GLO_BRANCH}) is missing the GLO-NCA
       model and/or the canonical split. Refusing to build a broken image."
pass "checkout verified: GLO-NCA model + canonical split present (branch ${GLO_BRANCH})"

# --- data / out dirs ---
# VM_CACHE_DIR persists the preprocessing cache across containers (mounted at /app/.cache).
sudo mkdir -p "${VM_DATA_DIR}" "${VM_OUT_DIR}" "${VM_CACHE_DIR}"
sudo chown "$USER" "${VM_DATA_DIR}" "${VM_OUT_DIR}" "${VM_CACHE_DIR}" || true
pass "data dir ${VM_DATA_DIR}, out dir ${VM_OUT_DIR}, cache dir ${VM_CACHE_DIR} ready"

# --- build image ---
log "building Docker image glo-nca:latest (uses the Phase 1 Dockerfile)"
# Stamp the image with its source commit; launchers refuse a mismatched image.
BUILD_COMMIT="$(git -c safe.directory="${VM_WORKSPACE}" -C "${VM_WORKSPACE}" rev-parse HEAD)"
docker build --label "glo.commit=${BUILD_COMMIT}" -t glo-nca:latest "${VM_WORKSPACE}"
pass "image built from commit ${BUILD_COMMIT:0:12}"

log "setup complete. Next: ./cloud/scripts/verify_gcp.sh"
