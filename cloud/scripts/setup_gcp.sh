#!/usr/bin/env bash
# VM setup (idempotent): check out the branch and build the commit-stamped image.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

log "GLO-NCA cascade VM setup (idempotent)"
command -v docker >/dev/null 2>&1 || die "docker missing (Deep Learning VM images ship it)."
pass "docker present"

REMOTE="$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null || echo https://github.com/Muhammadwaqas1234/GLO-NCA.git)"
if [[ -d "${GLO_WORKSPACE}/.git" ]]; then
  git -c safe.directory="${GLO_WORKSPACE}" -C "${GLO_WORKSPACE}" fetch -q origin "${GLO_BRANCH}" \
    && git -c safe.directory="${GLO_WORKSPACE}" -C "${GLO_WORKSPACE}" checkout -q -B "${GLO_BRANCH}" "origin/${GLO_BRANCH}" \
    || die "git update of ${GLO_WORKSPACE} to ${GLO_BRANCH} failed."
else
  git clone -q -b "${GLO_BRANCH}" "${REMOTE}" "${GLO_WORKSPACE}"
fi
[[ -f "${GLO_WORKSPACE}/glo_nca/trainer.py" && -f "${GLO_WORKSPACE}/configs/glo_nca_cascade.yaml" ]] \
  || die "checkout at ${GLO_WORKSPACE} is not the GLO-NCA cascade branch."
pass "checkout ${GLO_WORKSPACE} (${GLO_BRANCH})"

mkdir -p "${VM_OUT_DIR}"
BUILD_COMMIT="$(git -c safe.directory="${GLO_WORKSPACE}" -C "${GLO_WORKSPACE}" rev-parse HEAD)"
docker build -f "${GLO_WORKSPACE}/docker/Dockerfile" --label "glo.commit=${BUILD_COMMIT}" \
  -t "${IMAGE}" "${GLO_WORKSPACE}"
pass "image ${IMAGE} built from commit ${BUILD_COMMIT:0:12}"
