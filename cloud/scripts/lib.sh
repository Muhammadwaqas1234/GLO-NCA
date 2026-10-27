#!/usr/bin/env bash
# Shared shell library: config, logging, GCS helpers and safety guards.
set -euo pipefail

# --- locate repo + cloud dirs regardless of where a script is called from ----
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${CLOUD_DIR}/.." && pwd)"
CONFIG_FILE="${CLOUD_DIR}/config/gcp.env"

# Cascade names, kept separate from the production GLO-NCA setup on the same VM.
IMAGE="glo-nca-cascade:latest"
TRAIN_UNIT="glo-nca-cascade-training"
TRAIN_LOCK="/tmp/glo-nca-cascade-training.lock"
KEEPALIVE_FILE="/var/lib/glo-nca-cascade/keepalive.env"
AUTORESUME_UNIT="glo-nca-cascade-autoresume"
ENV_FILE="/etc/glo-nca-cascade.env"
RC_FILE="/var/lib/glo-nca-cascade/last_rc"
SLOT=""

# A slot gives a run its own systemd unit, environment, lock, exit code and keep-alive state, so two
# runs can share one GPU VM. Without a slot the single-run names above are used.
use_slot() {  # use_slot <name or empty>
  [[ -z "${1:-}" ]] && return 0
  [[ "$1" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "--slot must use lowercase letters, digits and hyphens: $1"
  SLOT="$1"
  TRAIN_UNIT="glo-nca-cascade-training-$1"
  TRAIN_LOCK="/tmp/glo-nca-cascade-training-$1.lock"
  KEEPALIVE_FILE="/var/lib/glo-nca-cascade/keepalive-$1.env"
  AUTORESUME_UNIT="glo-nca-cascade-autoresume-$1"
  ENV_FILE="/etc/glo-nca-cascade-$1.env"
  RC_FILE="/var/lib/glo-nca-cascade/last_rc-$1"
}

# Install this slot's training unit (the shared template, pointed at the slot's environment file).
install_train_unit() {
  sed "s#/etc/glo-nca-cascade.env#${ENV_FILE}#" "${GLO_WORKSPACE}/cloud/systemd/glo-nca-cascade-training.service" \
    | sudo tee "/etc/systemd/system/${TRAIN_UNIT}.service" >/dev/null
  # Wait for a usable GPU before the container starts (serialised across slots, see _wait_gpu.sh).
  sudo mkdir -p "/etc/systemd/system/${TRAIN_UNIT}.service.d"
  printf '[Service]\nExecStartPre=/bin/bash %s\nTimeoutStartSec=900\n' "${GLO_WORKSPACE}/cloud/scripts/_wait_gpu.sh" \
    | sudo tee "/etc/systemd/system/${TRAIN_UNIT}.service.d/wait-gpu.conf" >/dev/null
  sudo systemctl daemon-reload
}

# Environment of the training unit; the entrypoint reads it.
write_train_env() {  # write_train_env <run-id> <mode new|resume> <config> <extra-args> <auto-stop>
  sudo mkdir -p /var/lib/glo-nca-cascade
  sudo tee "${ENV_FILE}" >/dev/null <<EOF
GLO_REPO=${GLO_WORKSPACE}
GLO_DATA=${GLO_DATA_ROOT}
GLO_OUT=${VM_OUT_DIR}
GLO_RUN_ID=$1
GLO_MODE=$2
GLO_CONFIG=$3
GLO_EXTRA_ARGS="$4"
GLO_AUTO_STOP=$5
GLO_BUCKET=${GCS_BUCKET}
GLO_EXP_PREFIX=${EXPERIMENT_PREFIX}
GLO_SYNC_INTERVAL=${SYNC_INTERVAL_SECONDS}
GLO_UNIT=${TRAIN_UNIT}
GLO_LOCK=${TRAIN_LOCK}
GLO_RC_FILE=${RC_FILE}
EOF
}

# --- logging -----------------------------------------------------------------
if [[ -t 1 ]]; then _G=$'\033[32m'; _Y=$'\033[33m'; _R=$'\033[31m'; _N=$'\033[0m'
else _G=""; _Y=""; _R=""; _N=""; fi
log()  { echo "[$(date +%H:%M:%S)] $*"; }
pass() { echo "${_G}PASS${_N} $*"; }
warn() { echo "${_Y}WARN${_N} $*"; }
fail() { echo "${_R}FAIL${_N} $*" >&2; }
die()  { fail "$*"; exit 1; }

# --- config ------------------------------------------------------------------
load_config() {
  [[ -f "${CONFIG_FILE}" ]] || die "config not found: ${CONFIG_FILE}
       copy cloud/config/gcp.env.example to cloud/config/gcp.env and edit it."
  # shellcheck disable=SC1090
  set -a; source "${CONFIG_FILE}"; set +a
  : "${GCP_PROJECT_ID:?set GCP_PROJECT_ID in gcp.env}"
  : "${GCP_ZONE:?set GCP_ZONE in gcp.env}"
  : "${GCS_BUCKET:?set GCS_BUCKET in gcp.env}"
  : "${VM_NAME:?set VM_NAME in gcp.env}"
  EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-experiments}"
  GLO_DATA_ROOT="${GLO_DATA_ROOT:-/data/brats}"
  VM_OUT_DIR="${VM_OUT_DIR:-/out}"
  GLO_WORKSPACE="${GLO_WORKSPACE:-/opt/glo-nca-cascade}"
  GLO_BRANCH="${GLO_BRANCH:-main}"
  SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
  GCS_EXPERIMENTS="gs://${GCS_BUCKET}/${EXPERIMENT_PREFIX}"
}

# --- gcloud / gcs tooling ----------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
gcs() {
  if gcloud storage --help >/dev/null 2>&1; then gcloud storage "$@"
  else gsutil "$@"; fi
}
gcs_rsync() {
  if gcloud storage --help >/dev/null 2>&1; then gcloud storage rsync "$@"
  else gsutil -m rsync "$@"; fi
}
gcs_exists() { gcs ls "$1" >/dev/null 2>&1; }

require_gcloud_auth() {
  need gcloud
  gcloud auth list --filter=status:ACTIVE --format="value(account)" \
    | grep -q . || die "no active gcloud auth. run: gcloud auth login"
  gcloud config set project "${GCP_PROJECT_ID}" >/dev/null
}

vm_flags() { echo "--zone=${GCP_ZONE} --project=${GCP_PROJECT_ID}"; }

confirm() { # confirm "message" -- interactive guard for anything that costs money
  local msg="${1:-Proceed?}"
  [[ "${ASSUME_YES:-0}" == "1" ]] && { log "${msg} [yes: ASSUME_YES=1]"; return 0; }
  read -r -p "${msg} [y/N] " ans
  [[ "${ans}" == "y" || "${ans}" == "Y" ]] || die "aborted by user."
}

# The container runs the code baked into the image, so the image must match the checkout's commit.
assert_image_matches_repo() {
  local repo="${1:-${GLO_WORKSPACE}}" img head
  docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "image ${IMAGE} missing. Run setup_gcp.sh."
  head="$(git -c safe.directory="${repo}" -C "${repo}" rev-parse HEAD 2>/dev/null)"
  [[ -n "${head}" ]] || die "cannot read git HEAD in ${repo}."
  img="$(docker image inspect -f '{{ index .Config.Labels "glo.commit" }}' "${IMAGE}" 2>/dev/null)"
  [[ "${img}" == "${head}" ]] || die "STALE IMAGE: built from ${img:0:12}, repo at ${head:0:12}. Run setup_gcp.sh."
  git -c safe.directory="${repo}" -C "${repo}" diff --quiet HEAD -- \
    || die "repo ${repo} has uncommitted changes; the image cannot match one commit."
  pass "image ${IMAGE} matches repo commit ${head:0:12}"
}

# --- training concurrency guard ------------------------------------------------
training_is_active() { systemctl is-active --quiet "${TRAIN_UNIT}" 2>/dev/null; }

assert_no_training_running() {
  training_is_active && die "systemd unit '${TRAIN_UNIT}' is ACTIVE -- a cascade run is already going.
       Inspect: systemctl status ${TRAIN_UNIT}   Stop: sudo systemctl stop ${TRAIN_UNIT}"
  systemctl is-active --quiet glo-nca-training 2>/dev/null \
    && die "the production unit 'glo-nca-training' is ACTIVE on this GPU; stop it first."
  rm -f "${TRAIN_LOCK}"
}

enable_keep_alive() {  # enable_keep_alive <run-id> <config> <data-root> <stop-after-epoch or empty>
  sudo mkdir -p "$(dirname "${KEEPALIVE_FILE}")"
  printf 'KEEP_RUN_ID=%s
KEEP_CONFIG=%s
KEEP_DATA_ROOT=%s
KEEP_STOP_AFTER=%s
KEEP_SLOT=%s
'     "$1" "$2" "$3" "${4:-}" "${SLOT}" | sudo tee "${KEEPALIVE_FILE}" >/dev/null
  # The shared unit file, pointed at this slot's keep-alive file and passing the slot to the hook.
  sed -e "s#/var/lib/glo-nca-cascade/keepalive.env#${KEEPALIVE_FILE}#" \
      -e "s#_autoresume.sh\$#_autoresume.sh ${SLOT}#" \
      "${GLO_WORKSPACE}/cloud/systemd/glo-nca-cascade-autoresume.service" \
    | sudo tee "/etc/systemd/system/${AUTORESUME_UNIT}.service" >/dev/null
  # Boot guard for restarter.sh: an idle VM pauses the restarter and powers off.
  sudo cp "${GLO_WORKSPACE}/cloud/systemd/glo-nca-cascade-idle-guard.service" /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --quiet "${AUTORESUME_UNIT}" glo-nca-cascade-idle-guard
  pass "keep-alive on: after a restart this VM resumes run $1${SLOT:+ (slot ${SLOT})} on its own."
}

write_training_lock() {
  printf 'unit=%s\nstarted=%s\nrun=%s\n' "${TRAIN_UNIT}" "$(date -Is)" "${1:-unknown}" > "${TRAIN_LOCK}"
}
