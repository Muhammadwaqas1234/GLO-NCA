#!/usr/bin/env bash
# =============================================================================
# GLO-NCA cloud shared shell library: config loading, PASS/WARN/FAIL logging,
# gcloud/GCS helpers and safety guards. No secrets.
# =============================================================================
set -euo pipefail

# --- locate repo + cloud dirs regardless of where a script is called from ----
CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${CLOUD_DIR}/.." && pwd)"
CONFIG_FILE="${CLOUD_DIR}/config/gcp.env"

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
  DATA_PREFIX="${DATA_PREFIX:-datasets/brats}"
  EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-experiments}"
  VM_DATA_DIR="${VM_DATA_DIR:-/data}"
  VM_OUT_DIR="${VM_OUT_DIR:-/out}"
  VM_CACHE_DIR="${VM_CACHE_DIR:-/cache}"
  # Dataset-scan threads. Each thread may hold a whole large case in float64
  # (up to ~3 GB for 640x640x392), so the default stays well inside VM RAM.
  VALIDATE_WORKERS="${VALIDATE_WORKERS:-4}"
  VM_WORKSPACE="${VM_WORKSPACE:-/opt/glo-nca}"
  SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
  GCS_ROOT="gs://${GCS_BUCKET}"
  GCS_DATA="${GCS_ROOT}/${DATA_PREFIX}"
  GCS_EXPERIMENTS="${GCS_ROOT}/${EXPERIMENT_PREFIX}"
}

# --- gcloud / gcs tooling ----------------------------------------------------
# Prefer the modern `gcloud storage` CLI; fall back to gsutil if unavailable.
need() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

gcs() {
  if gcloud storage --help >/dev/null 2>&1; then gcloud storage "$@"
  else gsutil "$@"; fi
}
gcs_cp()   { gcs cp "$@"; }
gcs_rsync(){ # gcloud storage rsync <src> <dst> [flags...]
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

confirm() { # confirm "message" -- interactive guard for anything spendy
  local msg="${1:-Proceed?}"
  read -r -p "${msg} [y/N] " ans
  [[ "${ans}" == "y" || "${ans}" == "Y" ]] || die "aborted by user."
}

# --- training concurrency guard ------------------------------------------------
# systemd tracks the real job and is the source of truth; the lock file is only
# a human-readable breadcrumb.
TRAIN_UNIT="glo-nca-training"
TRAIN_LOCK="/tmp/glo-nca-training.lock"

training_is_active() { # 0 = a training job is genuinely running
  systemctl is-active --quiet "${TRAIN_UNIT}" 2>/dev/null
}

# --- image identity guard ------------------------------------------------------
# The container runs the code and config baked into the image, so an image from
# an older commit would silently train a different model. setup_gcp.sh stamps
# glo.commit; every launcher checks it. -c safe.directory avoids git's
# dubious-ownership refusal on a root-owned workspace.
assert_image_matches_repo() {
  local repo="${1:-${VM_WORKSPACE}}" img head
  docker image inspect glo-nca:latest >/dev/null 2>&1     || die "docker image glo-nca:latest missing. Run setup_gcp.sh."
  head="$(git -c safe.directory="${repo}" -C "${repo}" rev-parse HEAD 2>/dev/null)"
  [[ -n "${head}" ]] || die "cannot read git HEAD in ${repo}; cannot verify the image."
  img="$(docker image inspect -f '{{ index .Config.Labels "glo.commit" }}' glo-nca:latest 2>/dev/null)"
  [[ -n "${img}" && "${img}" != "<no value>" ]]     || die "image glo-nca:latest has no glo.commit label (built before this guard existed).
       Run setup_gcp.sh to rebuild it from the current commit."
  [[ "${img}" == "${head}" ]]     || die "STALE IMAGE: built from ${img:0:12}, but the repo is at ${head:0:12}.
       Training on it would run DIFFERENT code and config. Run setup_gcp.sh to rebuild."
  git -c safe.directory="${repo}" -C "${repo}" diff --quiet HEAD --     || die "repo ${repo} has uncommitted changes to tracked files, so the image
       cannot correspond to one commit. Commit or reset them, then run setup_gcp.sh."
  pass "image glo-nca:latest matches repo commit ${head:0:12}"
}

# --- persistent preprocessing cache ------------------------------------------------
# The container's /app/.cache (preprocessing + dataset-validation caches) is
# bind-mounted from VM_CACHE_DIR so `docker run --rm` and Spot restarts keep it.
ensure_cache_dir() {
  if [[ ! -d "${VM_CACHE_DIR}" ]]; then
    sudo mkdir -p "${VM_CACHE_DIR}" && sudo chown "$USER" "${VM_CACHE_DIR}" \
      || die "cannot create cache dir ${VM_CACHE_DIR}"
  fi
  [[ -w "${VM_CACHE_DIR}" ]] || sudo chown "$USER" "${VM_CACHE_DIR}" \
    || die "cache dir ${VM_CACHE_DIR} is not writable"
  pass "persistent cache dir ${VM_CACHE_DIR} (mounted at /app/.cache)"
}

# --- repo Python with the training image's dependencies ---------------------------
# The VM host has no torch / numpy / nibabel / pyyaml; the training image does.
# repo_python runs Python inside glo-nca:latest with the checkout and any given
# host paths bind-mounted at identical paths, so arguments need no translation.
# Without the image it falls back to host Python only if those packages import.
# Usage: repo_python [--gpu] [--ro PATH]... [--rw PATH]... [--env K=V]... -- <python args...>
repo_python() {
  local extra=() envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
      --gpu) extra+=(--gpus all); shift ;;
      --ro)  extra+=(-v "$2:$2:ro"); shift 2 ;;
      --rw)  extra+=(-v "$2:$2"); shift 2 ;;
      --env) extra+=(-e "$2"); envs+=("$2"); shift 2 ;;
      *) die "repo_python: unknown option $1" ;;
    esac
  done
  [[ "${1:-}" == "--" ]] && shift
  if docker image inspect glo-nca:latest >/dev/null 2>&1; then
    docker run --rm --shm-size=8g --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -e GLO_VALIDATE_WORKERS="${VALIDATE_WORKERS:-4}" \
      -v "${REPO_DIR}:${REPO_DIR}:ro" -w "${REPO_DIR}" "${extra[@]+"${extra[@]}"}" \
      --entrypoint python glo-nca:latest "$@"
  else
    local py; py="$(command -v python3 || command -v python || true)"
    [[ -n "${py}" ]] && "${py}" -c "import torch, numpy, nibabel, yaml" 2>/dev/null \
      || die "no glo-nca:latest image, and host Python lacks torch/numpy/nibabel/pyyaml.
       On the VM run setup_gcp.sh (builds the image); elsewhere install those packages."
    (cd "${REPO_DIR}" && env GLO_VALIDATE_WORKERS="${VALIDATE_WORKERS:-4}" "${envs[@]+"${envs[@]}"}" "${py}" "$@")
  fi
}

assert_no_training_running() {
  if training_is_active; then
    die "systemd unit '${TRAIN_UNIT}' is ACTIVE -- training is already running.
       Refusing to start a second job on this GPU.
       Inspect:  systemctl status ${TRAIN_UNIT}
       Stop it:  sudo systemctl stop ${TRAIN_UNIT}"
  fi
  # Not active: any lock file left by a crashed/declined launch is stale.
  if [[ -e "${TRAIN_LOCK}" ]]; then
    warn "removing stale lock ${TRAIN_LOCK} (unit '${TRAIN_UNIT}' is not active)"
    rm -f "${TRAIN_LOCK}"
  fi
}

write_training_lock() { # call ONLY after confirm(), just before systemctl start
  printf 'unit=%s\nstarted=%s\nexperiment=%s\n' \
    "${TRAIN_UNIT}" "$(date -Is)" "${1:-unknown}" > "${TRAIN_LOCK}"
}
