#!/usr/bin/env bash
# =============================================================================
# GLO-NCA V2 cloud -- shared shell library (sourced by every script).
# Provides: config loading, coloured PASS/WARN/FAIL logging, gcloud/gcs helpers,
# and safety guards. No secrets live here.
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
