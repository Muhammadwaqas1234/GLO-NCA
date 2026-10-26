#!/usr/bin/env bash
# Copy the staged dataset from GCS to this VM and verify it: fetch_dataset.sh [--data-root DIR]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) GLO_DATA_ROOT="${2:?--data-root needs a path}"; shift 2 ;;
    *) die "unknown option: $1" ;;
  esac
done

: "${DATA_PREFIX:?set DATA_PREFIX in gcp.env}"
SRC="gs://${GCS_BUCKET}/${DATA_PREFIX}"
gcs_exists "${SRC}/MANIFEST.txt" || die "no staged dataset at ${SRC}; run stage_dataset.sh first"

sudo mkdir -p "${GLO_DATA_ROOT}" && sudo chown "$(id -u):$(id -g)" "${GLO_DATA_ROOT}"
log "fetching ${SRC} -> ${GLO_DATA_ROOT}"
gcs_rsync -r "${SRC}" "${GLO_DATA_ROOT}"

# Checksums against the manifest written at staging time.
(cd "${GLO_DATA_ROOT}" && sha256sum --quiet -c SHA256SUMS) || die "checksum mismatch in ${GLO_DATA_ROOT}"
expected=$(sed -n 's/^cases=//p' "${GLO_DATA_ROOT}/MANIFEST.txt")
found=$(find "${GLO_DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'BraTS2021_*' | wc -l)
[[ "${found}" == "${expected}" ]] || die "expected ${expected} cases, found ${found}"
pass "dataset ready: ${found} cases at ${GLO_DATA_ROOT}, checksums verified"
