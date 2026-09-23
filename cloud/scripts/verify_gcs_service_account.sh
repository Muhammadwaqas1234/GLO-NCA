#!/usr/bin/env bash
# Verify the VM SERVICE ACCOUNT can write, read back and delete checkpoint
# objects -- run this ON THE VM, not from a workstation.
#
# WHY THIS EXISTS
#   A Phase 2 run failed with a GCS 403 because the service account held only
#   roles/storage.objectViewer. A workstation test cannot catch that: a
#   developer's own account usually has broader rights, so testing locally
#   proves nothing about what the training job will be allowed to do.
#
#   On the VM the service account is the ambient identity, so this script
#   exercises the true production credential end to end.
#
# It writes a few hundred KB, reads it back, compares checksums and deletes it.
# It never touches a production checkpoint.
#
# Usage (on the VM):
#   bash cloud/scripts/verify_gcs_service_account.sh
# Exit: 0 if the service account can write, read and delete.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

FAIL=0
pass_or_fail() { # name, rc
  if [[ "$2" -eq 0 ]]; then echo "  PASS  $1"; else echo "  FAIL  $1"; FAIL=1; fi
}

echo "=============================================================="
echo "GCS SERVICE-ACCOUNT PERMISSION TEST (run on the VM)"
echo "=============================================================="

# --- identity -----------------------------------------------------------
ACTIVE="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | tr -d '\r')"
echo "  active identity : ${ACTIVE:-<none>}"
if [[ "${ACTIVE}" != *"gserviceaccount.com" ]]; then
  echo
  echo "  REFUSING TO REPORT A PASS."
  echo "  The active identity is not a service account, so this test would"
  echo "  measure a human's permissions rather than the training job's."
  echo "  Run this script ON THE VM (${VM_NAME}), where the attached service"
  echo "  account is the ambient identity."
  exit 1
fi

DEST="${GCS_EXPERIMENTS}/_sa_permission_check/probe.bin"
TMP="$(mktemp -d)"
SRC="${TMP}/probe.bin"
BACK="${TMP}/probe.readback"
trap 'rm -rf "${TMP}"' EXIT

# ~256 KB of deterministic bytes; large enough to be a real transfer.
head -c 262144 /dev/urandom > "${SRC}"
SRC_SHA="$(sha256sum "${SRC}" | cut -d' ' -f1)"
echo "  probe object    : ${DEST}"
echo "  probe sha256    : ${SRC_SHA:0:32}..."
echo

# --- A: write -----------------------------------------------------------
gcs cp "${SRC}" "${DEST}" >/dev/null 2>&1
pass_or_fail "storage.objects.create (write)" $?

# --- B: object exists with the right size -------------------------------
SIZE="$(gcs objects describe "${DEST}" --format='value(size)' 2>/dev/null | tr -d '\r')"
[[ "${SIZE}" == "262144" ]]
pass_or_fail "storage.objects.get (object exists, size ${SIZE:-?})" $?

# --- C: read back and compare -------------------------------------------
gcs cp "${DEST}" "${BACK}" >/dev/null 2>&1
pass_or_fail "storage.objects.get (read back)" $?
BACK_SHA="$(sha256sum "${BACK}" 2>/dev/null | cut -d' ' -f1)"
[[ -n "${BACK_SHA}" && "${BACK_SHA}" == "${SRC_SHA}" ]]
pass_or_fail "checksum MATCH after round trip" $?

# --- D: list (sync_experiment.sh relies on rsync listing) ----------------
gcs ls "${GCS_EXPERIMENTS}/" >/dev/null 2>&1
pass_or_fail "storage.objects.list" $?

# --- E: delete the probe (objectAdmin, not objectCreator) ---------------
gcs rm "${DEST}" >/dev/null 2>&1
pass_or_fail "storage.objects.delete (cleanup)" $?

echo
if [[ "${FAIL}" -eq 0 ]]; then
  echo "  service-account permission: PASS"
  echo "  The training job can sync and recover checkpoints."
else
  echo "  service-account permission: FAIL"
  echo "  Grant the VM service account roles/storage.objectAdmin on"
  echo "  gs://${GCS_BUCKET} before starting a long Spot run, e.g.:"
  echo "    gcloud storage buckets add-iam-policy-binding gs://${GCS_BUCKET} \\"
  echo "      --member=serviceAccount:<VM_SA> --role=roles/storage.objectAdmin"
fi
echo "=============================================================="
exit "${FAIL}"
