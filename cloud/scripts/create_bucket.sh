#!/usr/bin/env bash
# Create the GCS bucket (idempotent) with the standard layout markers.
# Never deletes anything.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

if gcs_exists "${GCS_ROOT}"; then
  pass "bucket already exists: ${GCS_ROOT}"
else
  log "creating bucket ${GCS_ROOT} in ${GCP_REGION} (standard, uniform access)"
  gcs buckets create "${GCS_ROOT}" \
      --project="${GCP_PROJECT_ID}" \
      --location="${GCP_REGION}" \
      --default-storage-class=STANDARD \
      --uniform-bucket-level-access
  pass "bucket created: ${GCS_ROOT}"
fi

# Create prefix "folders" via placeholder objects (GCS has no real dirs).
for p in "${DATA_PREFIX}" "${EXPERIMENT_PREFIX}" "logs" "backups"; do
  echo "placeholder" | gcs cp - "${GCS_ROOT}/${p}/.keep" >/dev/null 2>&1 || true
done
log "layout:"
log "  ${GCS_DATA}/           (dataset -- durable master copy)"
log "  ${GCS_EXPERIMENTS}/    (experiment outputs)"
log "  ${GCS_ROOT}/logs/      (sync/system logs)"
pass "bucket layout ready"
