#!/usr/bin/env bash
# Spot restarter: a Cloud Scheduler job that starts the VM every 10 minutes (a no-op while it runs), so a
# preempted VM resumes its keep-alive runs without a local watchdog: restarter.sh create|pause|resume|delete|status
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config
require_gcloud_auth

GCP_REGION="${GCP_REGION:-${GCP_ZONE%-*}}"
JOB="${RESTARTER_JOB:-${VM_NAME}-restarter}"
LOC=(--location "${GCP_REGION}" --project "${GCP_PROJECT_ID}")
URI="https://compute.googleapis.com/compute/v1/projects/${GCP_PROJECT_ID}/zones/${GCP_ZONE}/instances/${VM_NAME}/start"
# The VM's service account: allowed to start the VM, and used by the VM to pause this job when idle.
SA="${RESTARTER_SA:-$(gcloud compute instances describe "${VM_NAME}" $(vm_flags) --format='value(serviceAccounts[0].email)')}"

case "${1:-status}" in
  create)
    [[ -n "${SA}" ]] || die "no service account found on ${VM_NAME}"
    if gcloud scheduler jobs describe "${JOB}" "${LOC[@]}" >/dev/null 2>&1; then
      gcloud scheduler jobs resume "${JOB}" "${LOC[@]}" >/dev/null
      pass "restarter ${JOB} exists; resumed"
    else
      gcloud scheduler jobs create http "${JOB}" "${LOC[@]}" --schedule "*/10 * * * *" --time-zone UTC \
        --uri "${URI}" --http-method POST --oauth-service-account-email "${SA}" \
        --attempt-deadline 60s --description "Start ${VM_NAME} after a Spot preemption" >/dev/null
      pass "restarter ${JOB} created: starts ${VM_NAME} every 10 minutes if it is stopped"
    fi ;;
  pause)  gcloud scheduler jobs pause "${JOB}" "${LOC[@]}" >/dev/null && pass "restarter ${JOB} paused" ;;
  resume) gcloud scheduler jobs resume "${JOB}" "${LOC[@]}" >/dev/null && pass "restarter ${JOB} resumed" ;;
  delete) gcloud scheduler jobs delete "${JOB}" "${LOC[@]}" --quiet >/dev/null && pass "restarter ${JOB} deleted" ;;
  status) gcloud scheduler jobs describe "${JOB}" "${LOC[@]}" --format="value(name.basename(),state,schedule,lastAttemptTime,status.code)" ;;
  *) die "usage: restarter.sh create|pause|resume|delete|status" ;;
esac
