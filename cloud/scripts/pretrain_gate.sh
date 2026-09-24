#!/usr/bin/env bash
# =============================================================================
# GLO-NCA pre-training gate (run on the GCP VM).
#
# Runs every check that needs the GPU and the real dataset and prints one
# PASS/FAIL verdict. It never starts the production run: the smoke trains a
# small master-train subset only and has no fallback.
#
# Prereqs: repo at $VM_WORKSPACE, image built (setup_gcp.sh), dataset cached,
# cloud/config/gcp.env filled. All Python runs inside the verified image.
#
# Usage: ./cloud/scripts/pretrain_gate.sh configs/glo_nca_production.yaml
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

FAILS=0
step() { echo; echo "### $* ###"; }
mark() { if [ "$1" -eq 0 ]; then pass "$2"; else fail "$2"; FAILS=$((FAILS+1)); fi; }

# lib.sh sets -e; `check <label> <cmd...>` runs a command with errexit off and
# records PASS/FAIL, so every failure is reported in one run.
check() {
  local label="$1"; shift
  set +e; "$@"; local rc=$?; set -e
  mark "${rc}" "${label}"
  return 0
}

cd "${REPO_DIR}"

# No default config: the caller must name it explicitly.
GATE_CFG="${1:-}"
if [[ -z "${GATE_CFG}" ]]; then
  die "usage: pretrain_gate.sh <config>
       production: ./cloud/scripts/pretrain_gate.sh configs/glo_nca_production.yaml
       No default is applied: gating the wrong architecture wastes GPU time."
fi
[[ -f "${REPO_DIR}/${GATE_CFG}" ]] || die "config not found in the repository: ${GATE_CFG}
       (give a repo-relative path, e.g. configs/glo_nca_production.yaml)"

# Gate the image production will run; every Python check below runs inside it.
assert_image_matches_repo "${VM_WORKSPACE}"
ensure_cache_dir

cfg_flag() { # print 1/0 for a Python expression over the parsed config `r`
  repo_python -- -c "import sys, yaml; r = yaml.safe_load(open(sys.argv[1])) or {}; \
m = r.get('model') or {}; print('1' if ($1) else '0')" "${GATE_CFG}" 2>/dev/null || echo 0
}
IS_V3=$(cfg_flag "str(m.get('version', '')).lower() == 'v3'")
# model.global_context is the key the runner uses to pick the two-level
# production builder; the gate follows the same signal.
HAS_GLOBAL_CTX=$(cfg_flag "m.get('global_context') is not None")
log "pre-training gate for config: ${GATE_CFG} (v3=${IS_V3}, global_context=${HAS_GLOBAL_CTX})"
[ "${IS_V3}" = "1" ] || die "${GATE_CFG} is not a GLO-NCA (model.version v3) config."

# --- Part 1-3: preflight (GPU/CUDA/PyTorch/dataset/master-split/config/disk) --
step "Part 1-3  GPU + CUDA + dataset preflight"
check "preflight_gcp" repo_python --gpu --ro "${VM_DATA_DIR}" --rw "${VM_OUT_DIR}" \
    --env "OUT_DIR=${VM_OUT_DIR}" -- \
    scripts/preflight_gcp.py --data-root "${VM_DATA_DIR}" \
    --split split/master_split.json --config "${GATE_CFG}"

# --- Part 3: explicit dataset validation (PASS/FAIL, refuses bad data) --------
step "Part 3  real BraTS dataset validation"
check "dataset validation" repo_python --ro "${VM_DATA_DIR}" -- \
    scripts/validate_dataset.py --root "${VM_DATA_DIR}"

# --- Part 4: master split (tracked; verified, never regenerated here) ----------
step "Part 4  master split verification"
check "master split verified" repo_python --ro "${VM_DATA_DIR}" -- \
    scripts/check_split.py --split-json split/master_split.json --data-root "${VM_DATA_DIR}"
FP=$(repo_python -- -c "import json; print(json.load(open('split/master_split.json'))['split_sha256'])" \
     2>/dev/null || echo unknown)
log "master split fingerprint: ${FP}"

# --- Part 4b: model build + parameter report ----------------------------------
step "Part 4b  GLO-NCA model build + parameter report"
BUILD_PY=$(cat <<'PY'
import sys, torch
from src.experiment.config import load_config
from src.models.Model_GLO_NCA_V3 import build_v3_from_config
cfg = load_config(sys.argv[1])
# Build the model the runner would build (global-context builder when configured).
if (cfg.raw.get("model", {}) or {}).get("global_context") is not None:
    from src.models.Model_GLO_NCA_GlobalContext import build_glo_nca_global_context
    m = build_glo_nca_global_context(cfg, 4, 3, torch.device("cpu"))
    n = sum(p.numel() for p in m.parameters())
    geo = [(l.resolution, l.channels, l.nca_steps) for l in m.levels]
    print(f"GLO-NCA (global context) parameters: {n} | levels (res,ch,steps): {geo}")
else:
    m = build_v3_from_config(cfg, 4, 3, torch.device("cpu"))
    pr = m.parameter_report()
    print(f"legacy V3 parameters: {pr['total_parameters']} (by component: {pr['by_level']})")
PY
)
check "GLO-NCA model builds + reports parameter count" \
  repo_python -- -c "${BUILD_PY}" "${GATE_CFG}"

# Production: identity gate on the real two-level geometry. gpu_memory_gate_v3.py
# only builds legacy three-level models, so it runs for legacy configs only.
if [ "${HAS_GLOBAL_CTX}" = "1" ]; then
  step "Part 4c  GLO-NCA production configuration identity"
  check "production configuration identity (fail-closed)" \
    repo_python -- scripts/verify_glo_nca_production_config.py "${GATE_CFG}"
else
  step "Part 4c  legacy V3 GPU memory gate (96^3 + 128^3)"
  check "legacy V3 96^3 + 128^3 GPU memory fit" \
    repo_python --gpu -- scripts/gpu_memory_gate_v3.py --config "${GATE_CFG}" --resolutions 96,128
fi

# --- Part 5-6-13: engineering smoke on a master-train subset -------------------
# make_gate_smoke.py stages 12 master-train cases (8 train / 2 val / 2 "test");
# only that stage is mounted, so the production val/test cases are unreachable.
# Run 1 is hard-killed after epoch 1 (simulated preemption); run 2 resumes it in
# a fresh container with the same persistent cache.
step "Part 5/6/13  engineering smoke (12 master-train cases, kill + resume)"
SMOKE_STAGE="/tmp/glo-nca-gate-smoke"
SMOKE_ID="GLO-NCA-GATE-SMOKE-$(date +%Y%m%d-%H%M%S)"
SMOKE_EXP="${VM_OUT_DIR}/${SMOKE_ID}"
SMOKE_CTR="glo-nca-gate-smoke"
mkdir -p "${SMOKE_STAGE}"
check "smoke subset staged (master-train only)" \
  repo_python --ro "${VM_DATA_DIR}" --rw "${SMOKE_STAGE}" -- \
    scripts/make_gate_smoke.py --data-root "${VM_DATA_DIR}" --stage "${SMOKE_STAGE}" \
    --container-data /data --container-stage /smoke
# Mounts mirror _train_entrypoint.sh, except the data mount is the staged subset.
# --shm-size=8g as in _train_entrypoint.sh (Docker's 64 MB default kills workers).
SMOKE_RUN=(--gpus all --shm-size=8g
  -v "${SMOKE_STAGE}/data:/data:ro" -v "${SMOKE_STAGE}:/smoke:ro"
  -v "${VM_OUT_DIR}:/out" -v "${VM_CACHE_DIR}:/app/.cache" -e DATA_ROOT=/data
  -e GLO_VALIDATE_WORKERS="${VALIDATE_WORKERS}")
CACHE_BEFORE=$(find "${VM_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l)

set +e
docker rm -f "${SMOKE_CTR}" >/dev/null 2>&1
docker run -d --name "${SMOKE_CTR}" "${SMOKE_RUN[@]}" glo-nca:latest \
  --config /smoke/smoke_config.yaml --output /out --experiment-id "${SMOKE_ID}" >/dev/null
KILLED=1; WAITED=0
while [ "${WAITED}" -lt 3600 ]; do
  ROWS=$(( $(cat "${SMOKE_EXP}/metrics/train.csv" 2>/dev/null | wc -l) - 1 ))
  if [ "${ROWS}" -ge 1 ]; then docker kill "${SMOKE_CTR}" >/dev/null 2>&1; KILLED=0; break; fi
  [ "$(docker inspect -f '{{.State.Running}}' "${SMOKE_CTR}" 2>/dev/null)" = "true" ] || break
  sleep 2; WAITED=$((WAITED + 2))
done
docker logs "${SMOKE_CTR}" > /tmp/glo-nca-gate-smoke-run1.log 2>&1
docker rm -f "${SMOKE_CTR}" >/dev/null 2>&1
set -e
mark "${KILLED}" "smoke run 1 reached epoch 1 and was hard-killed (log /tmp/glo-nca-gate-smoke-run1.log)"
CACHE_AFTER=$(find "${VM_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l)
[ "${CACHE_AFTER}" -gt 0 ]
mark $? "preprocessing cache on the host after the container exited (${CACHE_BEFORE} -> ${CACHE_AFTER} entries)"

set +e
docker run --rm "${SMOKE_RUN[@]}" glo-nca:latest --resume "/out/${SMOKE_ID}" \
  > /tmp/glo-nca-gate-smoke-run2.log 2>&1
RESUME_RC=$?
set -e
mark "${RESUME_RC}" "smoke run 2 resumed and completed (log /tmp/glo-nca-gate-smoke-run2.log)"
grep -q "ep 2/" /tmp/glo-nca-gate-smoke-run2.log && ! grep -q "ep 1/" /tmp/glo-nca-gate-smoke-run2.log
mark $? "resume continued from the checkpoint (no restart at epoch 1)"
grep -q "reusing cached PASS" /tmp/glo-nca-gate-smoke-run2.log
mark $? "fresh container reused the persisted dataset-validation cache"
check "smoke run verified (epochs, status, no production val/test ids)" \
  repo_python --ro "${SMOKE_STAGE}" --ro "${SMOKE_EXP}" -- \
    scripts/make_gate_smoke.py --verify "${SMOKE_EXP}" --stage "${SMOKE_STAGE}"

# --- Part 7: checkpoint round-trip (validate the smoke checkpoint) ------------
step "Part 7  checkpoint round-trip"
if [ -d "${SMOKE_EXP}" ]; then
  check "checkpoint validates + loads" \
    repo_python --ro "${SMOKE_EXP}" -- cloud/scripts/validate_checkpoint.py "${SMOKE_EXP}"
else
  fail "no smoke experiment dir found"; FAILS=$((FAILS+1))
fi

# --- Part 8: GCS round-trip (write, read back) --------------------------------
step "Part 8  GCS artifact round-trip"
if [ -d "${SMOKE_EXP}" ]; then
  # compound && chain cannot pass through `check`; guard errexit.
  set +e
  bash "${CLOUD_DIR}/scripts/sync_experiment.sh" "${SMOKE_EXP}" && \
  gcs_exists "${GCS_EXPERIMENTS}/${SMOKE_ID}" && \
  gcs cat "${GCS_EXPERIMENTS}/${SMOKE_ID}/status.json" >/dev/null 2>&1
  GCS_RC=$?
  set -e
  mark "${GCS_RC}" "GCS upload + read-back"
else
  warn "GCS round-trip skipped (no smoke experiment)"
fi

# The VM's service account is the training credential; prove it can create, list
# and delete objects, and refuse to pass when run as a human.
set +e
bash "${CLOUD_DIR}/scripts/verify_gcs_service_account.sh"
SA_RC=$?
set -e
mark "${SA_RC}" "GCS service-account permissions (create/get/list/delete)"

# --- Part 9-14: repository audits ---------------------------------------------
step "Part 9-14  repository audits"
# Parse-only (the checkout is mounted read-only, so nothing is written).
check "all Python parses" repo_python -- -c "import ast, pathlib
files = [p for d in ('src', 'scripts', 'cloud/scripts') for p in pathlib.Path(d).rglob('*.py')]
for p in files + [pathlib.Path('train.py')]: ast.parse(p.read_text(encoding='utf-8'), str(p))
print(f'{len(files) + 1} files parse')"

# --- Gate ---------------------------------------------------------------------
echo; echo "========================================"
if [ "${FAILS}" -eq 0 ]; then
  echo "FINAL PRE-TRAINING GATE: PASS"
  echo "GLO-NCA TRAINING: READY"
  echo "(Launch with ./cloud/scripts/run_training.sh ${GATE_CFG} -- never auto-started.)"
  echo "========================================"
  exit 0
else
  echo "FINAL PRE-TRAINING GATE: BLOCKED (${FAILS} failure(s) above)"
  echo "GLO-NCA TRAINING: NOT READY"
  echo "========================================"
  exit 1
fi
