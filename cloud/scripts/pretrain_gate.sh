#!/usr/bin/env bash
# =============================================================================
# GLO-NCA FINAL PRE-TRAINING GATE (run ON THE GCP VM). V2 + V3 aware.
#
# Chains every real check that requires the GPU + real BraTS dataset, in order,
# and prints a single PASS/FAIL gate. Runs NO 200-epoch training. Does NOT start
# A0-Final. It is the one command to prove the infrastructure is training-ready.
#
# Prereqs (Phase 2/3.1): repo checked out at $VM_WORKSPACE, image built
# (setup_gcp.sh), dataset cached locally (cache_dataset.sh or run once),
# cloud/config/gcp.env filled.
#
# SAFETY: this gate can NEVER start the science run. The 2-epoch smoke has no
# fallback -- if it fails, the gate fails (Phase 2, P0).
#
# Usage (on the VM):
#   ./cloud/scripts/pretrain_gate.sh configs/glo_nca_production.yaml
#
# Any other config in configs/ is historical and trains a DIFFERENT
# architecture. Passing one here gates the wrong model.
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

PYBIN="$(command -v python3 || command -v python)"
FAILS=0
step() { echo; echo "### $* ###"; }
mark() { if [ "$1" -eq 0 ]; then pass "$2"; else fail "$2"; FAILS=$((FAILS+1)); fi; }

# Phase 2 (P1 fix): lib.sh sets `set -e`, which this script inherits. That made
# the `cmd; mark $?` idiom dead code -- the script aborted on the FIRST failing
# check and never printed the gate summary, so "BLOCKED (N failures)" was
# unreachable. `check <label> <cmd...>` runs the command with errexit disabled,
# records PASS/FAIL, and lets the gate continue so the operator sees EVERY
# failure in one run. Use this instead of `cmd; mark $?`.
check() {
  local label="$1"; shift
  set +e; "$@"; local rc=$?; set -e
  mark "${rc}" "${label}"
  return 0
}

cd "${REPO_DIR}"

# Config under test. Phase 4: there is deliberately NO DEFAULT. The previous
# default (configs/historical/v3_multilevel_ckpt.yaml) is the FROZEN 32/96/128 thesis
# reference, so a bare `pretrain_gate.sh` silently validated the WRONG
# architecture and reported PASS for a config nobody intended to train.
# Fail closed instead: the caller must name the configuration explicitly.
GATE_CFG="${1:-}"
if [[ -z "${GATE_CFG}" ]]; then
  die "usage: pretrain_gate.sh <config>
       production : ./cloud/scripts/pretrain_gate.sh configs/glo_nca_production.yaml
       reference  : ./cloud/scripts/pretrain_gate.sh configs/historical/v3_multilevel_ckpt.yaml   (FROZEN 32/96/128)
       No default is applied: gating the wrong architecture wastes days of GPU time."
fi
IS_V3=$("${PYBIN}" - "${GATE_CFG}" <<'PY'
import sys, yaml
try:
    raw = yaml.safe_load(open(sys.argv[1]))
    print("1" if str(raw.get("model", {}).get("version","")).lower() == "v3" else "0")
except Exception:
    print("0")
PY
)
# Phase 4: distinguish the GLOBAL-CONTEXT production candidate (level3 absent,
# 48/64) from the frozen 32/96/128 reference. `model.global_context` is exactly
# the key the runner uses to pick build_glo_nca_global_context, so the gate
# follows the same signal rather than guessing from the filename.
HAS_GLOBAL_CTX=$("${PYBIN}" - "${GATE_CFG}" <<'PY'
import sys, yaml
try:
    raw = yaml.safe_load(open(sys.argv[1]))
    print("1" if (raw.get("model", {}) or {}).get("global_context") is not None else "0")
except Exception:
    print("0")
PY
)
log "pre-training gate for config: ${GATE_CFG} (v3=${IS_V3}, global_context=${HAS_GLOBAL_CTX})"

# --- Part 1-3: preflight (GPU/CUDA/PyTorch/dataset/master-split/config/disk) --
step "Part 1-3  GPU + CUDA + dataset preflight"
check "preflight_gcp" "${PYBIN}" scripts/preflight_gcp.py \
    --data-root "${VM_DATA_DIR}" --split split/master_split.json

# --- Part 3: explicit dataset validation (PASS/FAIL, refuses bad data) --------
step "Part 3  real BraTS dataset validation"
check "dataset validation" "${PYBIN}" scripts/validate_dataset.py --root "${VM_DATA_DIR}"

# --- Part 4: master split -- create ONCE if absent, then verify + freeze ------
step "Part 4  master split (create once, then verify)"
if [ ! -f split/master_split.json ]; then
  log "no master split yet -> creating from real dataset (seed 42)"
  "${PYBIN}" scripts/create_master_split.py --data-root "${VM_DATA_DIR}"
fi
check "master split verified" "${PYBIN}" scripts/check_split.py \
    --split split/master_split.json --data-root "${VM_DATA_DIR}"
if [ -f split/master_split.json ]; then
  FP=$("${PYBIN}" - <<PY
import json;print(json.load(open("split/master_split.json"))["split_sha256"])
PY
)
  log "master split fingerprint: ${FP}"
fi

# --- Part 4b (V3 only): model build + parameter report + production memory gate -
if [ "${IS_V3}" = "1" ]; then
  step "Part 4b  V3 model build + parameter report"
  # heredoc cannot pass through `check`; guard errexit explicitly.
  set +e
  "${PYBIN}" - "${GATE_CFG}" <<'PY'
import sys, torch
from src.experiment.config import load_config
from src.models.Model_GLO_NCA_V3 import build_v3_from_config
cfg = load_config(sys.argv[1])
# Phase 4: build the model the RUNNER would actually build. When
# `model.global_context` is present the runner uses the global-context model;
# calling build_v3_from_config unconditionally reported the parameter count of
# an architecture that production never instantiates.
if (cfg.raw.get("model", {}) or {}).get("global_context") is not None:
    from src.models.Model_GLO_NCA_GlobalContext import build_glo_nca_global_context
    m = build_glo_nca_global_context(cfg, 4, 3, torch.device("cpu"))
    n = sum(p.numel() for p in m.parameters())
    geo = [(l.resolution, l.channels, l.nca_steps) for l in m.levels]
    print(f"GLO-NCA (global context) parameters: {n} | levels (res,ch,steps): {geo}")
else:
    m = build_v3_from_config(cfg, 4, 3, torch.device("cpu"))
    pr = m.parameter_report()
    print(f"V3 parameters: {pr['total_parameters']} (by component: {pr['by_level']})")
PY
  V3_BUILD_RC=$?
  set -e
  mark "${V3_BUILD_RC}" "V3 builds + reports parameter count"

  # Phase 4: gpu_memory_gate_v3.py sweeps the LEVEL3 resolution and DERIVES
  # level1/level2 from it (res//4, res*3//4), so it always builds a THREE-level
  # model. The production candidate has level3 disabled (L1 48^3, L2 64^3), and
  # no --resolutions value reproduces that geometry. Running it against the
  # production config would have measured an architecture production never
  # builds. Route by config instead of assuming the frozen reference.
  if [ "${HAS_GLOBAL_CTX}" = "1" ]; then
    step "Part 4c  GLO-NCA production config identity + memory (actual geometry)"
    check "production configuration identity (fail-closed)" \
      "${PYBIN}" scripts/verify_glo_nca_production_config.py "${GATE_CFG}"
  else
    step "Part 4c  V3 frozen-reference GPU memory gate (96^3 + 128^3)"
    check "V3 96^3 + 128^3 GPU memory fit" \
      "${PYBIN}" scripts/gpu_memory_gate_v3.py --config "${GATE_CFG}" --resolutions 96,128
  fi
fi

# --- Part 5-6-13: real-data short training smoke (gate config, tiny) ----------
# Uses the REAL dataset/model/loss/CUDA via the container, at the thesis patch,
# for a couple of epochs on a handful of cases -- proves the full path incl. VRAM
# fit, checkpointing, validation, logging. NOT the science run.
step "Part 5/6/13  real-data short training smoke (${GATE_CFG})"
SMOKE_CFG="/tmp/gate_smoke.yaml"
sed -e 's/^  epochs: .*/  epochs: 2/' \
    -e 's/^  number_of_patients: .*/  number_of_patients: 12/' \
    -e 's/^  smoothing_window: .*/  smoothing_window: 2/' \
    "${GATE_CFG}" > "${SMOKE_CFG}"
# run inside the container on the GPU, writing to a throwaway out dir
# SAFETY (Phase 2, P0): there is NO fallback here. The previous version chained
#   ... --config "${SMOKE_CFG}" || docker run ... --config "${GATE_CFG}"
# so any failure of the 2-epoch smoke silently launched the FULL 300-epoch
# production config on the GPU, with no confirmation -- contradicting this
# script's own header. A gate must never be able to start the science run.
# Smoke fails -> the gate fails. Nothing else runs.
#
# Mounts mirror the PRODUCTION container exactly (see _train_entrypoint.sh) so
# the gate cannot pass on a mount set that production does not use. The split
# now ships inside the image, so no split mount is needed here either.
set +e
docker run --rm --gpus all \
  -v "${VM_DATA_DIR}:/data:ro" -v "${VM_OUT_DIR}:/out" \
  -v "${SMOKE_CFG}:/app/configs/_gate_smoke.yaml:ro" \
  -e DATA_ROOT=/data \
  glo-nca:latest --config configs/_gate_smoke.yaml --output /out
GATE_SMOKE=$?
set -e
mark ${GATE_SMOKE} "real-data training smoke (fwd/bwd/opt/val/ckpt)"
# Experiment dir prefix comes from the config's experiment.name.
EXP_NAME=$("${PYBIN}" - "${GATE_CFG}" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get("experiment", {}).get("name", "GLO-NCA"))
PY
)
SMOKE_EXP=$(ls -td "${VM_OUT_DIR}"/${EXP_NAME}-* 2>/dev/null | head -1)

# --- Part 7: checkpoint round-trip (validate the smoke checkpoint) ------------
step "Part 7  checkpoint round-trip"
if [ -n "${SMOKE_EXP}" ]; then
  check "checkpoint validates + loads" \
    "${PYBIN}" "${CLOUD_DIR}/scripts/validate_checkpoint.py" "${SMOKE_EXP}"
else
  fail "no smoke experiment dir found"; FAILS=$((FAILS+1))
fi

# --- Part 8: GCS round-trip (write, read back) --------------------------------
step "Part 8  GCS artifact round-trip"
if [ -n "${SMOKE_EXP}" ]; then
  # compound && chain cannot pass through `check`; guard errexit.
  set +e
  bash "${CLOUD_DIR}/scripts/sync_experiment.sh" "${SMOKE_EXP}" && \
  gcs_exists "${GCS_EXPERIMENTS}/$(basename "${SMOKE_EXP}")" && \
  gcs cat "${GCS_EXPERIMENTS}/$(basename "${SMOKE_EXP}")/status.json" >/dev/null 2>&1
  GCS_RC=$?
  set -e
  mark "${GCS_RC}" "GCS upload + read-back"
else
  warn "GCS round-trip skipped (no smoke experiment)"
fi

# The round-trip above proves the CURRENT identity can write and read. On the
# VM that identity is the attached service account, which is the credential the
# training job actually uses -- and the one that held only objectViewer when a
# Phase 2 run died on a 403. This additionally proves create/list/DELETE for
# that account, and refuses to report a pass when run as a human.
set +e
bash "${CLOUD_DIR}/scripts/verify_gcs_service_account.sh"
SA_RC=$?
set -e
mark "${SA_RC}" "GCS service-account permissions (create/get/list/delete)"

# --- Part 9-14: repository audits (config integrity, discipline, compile) -----
step "Part 9-14  repository audits"
check "compileall" "${PYBIN}" -m compileall -q src scripts train.py
# verify_phase3_ready validates the V2 ablation matrix + v2 branch; only relevant
# for a V2 gate. Skip it for a V3 gate (V3 has its own model/param/memory checks).
if [ "${IS_V3}" = "1" ]; then
  log "skipping verify_phase3_ready (V2-specific ablation-matrix audit)"
else
  check "verify_phase3_ready" "${PYBIN}" scripts/verify_phase3_ready.py
fi

# --- Gate ---------------------------------------------------------------------
echo; echo "========================================"
if [ "${FAILS}" -eq 0 ]; then
  echo "FINAL PRE-TRAINING GATE: PASS"
  echo "GLO-NCA TRAINING: READY"
  echo "(Then launch A0->A1->A2->A3->Final per PHASE3_RUNBOOK.md -- NOT auto-started.)"
  echo "========================================"
  exit 0
else
  echo "FINAL PRE-TRAINING GATE: BLOCKED (${FAILS} failure(s) above)"
  echo "GLO-NCA TRAINING: NOT READY"
  echo "========================================"
  exit 1
fi
