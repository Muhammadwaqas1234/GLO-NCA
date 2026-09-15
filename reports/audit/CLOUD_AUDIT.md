# CLOUD / GCP AUDIT

**No cloud command was executed.** Read-only source inspection only.

## 1. Credentials — CLEAN (PASS)

`git ls-files cloud/config/` returns only `gcp.env.example`. The live
`cloud/config/gcp.env` is git-ignored (`.gitignore:40`) and contains only a
project ID, bucket name, machine/GPU type and paths — no keys or tokens.
Authentication is via VM Application Default Credentials
(`start_vm.sh:34 --scopes=cloud-platform`), never service-account JSON.
`gcp.env.example:7-9` explicitly mandates this. **No committed secrets.**

## 2. FINDING CL-01 (P0, CONFIRMED) — the master split cannot reach the container

Independently verified:

- `Dockerfile:48-51` copies `src/`, `configs/`, `scripts/`, `train.py`.
  **`split/` is NOT copied.**
- `.dockerignore:34` excludes `*.json`, which would block
  `split/master_split.json` even if a `COPY split/` were added.
- `_train_entrypoint.sh:54-61` (the production run path) mounts only
  `-v "${DATA}:/data:ro" -v "${OUT}:/out"`. **No `split/` mount.**

Every V3 config sets `data.split_file: split/master_split.json`
(`v3_multilevel.yaml:25`). `runner.py:304-305` tries the path then
`_HERE_ROOT/split_file`; both resolve inside `/app`, where the file does not
exist. `datasource.load_master_split` then raises — training **fails at startup**.

**It is not a silent-wrong-split risk** (the loader fails loudly, which is good
design), but it means **the production cloud training path cannot start at all**
as currently configured.

**Aggravating factor:** `pretrain_gate.sh:103` *does* mount
`-v "${REPO_DIR}/split:/app/split:ro"`. So **the gate passes while production
fails** — the gate does not exercise the production mount set.

## 3. FINDING CL-02 (P0, CONFIRMED) — the gate's fallback can launch the full campaign

`cloud/scripts/pretrain_gate.sh:99-108`, verified verbatim:

```bash
docker run --rm --gpus all ... --config "${SMOKE_CFG}" --output /out \
  || docker run --rm --gpus all ... --config "${GATE_CFG}" --output /out
```

`SMOKE_CFG` is a sed-reduced 2-epoch/12-patient copy. `GATE_CFG` is the
**unmodified production config — `epochs: 300`**.

If the smoke invocation fails for *any* reason, the `||` branch launches the
**full 300-epoch thesis campaign on the GPU**, with:
- **no `confirm` prompt** (`pretrain_gate.sh` never calls `confirm` anywhere), and
- in direct contradiction of the script's own header (`:6`: "Runs NO 200-epoch
  training").

The most likely trigger is exactly CL-01: the fallback branch **drops the
`split/` mount**, and it also drops the `-v /tmp:/tmp:ro` mount that `SMOKE_CFG`
lives behind — so the first command failing and the second running is a very
plausible sequence.

**This is the single most dangerous line in the repository.** A script the
operator reasonably believes is a safe pre-flight gate can start a multi-day
billed run without asking.

## 4. FINDING CL-03 (P1, CONFIRMED) — `set -e` defeats the gate-accumulation idiom

`lib.sh:7` sets `set -euo pipefail`, and every cloud script sources `lib.sh`
without setting its own options. The `mark $?` pattern used throughout
`pretrain_gate.sh` (`:45,50,60,81,85,123,134`) and `verify_results.sh`
(`:33,57,58,62,72`) is therefore **dead**: under `set -e` the script aborts on
the first failing check instead of recording it.

Consequence: the intended "BLOCKED (N failures)" summary
(`pretrain_gate.sh:159-161`) is **unreachable** for any check using `mark $?`.
The gate aborts with a bare non-zero exit and no summary.

Note the interaction with CL-02: line 108's `||` makes the compound command
*succeed*, which is precisely why `GATE_SMOKE=$?` at `:109` does work — and why
the dangerous fallback is reachable while the safe reporting path is not.

## 5. FINDING CL-04 (P1, CONFIRMED) — `setup_gcp.sh` provisions the V2 branch

`setup_gcp.sh:31-37` hard-codes branch `v2` (fetch/checkout and `clone -b v2`).
A VM provisioned by this script gets **V2 code** — without
`Model_GLO_NCA_V3.py`, `Agent_GLO_NCA_V3.py` or any `configs/v3_*.yaml` (all of
which are untracked anyway; see F-01). A V3 run from such a VM cannot work.

## 6. FINDING CL-05 (P1, CONFIRMED) — sync watcher can target the wrong directory

`_train_entrypoint.sh:35` selects the sync target via `ls -t "${OUT}" | head -1`
— the most recently *modified* entry in `/out`. If any other directory there is
touched later (a previous experiment, or the gate's throwaway output), the live
experiment is **never pushed to GCS**. Errors are silenced by `>/dev/null 2>&1`
(`:39`). Baseline checkpoint-loss window is `SYNC_INTERVAL_SECONDS=300`.

Related (P2): the final sync is wrapped in `if [[ -d "${EXP_DIR}" ]]`
(`:78-95`); if `EXP_DIR` fails to resolve, the final sync is skipped silently.

## 7. FINDING CL-06 (P1, CONFIRMED) — VM survives a failed run; lock is ineffective

- `verify_results.sh:62` requires `status.json` state `completed`;
  `delete_vm.sh:32-35` dies if verification fails. **A crashed run's GPU VM keeps
  billing** unless the operator knows to pass `--force` (`delete_vm.sh:24`).
- `start_vm.sh:17` **starts a stopped GPU VM with no `confirm`** (only creation
  is guarded, `:24`). Billing resumes on one command.
- `run_training.sh:57` and `resume_training.sh:59` write `$$` — the *launcher
  shell's* PID — into the lock file, but training runs under **systemd**. The
  launcher exits immediately, so the recorded PID is dead and the guard
  (`run_training.sh:22-25`) reports "not running" while training is live.
  **A second concurrent run can be started on the same GPU.**

## 8. What is well designed (PASS — do not change)

- **`glo-nca-training.service`** — `Restart=no` (`:18`) with an explicit
  anti-loop rationale (`:4-7`), `TimeoutStopSec=120` (`:20`) to let a checkpoint
  write finish, and `_train_entrypoint.sh:99` propagating the true exit code.
  **No restart-loop hazard. This is the best-designed piece of the cloud stack.**
- **`validate_checkpoint.py`** — fail-closed, non-destructive, never repairs a
  corrupt file, prefers `last.pth` with a periodic fallback.
- **`cache_dataset.sh:19-23`** and **`upload_dataset.sh:17-21`** — both validate
  fail-closed *before* proceeding. Correct ordering.
- **`sync_experiment.sh:4-7`** — `.tmp` exclusion, correctly reasoned against the
  atomic write-then-rename in `checkpoint.save_checkpoint`.
- **`gpu_memory_gate_v3.py`** — returns exit **2** when CUDA is absent rather
  than faking a pass (`:50`), and refuses to suggest shrinking the architecture
  (`:126`). The strongest anti-assumption discipline in the repo.

## 9. Script classification

| ACTIVE (production) | `_train_entrypoint.sh`, `run_training.sh`, `resume_training.sh`, `cache_dataset.sh`, `sync_experiment.sh`, `glo-nca-training.service`, `Dockerfile` |
|---|---|
| **OPTIONAL / diagnostic** | `monitor.sh`, `status.sh`, `verify_gcp.sh`, `download_experiment.sh`, `cost_report.sh` |
| **DANGEROUS** | **`pretrain_gate.sh` (CL-02 — can launch the full campaign unprompted)**; `start_vm.sh` (no confirm on restart) |
| **OBSOLETE / stale** | `setup_gcp.sh` (branch `v2`), `verify_phase3_ready.py` (asserts branch `v2` → always FAIL on this branch) |
| **UNUSED** | `xfer2.sh` (zero references), `scripts/dataset_identity.py` (zero references) |

Nothing deleted. Classification only.
