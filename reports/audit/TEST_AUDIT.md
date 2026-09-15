# TEST SUITE AUDIT

## 1. Inventory

| File | Kind | Covers | Needs |
|---|---|---|---|
| `scripts/test_preflight_logic.py` | synthetic unit | 6 fail-closed discovery cases | CPU |
| `scripts/test_patchify_equivalence.py` | equivalence | patchify short-circuit output + RNG-draw accounting | CPU |
| `scripts/validate_v3_local.py` | integration smoke | real V3 model + loss, synthetic data | CPU/GPU |
| `scripts/local_gpu_smoke_test.py` | integration smoke | real **V2** model + loss | GPU |
| `scripts/gpu_memory_gate_v3.py` | hardware gate | 96³/128³ TRUE FIT / SPILL / OOM | GPU |
| `scripts/validate_dataset.py` | data gate | real dataset PASS/FAIL | any |
| `scripts/check_split.py` | data invariant | disjointness, fingerprint, subject leakage | any |
| `scripts/validate_kaggle_sample.py` | data gate | 100-case sample | any |
| `scripts/verify_phase3_ready.py` | static audit | config matrix + source string audits | any |
| `scripts/preflight_gcp.py` | composite gate | GPU+data+split+config+disk | VM |
| `cloud/scripts/validate_checkpoint.py` | artifact validator | checkpoint resume-state keys | any |
| `cloud/scripts/verify_gcp.sh` | env check | gcloud/Docker/CUDA | any |
| `cloud/scripts/pretrain_gate.sh` | meta-gate | chains the above | VM |

## 2. Coverage-by-behaviour matrix

| AREA | TESTED | HOW | GAP |
|---|---|---|---|
| Dataset discovery | PARTIAL | `test_preflight_logic.py` (synthetic) | tests a **reimplementation**, not production |
| Dataset validation | YES | `validate_dataset.py` on real data | — |
| Split integrity | **YES (strong)** | `check_split.py` — disjointness, sha256, subject-level leakage | — |
| Patchify | PARTIAL | `test_patchify_equivalence.py` | tests a **copy**; D-02 no-op behaviour not asserted |
| Augmentation | **NO** | — | no label-alignment test; no RNG-diversity test (would have caught T-03/R-01) |
| Normalisation | **NO** | — | no known-answer test |
| V3 model | PARTIAL | `validate_v3_local.py` (synthetic), `gpu_memory_gate_v3.py` | no numerical correctness test of multi-level fusion; no shape/param assertion in CI |
| Gradient checkpointing equivalence | **NO** | — | **nothing asserts ckpt-ON == ckpt-OFF outputs/gradients.** Highest-value missing model test |
| Loss | **NO** | — | no known-answer FocalTversky test; no empty-region-branch test |
| Training loop | **NO** | — | no test of step order, grad clip, EMA timing |
| **Resume** | PARTIAL | `validate_checkpoint.py` checks *keys* only | **nothing tests that `--resume` actually continues correctly** — would have caught T-04 |
| Reproducibility | **NO** | — | no same-seed-twice test — would have caught R-01 |
| Metrics | **NO** | substring search in `verify_phase3_ready.py:114-116` | **no known-answer Dice/IoU/HD95 test** — would have caught the empty-empty 0.0-vs-1.0 inconsistency (E-04) |
| Config schema | PARTIAL | `verify_phase3_ready.py` (V2 only) | no V3 config-matrix test; unknown keys never rejected |
| Checkpoint I/O | PARTIAL | key presence | no round-trip restore test |
| Cloud / GCS sync | **NO** | — | `.tmp` atomicity asserted in a comment, never verified |
| **Container production path** | **NO** | — | **nothing runs the production mount set** — this is why CL-01 went undetected |
| Real-data end-to-end | PARTIAL | `pretrain_gate.sh` smoke | uses a **different mount set** than production |

## 3. Structural findings

**TS-01 (P1, CONFIRMED) — no test framework.** No `pytest`, no `tests/`, no CI
config, no `conftest.py`, no test dependency in `requirements-docker.txt`. Every
"test" is a hand-rolled `main()` returning an exit code, invoked manually. There
is no way to run the suite as a suite, and no evidence any of it runs
automatically.

**TS-02 (P1, CONFIRMED) — both `test_*` files test reimplementations, not
production code.** `test_preflight_logic.py:21` defines a "structural subset"
copy; `test_patchify_equivalence.py:48` defines a "faithful copy" of
`patchify_multimodal`. Both will keep passing after the production code diverges.
Worse, `test_patchify_equivalence.py:11` **explicitly claims** it "imports the
real dataset method" — the docstring contradicts the code. **These tests can pass
while production is broken**, which is exactly the failure mode the audit brief
asked about.

**TS-03 (P2, CONFIRMED) — `verify_phase3_ready.py` substitutes string search for
testing.** `:114-116` audits "HD95 in voxels" by checking the source text for
`hd95` and the absence of `mm`. That is not a test of the metric, and it is
fragile (the substring `mm` inside any word fails it). Presented as a
correctness gate; it is not one.

**TS-04 (P2, CONFIRMED) — `verify_phase3_ready.py:86` asserts `branch == "v2"`**,
so it **always FAILs** on `v3-multilevel`. `pretrain_gate.sh:144-148` works
around it by skipping it for V3 — but `preflight_gcp.py:92-100` still imports its
V2 config checkers and runs them unconditionally as critical checks (see
CLOUD_AUDIT).

**TS-05 (P3, CONFIRMED) — pytest-discovery false green.** If pytest is ever
added, it will collect both `test_*.py` files, find no `test_` functions (both
use `main()`), and report zero tests collected — a silent pass.

## 4. The highest-value missing tests

1. **Production container path** — `docker run` with the *production* mount set
   against a 1-epoch config. Would have caught CL-01 (P0).
2. **Gradient-checkpointing equivalence** — assert ckpt-ON and ckpt-OFF produce
   identical loss and gradients for a fixed seed. Directly underwrites the
   "checkpointing is safe" claim the thesis depends on.
3. **Resume continuity** — train 2 epochs, resume, assert epoch/LR/optimizer
   state continue correctly. Would have caught T-04 (P1).
4. **Same-seed reproducibility** — two short runs, same seed, assert identical
   loss curve. Would have caught R-01 (P1).
5. **Known-answer metrics** — hand-computed Dice/IoU/HD95 on tiny arrays. Would
   have caught E-04.
