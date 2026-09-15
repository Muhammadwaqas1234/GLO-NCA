# REPRODUCIBILITY AUDIT

## 1. Every source of randomness, traced

| Source | Where used | Seeded? | Restored on resume? |
|---|---|---|---|
| Python `random` (parent) | — | YES `reproducibility.py:22` | YES `:46` |
| Python `random` (worker) | `_augment`, `patchify_multimodal` (`Nii_Gz_Dataset_3D.py:266,270,278,289-305,344,354-356`) | YES `runner.py:428` | **NO — reseeded from scratch every epoch (T-03)** |
| NumPy (parent) | `statistics` bootstrap (`seed=cfg.seed`, `runner.py:563`) | YES `:23` | YES `:48` |
| NumPy (worker) | `_augment` noise `:302`, `_elastic` `:319-321` | YES `runner.py:427` | **NO — same as above** |
| torch CPU | shuffle order, model init | YES `:24` | YES `:50` |
| torch CUDA | fire-rate mask `Model_BasicNCA3D.py:172` | YES `:26` | YES (best-effort) `:51-55` |
| DataLoader worker torch seed | — | handled by PyTorch (base_seed + worker_id) | N/A |
| Model stochasticity | fire rate `:172`, dropout 0.1 `:115` | via torch seeds | via torch RNG state |
| Gradient-checkpoint recompute | `Model_BasicNCA3D.py:199-201` | `preserve_rng_state=True` | N/A — no stream drift |

## 2. What IS correct (PASS)

- `set_all_seeds` covers python/numpy/torch/torch-cuda — `reproducibility.py:19-26`.
- Full RNG state captured into **every** checkpoint (`runner.py:497`) and restored
  on resume (`:402`). This is better than most research code.
- `worker_init_fn` seeds **both** `random` and `np.random` — the usual bug
  (seeding only numpy, leaving python `random` identical across all workers) is
  **not** present here. `runner.py:425-428`.
- Gradient checkpointing uses `preserve_rng_state=True`, so it does **not**
  perturb the RNG stream or change outputs. Verified `Model_BasicNCA3D.py:201`.
- `describe()` (`:66-70`) is **honest**: it explicitly states bit-exact
  determinism is NOT enforced and cudnn is left in default mode. No overclaiming.

## 3. FINDING R-01 (P1, CONFIRMED) — worker seeds do not vary by epoch

See TRAINING_LOOP_AUDIT §5 (T-03) for the full trace. Summary: `_worker_init`
derives its seed from `cfg.seed + worker_id` only. Since a new DataLoader spawns
new workers every epoch (`runner.py:434`), the same 4 seeds are re-applied every
epoch, so the augmentation/patch-sampling draw sequence repeats.

This is a **reproducibility-logic defect that reduces augmentation diversity**.
Effect size on final metrics is unmeasured — do not state one.

## 4. FINDING R-02 (P2, CONFIRMED) — worker RNG state is not part of the checkpoint

`capture_rng_state` (`:29-38`) snapshots only the **parent** process RNGs.
Worker RNG state cannot be captured (workers do not exist between epochs). On
resume, workers restart from `cfg.seed + worker_id` regardless of which epoch is
resumed. Given R-01 this is currently consistent — but it means resume is not
bit-exact with respect to augmentation even in principle. Honest documentation
of this limitation is missing.

## 5. FINDING R-03 (P2, CONFIRMED) — cudnn nondeterminism not controlled

`torch.backends.cudnn.deterministic` and `.benchmark` are never set (grep: absent
from all of `src/`). Run-to-run GPU variation is therefore possible. This is a
**deliberate, documented** choice (`reproducibility.py:5-8, 66-70`) trading
determinism for speed. Not a defect — but the thesis must not claim bit-exact
reproducibility. It may correctly claim seeded, resumable, fingerprinted runs.

## 6. Resume behaviour verdict

Epoch indexing is **CORRECT — no off-by-one.** Verified:
- save `epoch=ep+1` (`runner.py:494`) = count of completed epochs
- restore `start_epoch = ck["epoch"]` (`:401`)
- loop `range(start_epoch, epochs)` (`:431`), logging `ep+1`

After completing epoch 5, resume starts the loop at index 5 and logs "ep 6/300".
No duplicated and no skipped epoch.

Resume risks are T-04 (silent scheduler reset) and T-05 (no config identity
check), documented in TRAINING_LOOP_AUDIT.

## 7. Honest reproducibility statement

What this repository **can** truthfully claim:
- Fixed seed, seeded and fingerprinted subject-disjoint split (sha256 verified on
  load, `datasource.load_master_split`).
- Full parent RNG state checkpointed and restored; resume continues the stream.
- Complete config, environment, git state and manifest captured per run.

What it **cannot** claim:
- Bit-exact determinism (cudnn default mode — R-03).
- Bit-exact resume of augmentation (worker RNG — R-02).
- Intended per-epoch augmentation diversity (R-01).
