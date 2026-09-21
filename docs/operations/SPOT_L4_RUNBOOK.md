# Running GLO-NCA on a single Spot L4

Operational runbook for the production training run on one preemptible
(Spot) NVIDIA L4. **No GCP resource has been created by the repository and no
cloud cost has been incurred** — this document describes the intended
provisioning, it does not perform it.

Spot is a *pricing mode*, not a hardware change: the same single L4, at
roughly one third the on-demand price, with the condition that Google may
reclaim the VM at 30 seconds' notice. That trade is only sensible if resume
is reliable, which is why the checkpoint guarantees below matter more than
the price.

---

## 1. Why Spot is safe here

A preemption destroys the VM, not the disk. The run continues from the last
periodic checkpoint, and the following are all restored — verified on real
BraTS data, not asserted:

| State | Restored from |
|---|---|
| model weights | `last.pth` → `model` |
| optimizer (AdamW moments) | `last.pth` → `optimizer` |
| scheduler (cosine position) | `last.pth` → `scheduler` |
| EMA weights | `last.pth` → `ema` |
| epoch / global step | `last.pth` → `epoch` |
| best score and best epoch | `last.pth` → `best_score`, `best_epoch` |
| validation history | `last.pth` → `history` |
| **early-stopping patience** | `last.pth` → `early_stopping` |
| top-3 ranking | `checkpoints/top_k/top_k.json` |
| RNG state | `last.pth` → `rng_state` |

Early-stopping state is the one most easily forgotten. Without it the
patience counter resets on every restart, so a plateaued run trains another
full patience window after each preemption — on a 15-epoch patience that is
a substantial waste of exactly the budget Spot was meant to save.

`logging.checkpoint_frequency: 5` bounds the worst case: a preemption costs
at most five epochs of work. A checkpoint is 0.546 MB and the write was
measured at ~0.01 s, so the frequency is effectively free.

---

## 2. Provisioning

Set in `cloud/config/gcp.env`:

```
PROVISIONING_MODEL=SPOT
```

`cloud/scripts/start_vm.sh` then passes:

```
--machine-type       g2-standard-8
--accelerator        type=nvidia-l4,count=1
--provisioning-model SPOT
--instance-termination-action STOP
```

STOP, not DELETE. On preemption the instance and its boot disk survive, so
`start_vm.sh` restarts the SAME VM and `resume_training.sh` continues from
`last.pth`. DELETE would discard the experiment directory and force a full
re-download from GCS, and would lose anything written since the last sync.

Attach a **persistent disk** for the experiment directory. The boot disk is
deleted with the VM; anything only on the boot disk is lost on preemption.

```
experiments/<run-id>/        <- on the persistent disk
  checkpoints/
    best.pth                 selected model (validation metric)
    last.pth                 full state, written every epoch
    epoch_5.pth, epoch_10.pth, ...   periodic recovery snapshots
    top_k/
      best_1.pth best_2.pth best_3.pth
      top_k.json             ranked manifest with provenance
  reports/
  metrics/
```

---

## 3. Launch

```
python train.py --config configs/glo_nca_production.yaml \
                --experiment-id glo-nca-prod-001 \
                --output /mnt/disks/experiments
```

`training.epochs: 300` is a **maximum budget**, not a target. Early stopping
(monitor = validation mean foreground Dice, patience 15, min_delta 0.001)
ends the run once validation has plateaued. There is no evidence that 300
epochs are necessary.

## 4. After a preemption

```
python train.py --resume /mnt/disks/experiments/glo-nca-prod-001
```

Resume continues toward the original budget. It restores everything in the
table above and logs the epoch it is continuing from — check that line rather
than assuming.

## 5. Continuing past the budget

If the full 300 epochs complete and further training is justified:

```
python train.py --resume /mnt/disks/experiments/glo-nca-prod-001 \
                --extend-to 310 \
                --extension-reason "validation still improving at epoch 300"
```

This is deliberately *not* the same command as `--resume`. An extension is
recorded in `reports/extension.json` so the thesis can state what actually
happened: a run planned for 300 epochs, then explicitly continued.

`--lr-policy` defaults to `freeze`, which holds the learning rate at
`eta_min`. This is not an arbitrary default. `CosineAnnealingLR` is periodic,
so past `T_max` its learning rate **rises again** rather than staying at the
floor — measured on the production schedule:

| epoch | learning rate |
|---|---|
| 300 | 1.00e-05 (eta_min, end of the planned decay) |
| 310 | 1.44e-05 |
| 350 | 1.17e-04 (11.6× eta_min) |

Stepping the saved scheduler onward is therefore a warm restart, and
rebuilding the cosine over a new horizon retroactively changes the curve that
epochs 1–300 actually used. `freeze` does neither.

---

## 6. Cost

At roughly $0.25/h Spot versus $0.70/h on-demand for a single L4. The
per-epoch cost on this hardware has **not been measured** — the only
production-preset timings available are from a power-capped RTX 3050 and a
Kaggle T4, and neither extrapolates reliably to Ada. Run
`time_presets()` (in the standalone) on the L4 for real numbers before
committing to a long run.

---

## 7. What to check before a long run

```
python scripts/audit_training_firewall.py     # test firewall + checkpoint controls
python kaggle_glo_nca_50epoch.py --self-check # architecture identity
```

Both must pass. Neither takes a GPU-minute.
