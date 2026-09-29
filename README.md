# GLO-NCA Cascade

The **Kaggle v7 recipe** of GLO-NCA (Global Context-Aware Neural Cellular Automata
for BraTS brain-tumour segmentation), restructured as a resumable training package
with GCP tooling.

The training code is the Kaggle v7 code (`legacy/kaggle/kaggle_v7.py`, commit `c24c172`),
split into modules with its computation unchanged. `tests/test_equivalence.py`
runs both and checks they give the same losses, validation Dice and test metrics.

## The recipe

| | |
|---|---|
| Model | 2-level NCA cascade (low-res 32×32×24 → high-res 64×64×48), kernels 7 / 3 |
| NCA | 24 channels, hidden 128, 20 + 20 steps, fire rate 0.6, dropout 0.1 |
| Global context | squeeze-and-excitation + spatial global-context block |
| Data | foreground crop, cubic resize to 64×64×48, nonzero z-norm, ET-biased patches |
| Loss | Focal-Tversky + BCE (β 0.75, γ 1.33, CE weight 0.5) |
| Optimiser | AdamW, LR 1.6e-3 → 1e-5 per-step cosine, element-wise gradient clip ±1 |
| Weights | EMA 0.999 (the saved best model) |
| Selection | best validation mean Dice (fixed 0.5 threshold) |
| Budget | 300 epochs (v7 used 150; the budget also sets the cosine LR length) |
| Split | seeded random 70 / 15 / 15 of the case folders |
| Test | plain, then 10× ensemble + test-time flips |

All values live in `configs/glo_nca_cascade.yaml`.

## Layout

```
train.py                  command-line entry point
configs/glo_nca_cascade.yaml   the recipe settings
glo_nca/                  training package
  config.py               YAML -> settings
  data.py                 dataset, data-root discovery, split
  evaluation.py           Dice / mIoU / HD95 evaluation
  trainer.py              training loop, best-model selection, final test
  checkpoint.py           full checkpoints for resume
  reporting.py            final table, results JSON, curves
src/                      model, agent, dataset and loss code (unchanged from v7)
legacy/kaggle/            the original Kaggle scripts (v4-v7)
tests/                    equivalence test against legacy/kaggle/kaggle_v7.py
docker/                   training image
cloud/                    GCP scripts and systemd unit
```

## First check: 10 epochs

Start the full 300-epoch run and pause it after epoch 10 (`--stop-after-epoch 10`).
This checks the pipeline end to end on the real schedule. If the numbers look right,
`--resume` continues the same run from epoch 11; nothing is repeated.

## Train locally

```bash
python train.py --config configs/glo_nca_cascade.yaml --data-root /path/to/BraTS
python train.py --resume experiments/GLO-NCA-CASCADE-<timestamp> --data-root /path/to/BraTS
python train.py --config configs/glo_nca_cascade.yaml --data-root /path/to/BraTS --stop-after-epoch 30
```

Each run writes to `experiments/<run-id>/`: `config.yaml`, `split.json`,
`train.log`, `history.csv`, `status.json`, `last.pth` (every epoch, for resume),
`best.pth` (EMA weights of the best epoch), and at the end `results.json` and
`training_curves.png`.

## Train on GCP

Copy `cloud/config/gcp.env.example` to `cloud/config/gcp.env` and fill it in, then:

```bash
./cloud/scripts/start_vm.sh                      # from your machine
# on the VM:
./cloud/scripts/setup_gcp.sh                     # checkout + image build
./cloud/scripts/run_training.sh --stop-after-epoch 10 --auto-stop   # 10-epoch check, VM stops
./cloud/scripts/resume_training.sh <run-id> --auto-stop           # continue to 300, VM stops
./cloud/scripts/status.sh                        # progress
./cloud/scripts/resume_training.sh <run-id>      # after a preemption
./cloud/scripts/stop_vm.sh                       # from your machine
```

BraTS 2021 (1,251 glioma cases) uses its own config and data folder:

```bash
./cloud/scripts/run_training.sh --config configs/glo_nca_cascade_brats2021.yaml --data-root /data/brats2021 --stop-after-epoch 10 --auto-stop
```

Runs are synced to `gs://<bucket>/experiments/<run-id>/` every 5 minutes and at
the end. A Spot preemption loses at most the current epoch: `last.pth` is written
after every epoch and `resume_training.sh` continues from it.

## Verify

```bash
python tests/test_equivalence.py /path/to/small/BraTS
```

## Credits

Built on the Med-NCA / M3D-NCA framework by John Kalkhof et al. (MIT). See `LICENSE`.
