# GLO-NCA V2 — Docker on GCP

Phase 2 reuses the **Phase 1 `Dockerfile`** unchanged (CUDA 12.1 runtime +
PyTorch 2.5.1). Do not rebuild on a newer CUDA/PyTorch unless there is a real
compatibility problem — the tested stack is:

```
GCP GPU VM  →  NVIDIA driver (from the Deep Learning VM image)
            →  NVIDIA Container Toolkit  →  Docker
            →  CUDA 12.1 (base image)    →  PyTorch 2.5.1+cu121
            →  GLO-NCA V2
```

## Build (on the VM)
```bash
cd /opt/glo-nca
docker build -t glo-nca:latest .
```

## Sanity checks (on a GPU VM)
```bash
nvidia-smi
docker run --rm --gpus all --entrypoint python glo-nca:latest \
    -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
docker run --rm --entrypoint python glo-nca:latest -c "import src.experiment.runner"
docker run --rm glo-nca:latest --help        # train.py CLI
```

## What the container sees
- `/data` — dataset (mounted read-only from the VM's local cache)
- `/out`  — experiments base (mounted read-write; synced to GCS)
- `configs/`, `scripts/`, `src/` — copied in at build time

## Run (normally via `run_training.sh`, but manually for debugging)
```bash
docker run --rm --gpus all \
    -v /data:/data:ro -v /out:/out -e DATA_ROOT=/data \
    glo-nca:latest --config configs/smoke_test.yaml --output /out
```

## Image identity for reproducibility
`_train_entrypoint.sh` records the image **id** and **digest** (when the image
was pushed to a registry; a purely local build has no digest — recorded as
`local-build:no-digest`) into `cloud_metadata.json` inside the experiment dir,
alongside the driver version, VM name, machine type and zone.
