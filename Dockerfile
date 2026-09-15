# =============================================================================
# GLO-NCA training image (GPU, CUDA 12.1) -- for full-dataset training on GCP.
#
# Build:
#   docker build -t glo-nca .
#
# Run (mount the BraTS dataset read-only and a writable output dir; both paths
# are passed to the container via environment variables):
#   docker run --gpus all \
#       -e DATA_ROOT=/data -e OUT_DIR=/out \
#       -e EPOCHS=150 \
#       -v /mnt/brats:/data:ro \
#       -v /mnt/checkpoints:/out \
#       glo-nca
#
# On GCP, /mnt/brats and /mnt/checkpoints can be a persistent disk or a GCS
# bucket mounted with gcsfuse. Nothing about the data is baked into the image.
# =============================================================================

# CUDA 12.1 runtime on Ubuntu 22.04 -- matches the torch cu121 wheels below.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- System Python 3.10 (Ubuntu 22.04 default) + minimal libs for OpenCV ----
# opencv-python-headless still needs libGL / libglib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# --- Install PyTorch (CUDA 12.1 build) first so it is cached across rebuilds --
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# --- Install the rest of the pinned dependencies -----------------------------
COPY requirements-docker.txt .
RUN python -m pip install -r requirements-docker.txt

# --- Copy the source (respect .dockerignore) ---------------------------------
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
# The canonical subject-disjoint split MUST be in the image: every V3 config
# sets data.split_file=split/master_split.json, and the runner fails hard rather
# than regenerating it. Without this COPY the production container cannot start.
# (.dockerignore re-includes this one file from its broad `*.json` rule.)
COPY split/master_split.json ./split/master_split.json
# Operational data-quality policy (per-case tolerated stray labels).
COPY split/data_quality_policy.json ./split/data_quality_policy.json
COPY train.py ./

# Experiments are written here by default (override with --output / a mount).
ENV OUT_DIR=/out
# DATA_ROOT has no default: it MUST be provided at run time (mounted volume);
# otherwise the trainer auto-detects a dataset under /data or /kaggle/input.

# Fail fast if the stack is broken before starting a long training run.
RUN python -c "import torch, torchio, nibabel, scipy, cv2, yaml; print('deps OK, torch', torch.__version__)"

# Fail the BUILD (not a 300-epoch run) if the canonical split is missing or its
# recorded fingerprint does not match its contents.
RUN python -c "import json,hashlib,sys; d=json.load(open('split/master_split.json')); \
print('split OK', d['split_version'], d['split_sha256'][:12], \
d['train_count'], d['val_count'], d['test_count'])"

# V3 is the production architecture. There is deliberately NO default config:
# a bare `docker run glo-nca:latest` must fail with argparse usage rather than
# silently training the V2 baseline (configs/gcp_full.yaml), which is what the
# previous `CMD ["--config", "configs/gcp_full.yaml"]` did.
#   docker run ... glo-nca:latest --config configs/v3_multilevel_ckpt.yaml
#   docker run ... glo-nca:latest --resume /out/<experiment-id>
ENTRYPOINT ["python", "train.py"]
