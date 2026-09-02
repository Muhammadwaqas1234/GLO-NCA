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
COPY train.py ./

# Default output location inside the container (override with -e OUT_DIR / -v).
ENV OUT_DIR=/out
# DATA_ROOT has no default: it MUST be provided at run time (mounted volume),
# otherwise train.py falls back to auto-detection under /data.

# Fail fast if CUDA/torch is broken before starting a long training run.
RUN python -c "import torch, torchio, nibabel, scipy, cv2; print('deps OK, torch', torch.__version__)"

ENTRYPOINT ["python", "train.py"]
