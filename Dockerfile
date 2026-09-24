# =============================================================================
# GLO-NCA training image (GPU, CUDA 12.1) for full-dataset training on GCP.
#
# Build:  docker build -t glo-nca .
# Run:    docker run --gpus all --shm-size=8g -e DATA_ROOT=/data \
#             -v /mnt/brats:/data:ro -v /mnt/checkpoints:/out \
#             glo-nca --config configs/glo_nca_production.yaml --output /out
#
# No data is baked into the image; the dataset and outputs are mounted.
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

# --- PyTorch (CUDA 12.1) first, cached across rebuilds -------------------------
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# --- Remaining pinned dependencies ---------------------------------------------
COPY requirements-docker.txt constraints-docker.txt ./
RUN python -m pip install -r requirements-docker.txt -c constraints-docker.txt

# --- Source (respects .dockerignore) -------------------------------------------
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
# The production config requires split/master_split.json; the runner never regenerates it.
COPY split/master_split.json ./split/master_split.json
# Data-quality policy (per-case tolerated stray labels).
COPY split/data_quality_policy.json ./split/data_quality_policy.json
COPY train.py ./

# Read by scripts/preflight_gcp.py; train.py writes to --output (the entrypoint passes /out).
ENV OUT_DIR=/out
# DATA_ROOT has no default; set it at run time (else /kaggle/input, /data or CWD is searched).

# Fail fast if the dependency stack is broken.
RUN python -c "import torch, torchio, nibabel, scipy, cv2, yaml; print('deps OK, torch', torch.__version__)"

# Fail the build unless the split is the canonical one: the fingerprint is
# recomputed from the case ids (load_master_split) and must equal the expected
# value, with 898 / 200 / 198 cases.
ARG EXPECTED_SPLIT_SHA256=d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d
RUN python -c "import sys; from src.experiment.datasource import load_master_split; \
m = load_master_split('split/master_split.json'); \
fp = m['split_sha256']; n = (len(m['train']), len(m['validation']), len(m['test'])); \
ok = fp == '${EXPECTED_SPLIT_SHA256}' and n == (898, 200, 198); \
print(('split VERIFIED ' if ok else 'SPLIT MISMATCH ') + fp, n); sys.exit(0 if ok else 1)"

# No default config: a bare `docker run glo-nca` fails with argparse usage.
# configs/glo_nca_production.yaml is the only production configuration:
#   two-level GLO-NCA, 128^3 working volume, L1 48^3 + L2 64^3, 15+15 steps,
#   SE + spatial global context (k=7), deep supervision, warmup 3 epochs,
#   small-lesion sampling, patchify off, bf16; 30,209 inference / 30,284 training params.
# Historical configs are excluded from the image by .dockerignore.
#
#   docker run ... glo-nca:latest --config configs/glo_nca_production.yaml --output /out
#   docker run ... glo-nca:latest --resume /out/<experiment-id>
ENTRYPOINT ["python", "train.py"]
