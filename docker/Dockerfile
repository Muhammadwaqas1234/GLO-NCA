# GLO-NCA cascade training image (CUDA 12.1); build from the repo root: docker build -f docker/Dockerfile -t glo-nca-cascade .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Python 3.10 + the libs opencv-python-headless needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# PyTorch (CUDA 12.1) first, cached across rebuilds.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Pinned dependencies; the constraints file locks the transitive set.
COPY docker/requirements-docker.txt docker/constraints-docker.txt ./
RUN python -m pip install -r requirements-docker.txt -c constraints-docker.txt

COPY src/ ./src/
COPY glo_nca/ ./glo_nca/
COPY configs/ ./configs/
COPY train.py ./

RUN python -c "import torch, torchio, nibabel, scipy, cv2, yaml, glo_nca.trainer; print('deps OK, torch', torch.__version__)"

# No default run: a bare `docker run glo-nca-cascade` prints usage.
ENTRYPOINT ["python", "train.py"]
