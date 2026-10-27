#!/usr/bin/env bash
# Before a training container starts: wait until the GPU is usable inside a container (the driver can lag
# after a boot). A lock serialises concurrent starts, and each start holds it 30 s after the check, so
# training slots never initialise the GPU at the same moment.
exec 9>/run/glo-nca-gpu-start.lock
flock 9
for _ in $(seq 1 30); do
  # The image's entrypoint is train.py, so the check runs python explicitly.
  if docker run --rm --gpus all --entrypoint python glo-nca-cascade:latest \
       -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    sleep 30
    exit 0
  fi
  sleep 10
done
echo "[wait-gpu] GPU not usable in a container after 5 minutes" >&2
exit 1
