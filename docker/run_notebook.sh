#!/usr/bin/env bash
set -euo pipefail

DOCKERHUB_USERNAME="henningrose"
IMAGE_NAME="spade"
TAG="${1:-1.0.1}"

docker run --rm -it \
  -p 8888:8888 \
  --gpus all \
  -v "$PWD":/workspace \
  docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG} \
  bash -lc "python -m jupyter lab --ip=0.0.0.0 --no-browser --allow-root"
