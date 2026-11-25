#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./docker/build_sif.sh latest
#   ./docker/build_sif.sh 1.0.1

TAG="${1:-1.0.1}"

DOCKERHUB_USERNAME="henningrose"
IMAGE_NAME="spade"

IMAGE="docker://docker.io/${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"

OUTDIR="${OUTDIR:-/home/${USER}/images}"
mkdir -p "${OUTDIR}"

if [[ "${TAG}" == "latest" ]]; then
  OUT="${OUTDIR}/${IMAGE_NAME}-latest.sif"
else
  OUT="${OUTDIR}/${IMAGE_NAME}-${TAG}.sif"
fi

echo "Building SIF from: ${IMAGE}"
echo "Output: ${OUT}"

# Use apptainer if available, otherwise singularity
if command -v apptainer >/dev/null 2>&1; then
  apptainer build "${OUT}" "${IMAGE}"
else
  singularity build "${OUT}" "${IMAGE}"
fi

echo "Done."
