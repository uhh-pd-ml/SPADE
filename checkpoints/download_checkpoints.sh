#!/bin/bash
# Download and unpack the pretrained checkpoints. Run this from the `checkpoints/` directory.
#
# The checkpoints ship as two archives:
#   1. checkpoints.tar.gz              - all models except the x16 Combined baseline
#   2. checkpoints_x16_combined.tar.gz - the x16 Combined checkpoint only (~0.9 GB),
#                                        hosted separately because of its size.
# Both unpack into this directory (spade/<variant>/, combined/<variant>/).
# Skip the large x16 Combined download with:  SKIP_X16_COMBINED=1 ./download_checkpoints.sh
set -euo pipefail


MAIN_URL="https://syncandshare.desy.de/index.php/s/7JwG9qsZ3DYWcWR/download?path=%2F&files=checkpoints_SPADE.tar.gz"
x16_COMBINED_URL="https://syncandshare.desy.de/index.php/s/aHiqfHQXWEojGRE/download?path=%2F&files=checkpoints_x16_combined.tar.gz"
fetch_and_unpack() {
    local url="$1" archive="$2"
    echo "Downloading ${archive} ..."
    curl -L -o "${archive}" "${url}"
    tar -xvf "${archive}"
    rm -f "${archive}"
}

fetch_and_unpack "${MAIN_URL}" checkpoints.tar.gz

if [[ "${SKIP_X16_COMBINED:-0}" == "1" ]]; then
    echo "Skipping x16 Combined (SKIP_X16_COMBINED=1)."
else
    fetch_and_unpack "${x16_COMBINED_URL}" checkpoints_x16_combined.tar.gz
fi
