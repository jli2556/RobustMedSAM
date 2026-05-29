#!/usr/bin/env bash
# Generate degraded counterparts for each split.
#
# Expects clear medical images (exported from MedSegBench) laid out as:
#   all_data/<split>/clear/*.jpg
#   all_data/<split>/masks/*.npy
# For every split, augment.py writes one sub-folder per degradation type
# (gauss_noise, gaussian_blur, ... step_motion) next to `clear`.

set -euo pipefail

for split in train val test; do
    echo " "
    echo " #############################################################"
    echo "=> Generating degraded images for '${split}' split <=="
    echo " #############################################################"
    python augment.py --data_dir "all_data/${split}"
done
