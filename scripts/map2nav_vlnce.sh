#!/usr/bin/env bash
set -euo pipefail

# Edit these four values for the one split you want to convert.
input_split="/mnt/glx/data/map2nav/r2r_replay_4_view_2048/train"
dataset_name="r2r"          # r2r or rxr_guide
output_root="/mnt/glx/data/map2nav/processed/r2r"
num_workers=8

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_root="$(dirname "${input_split}")"
split="$(basename "${input_split}")"

uv run python "${repo_root}/map2nav_vlnce.py" \
  --input-root "${input_root}" \
  --output-root "${output_root}" \
  --dataset-name "${dataset_name}" \
  --split "${split}" \
  --num-workers "${num_workers}"
