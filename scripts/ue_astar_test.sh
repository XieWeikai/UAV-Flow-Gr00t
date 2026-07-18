#!/usr/bin/env bash
set -euo pipefail

raw_root="/data/astarue/raw/test"
output_root="/data/astarue/processed/test"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${repo_root}"

for instruction_type in objectnav vln; do
  for split in val_seen val_unseen; do
    uv run python "${repo_root}/ue_astar.py" \
      --raw_dir "${raw_root}/${split}" \
      --output_dir "${output_root}/${instruction_type}/${split}" \
      --instruction_type "${instruction_type}" \
      --evaluation_split "${split}" \
      --num_processes 32 \
      --skip_invalid_episodes \
      --trim_extra_tail_frame
  done
done
