#!/usr/bin/env bash
set -euo pipefail

raw_dir="/data/astarue/raw/0711"
vln_output_dir="/home/glx/astarue/processed/0711/vln"
objectnav_output_dir="/home/glx/astarue/processed/0711/objectnav"
num_processes=32

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${vln_output_dir}" "${objectnav_output_dir}"

common_args=(
  --raw_dir "${raw_dir}"
  --num_processes "${num_processes}"
  --skip_invalid_episodes
  --trim_extra_tail_frame
)

uv run python "${repo_root}/ue_astar.py" \
  "${common_args[@]}" \
  --instruction_type objectnav \
  --output_dir "${objectnav_output_dir}"

uv run python "${repo_root}/ue_astar.py" \
  "${common_args[@]}" \
  --instruction_type vln \
  --output_dir "${vln_output_dir}"

