#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tracking_root="/data4/glx/tracking"
python="${tracking_root}/.venv/bin/python"
work_dir="${tracking_root}/work"

mkdir -p "${work_dir}/tmp"
export TMPDIR="${work_dir}/tmp"
export TEMP="${work_dir}/tmp"
export TMP="${work_dir}/tmp"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

"${python}" "${repo_root}/tracking.py" \
  --jsonl-archive "${tracking_root}/raw/archives/jsonl/seed_101.tar" \
  --frames-dir "${tracking_root}/raw/archives/frames/seed_101" \
  --output-dir "${tracking_root}/processed" \
  --work-dir "${work_dir}" \
  --workers 8 \
  "$@"
