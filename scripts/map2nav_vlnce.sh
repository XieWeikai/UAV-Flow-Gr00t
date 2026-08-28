#!/usr/bin/env bash
set -euo pipefail

MAP2NAV_NUM_WORKERS=32
processed_root="/data/glx/indoor_data/map2nav/processed_v2"
rxr_annotations="/data/glx/indoor_data/habitat/data/vln_ce/raw_data/rxr/train/train_guide.json.gz"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/2] Converting R2R train -> ${processed_root}/r2r"
"${repo_root}/.venv/bin/python" "${repo_root}/map2nav_vlnce.py" \
  --input-root /data/glx/indoor_data/map2nav/r2r_replay_4_view_2048 \
  --output-root "${processed_root}/r2r" \
  --dataset-name r2r \
  --split train \
  --flat-output \
  --num-workers "${MAP2NAV_NUM_WORKERS}"

echo "[2/2] Converting RxR English train -> ${processed_root}/rxr"
"${repo_root}/.venv/bin/python" "${repo_root}/map2nav_vlnce.py" \
  --input-root /data/glx/indoor_data/map2nav/rxr_replay_guide_4_view_2048 \
  --output-root "${processed_root}/rxr" \
  --dataset-name rxr_guide \
  --split train \
  --rxr-annotations "${rxr_annotations}" \
  --flat-output \
  --num-workers "${MAP2NAV_NUM_WORKERS}"
