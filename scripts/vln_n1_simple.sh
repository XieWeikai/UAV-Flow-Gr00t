# !/bin/bash
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data/InternData-N1/hm3d_d435i --output_dir /data/InternData-N1/exp --roll_limit 5.0
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data/InternData-N1/hm3d_zed --output_dir /data/InternData-N1/exp --roll_limit 5.0

uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/3dfront_d435i --output_dir /data/InternData-N1/exp --roll_limit 5.0
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/3dfront_zed --output_dir /data/InternData-N1/exp --roll_limit 5.0

uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/gibson_d435i --output_dir /data/InternData-N1/exp --roll_limit 5.0
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/gibson_zed --output_dir /data/InternData-N1/exp --roll_limit 5.0

uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/hssd_d435i --output_dir /data/InternData-N1/exp --roll_limit 5.0
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/hssd_zed --output_dir /data/InternData-N1/exp --roll_limit 5.0

uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/matterport3d_d435i --output_dir /data/InternData-N1/exp --roll_limit 5.0
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/matterport3d_zed --output_dir /data/InternData-N1/exp --roll_limit 5.0

uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/replica_d435i --output_dir /data/InternData-N1/exp --roll_limit 5.0
uv run vln_n1_v2.py --num_processes 32 --raw_dir /data-10T/InternData-N1/replica_zed --output_dir /data/InternData-N1/exp --roll_limit 5.0

