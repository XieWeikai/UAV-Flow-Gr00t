from __future__ import annotations

import argparse

from utils.map2nav_indoor import convert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Map2Nav replay data to xnav_indoor format.")
    parser.add_argument("--input-root", required=True, help="Map2Nav aligned replay root, e.g. /mnt/glx/data/map2nav/r2r_replay_4_view_aligned")
    parser.add_argument("--output-root", required=True, help="Output dataset root, e.g. /mnt/glx/data/map2nav/xnav_indoor/r2r")
    parser.add_argument("--dataset-name", required=True, help="Dataset name stored in metadata, e.g. r2r or rxr_guide")
    parser.add_argument("--split", required=True, choices=["train", "val_seen", "val_unseen"], help="Replay split to convert")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for smoke conversions")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of episodes per output chunk")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output split before conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = convert_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        split=args.split,
        max_episodes=args.max_episodes,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    print(f"Wrote xnav_indoor split to {dataset_root}")


if __name__ == "__main__":
    main()
