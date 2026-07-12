from __future__ import annotations

import argparse

from utils.map2nav_vlnce import convert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert single-floor Map2Nav VLN-CE replay data to stable xNav data."
    )
    parser.add_argument("--input-root", required=True, help="Replay root containing split folders")
    parser.add_argument("--output-root", required=True, help="Dataset root; the split is appended")
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=["r2r", "rxr_guide"],
        help="Stable source dataset name",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val_seen", "val_unseen"],
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional number of accepted single-floor episodes to write",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of concurrent episode writers; output order remains manifest-deterministic",
    )
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
        num_workers=args.num_workers,
    )
    print(f"Wrote map2nav_vlnce split to {dataset_root}")


if __name__ == "__main__":
    main()
