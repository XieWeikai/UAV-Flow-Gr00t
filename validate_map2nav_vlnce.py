from __future__ import annotations

import argparse
import json

from utils.map2nav_vlnce.validator import validate_delivery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the complete R2R and RxR guide Map2Nav VLN-CE delivery."
    )
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--r2r-input-root", required=True)
    parser.add_argument("--rxr-input-root", required=True)
    parser.add_argument("--hash-sample-size", type=int, default=32)
    parser.add_argument("--decode-video-sample-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_delivery(
        processed_root=args.processed_root,
        r2r_input_root=args.r2r_input_root,
        rxr_input_root=args.rxr_input_root,
        hash_sample_size=args.hash_sample_size,
        decode_video_sample_size=args.decode_video_sample_size,
    )
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

