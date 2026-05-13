from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
GROUNDING_SAM_DIR = ROOT_DIR / "Grounded-SAM-2"

if str(GROUNDING_SAM_DIR) not in sys.path:
    sys.path.insert(0, str(GROUNDING_SAM_DIR))

from cv_proc import gen_mask, segment_image_ground  # noqa: E402


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object masks from current input images.")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["apple", "pear"],
        help="Target objects to segment. Default: apple pear",
    )
    parser.add_argument(
        "--mode",
        default="real",
        choices=["real", "sim"],
        help="Recognition mode. Use 'real' for real photos, 'sim' for simulator renders.",
    )
    parser.add_argument(
        "--single-image",
        default=None,
        help="If set, only process this image instead of the default left/right pair.",
    )
    parser.add_argument(
        "--single-prefix",
        default="custom",
        help="Output prefix when --single-image is used. Example: left -> left_mask_apple.png",
    )
    args = parser.parse_args()

    objects = [str(obj).strip() for obj in args.objects if str(obj).strip()]
    if not objects:
        raise ValueError("No valid objects provided.")

    if args.single_image:
        image_path = _resolve_path(args.single_image)
        for obj in objects:
            output_name = f"{args.single_prefix}_mask_{obj}.png"
            mask = segment_image_ground(str(image_path), obj, output_name, recognition_mode=args.mode)
            print(
                {
                    "image": str(image_path),
                    "object": obj,
                    "mode": args.mode,
                    "output_mask": str(ROOT_DIR / "inputs" / output_name),
                    "mask_shape": None if mask is None else list(mask.shape),
                }
            )
        return

    gen_mask(objects, recognition_mode=args.mode)
    for side in ("left", "right"):
        for obj in objects:
            print(
                {
                    "image": str(ROOT_DIR / "inputs" / f"c{side}001.png"),
                    "object": obj,
                    "mode": args.mode,
                    "output_mask": str(ROOT_DIR / "inputs" / f"{side}_mask_{obj}.png"),
                }
            )


if __name__ == "__main__":
    main()
