from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cv_process import choose_model, process_sam_results


ROOT_DIR = Path(__file__).resolve().parent


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _clip_box(box_xyxy: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box after clipping: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def segment_from_box(image_path: Path, box_xyxy: list[float], output_mask: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    height, width = image_bgr.shape[:2]
    clipped_box = _clip_box(box_xyxy, width, height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor = choose_model()
    predictor.set_image(image_rgb)

    results = predictor(bboxes=[clipped_box])
    _, mask = process_sam_results(results)
    if mask is None:
        raise RuntimeError(f"Failed to produce a mask from box: {clipped_box}")

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask), mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a binary mask from a manual bounding box using SAM.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--box",
        required=True,
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Bounding box in xyxy format.",
    )
    parser.add_argument("--out", required=True, help="Output mask PNG path.")
    args = parser.parse_args()

    image_path = _resolve_path(args.image)
    output_mask = _resolve_path(args.out)
    mask = segment_from_box(image_path=image_path, box_xyxy=list(args.box), output_mask=output_mask)

    print(
        {
            "image": str(image_path),
            "box_xyxy": [float(v) for v in args.box],
            "out": str(output_mask),
            "mask_shape": list(mask.shape),
            "mask_pixels": int(np.count_nonzero(mask)),
        }
    )


if __name__ == "__main__":
    main()
