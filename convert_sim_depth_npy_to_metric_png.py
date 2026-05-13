from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def convert_depth_npy_to_uint16_png(src: Path, dst: Path) -> dict[str, object]:
    depth_m = np.load(src)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.ndim != 2:
        raise ValueError(f"Depth npy must be HxW: {src}, got shape={depth_m.shape}")
    depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    depth_u16 = np.clip(np.round(depth_mm), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), depth_u16)
    return {
        "src": str(src),
        "dst": str(dst),
        "shape": list(depth_u16.shape),
        "dtype": str(depth_u16.dtype),
        "min_mm": int(depth_u16.min()),
        "max_mm": int(depth_u16.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MuJoCo depth .npy files in meters to uint16 millimeter PNGs.")
    parser.add_argument("--left-src", default="outputs/left_depth.npy")
    parser.add_argument("--right-src", default="outputs/right_depth.npy")
    parser.add_argument("--left-out", default="inputs/dleft001.png")
    parser.add_argument("--right-out", default="inputs/dright001.png")
    args = parser.parse_args()

    results = [
        convert_depth_npy_to_uint16_png(_resolve_path(args.left_src), _resolve_path(args.left_out)),
        convert_depth_npy_to_uint16_png(_resolve_path(args.right_src), _resolve_path(args.right_out)),
    ]
    print({"status": "success", "converted": results})


if __name__ == "__main__":
    main()
