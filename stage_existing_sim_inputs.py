from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = ROOT_DIR / "inputs"
DEFAULT_OUTPUTS_DIR = ROOT_DIR / "outputs"


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _require_existing_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _validate_rgb(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RGB image: {path}")
    height, width = image.shape[:2]
    return {"path": str(path), "shape": [int(height), int(width), 3], "dtype": str(image.dtype)}


def _validate_depth(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read depth image: {path}")
    if image.ndim != 2:
        raise ValueError(f"Depth image must be single-channel: {path}, got shape={image.shape}")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Depth image must use an integer dtype: {path}, got dtype={image.dtype}")
    if int(np.max(image)) <= 255:
        raise ValueError(
            f"Depth image looks like a visualization PNG, not metric depth: {path}, max={int(np.max(image))}"
        )
    height, width = image.shape[:2]
    return {"path": str(path), "shape": [int(height), int(width)], "dtype": str(image.dtype), "max": int(np.max(image))}


def _validate_mask(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read mask image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzero = int(np.count_nonzero(image))
    if nonzero == 0:
        raise ValueError(f"Mask is empty: {path}")
    height, width = image.shape[:2]
    return {"path": str(path), "shape": [int(height), int(width)], "dtype": str(image.dtype), "nonzero": nonzero}


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage simulator RGB/depth/mask files into the existing inputs/ protocol.")
    parser.add_argument("--left-rgb", default=str(DEFAULT_OUTPUTS_DIR / "left_scene.png"))
    parser.add_argument("--right-rgb", default=str(DEFAULT_OUTPUTS_DIR / "right_scene.png"))
    parser.add_argument("--left-depth", default=str(DEFAULT_OUTPUTS_DIR / "left_depth.png"))
    parser.add_argument("--right-depth", default=str(DEFAULT_OUTPUTS_DIR / "right_depth.png"))
    parser.add_argument("--left-arm-mask", default=str(DEFAULT_OUTPUTS_DIR / "left_mask_roboticarm.png"))
    parser.add_argument("--right-arm-mask", default=str(DEFAULT_OUTPUTS_DIR / "right_mask_roboticarm.png"))
    parser.add_argument("--left-apple-mask", default=str(DEFAULT_OUTPUTS_DIR / "left_mask_apple.png"))
    parser.add_argument("--right-apple-mask", default=str(DEFAULT_OUTPUTS_DIR / "right_mask_apple.png"))
    parser.add_argument("--left-pear-mask", default=str(DEFAULT_OUTPUTS_DIR / "left_mask_pear.png"))
    parser.add_argument("--right-pear-mask", default=str(DEFAULT_OUTPUTS_DIR / "right_mask_pear.png"))
    args = parser.parse_args()

    mapping = {
        _require_existing_file(_resolve_path(args.left_rgb)): INPUTS_DIR / "cleft001.png",
        _require_existing_file(_resolve_path(args.right_rgb)): INPUTS_DIR / "cright001.png",
        _require_existing_file(_resolve_path(args.left_depth)): INPUTS_DIR / "dleft001.png",
        _require_existing_file(_resolve_path(args.right_depth)): INPUTS_DIR / "dright001.png",
        _require_existing_file(_resolve_path(args.left_arm_mask)): INPUTS_DIR / "left_mask_roboticarm.png",
        _require_existing_file(_resolve_path(args.right_arm_mask)): INPUTS_DIR / "right_mask_roboticarm.png",
        _require_existing_file(_resolve_path(args.left_apple_mask)): INPUTS_DIR / "left_mask_apple.png",
        _require_existing_file(_resolve_path(args.right_apple_mask)): INPUTS_DIR / "right_mask_apple.png",
        _require_existing_file(_resolve_path(args.left_pear_mask)): INPUTS_DIR / "left_mask_pear.png",
        _require_existing_file(_resolve_path(args.right_pear_mask)): INPUTS_DIR / "right_mask_pear.png",
    }

    validation = {
        "left_rgb": _validate_rgb(_resolve_path(args.left_rgb)),
        "right_rgb": _validate_rgb(_resolve_path(args.right_rgb)),
        "left_depth": _validate_depth(_resolve_path(args.left_depth)),
        "right_depth": _validate_depth(_resolve_path(args.right_depth)),
        "left_arm_mask": _validate_mask(_resolve_path(args.left_arm_mask)),
        "right_arm_mask": _validate_mask(_resolve_path(args.right_arm_mask)),
        "left_apple_mask": _validate_mask(_resolve_path(args.left_apple_mask)),
        "right_apple_mask": _validate_mask(_resolve_path(args.right_apple_mask)),
        "left_pear_mask": _validate_mask(_resolve_path(args.left_pear_mask)),
        "right_pear_mask": _validate_mask(_resolve_path(args.right_pear_mask)),
    }

    for source, target in mapping.items():
        _copy_file(source, target)

    print(
        {
            "status": "success",
            "inputs_dir": str(INPUTS_DIR),
            "staged_files": {str(target.name): str(source) for source, target in mapping.items()},
            "validation": validation,
        }
    )


if __name__ == "__main__":
    main()
