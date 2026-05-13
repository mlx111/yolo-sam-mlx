from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_XML = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime_refined111_no_gripper.xml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _camera_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx) for idx in range(model.ncam)]


def _normalize_depth_for_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)

    values = depth[valid]
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-9:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = (depth[valid] - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def render_scene_capture(
    *,
    xml_path: Path,
    camera_name: str,
    rgb_out: Path,
    depth_out: Path,
    depth_png_out: Path,
    depth_preview_out: Path,
    width: int,
    height: int,
    steps: int,
) -> dict[str, str]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    available_cameras = _camera_names(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found: {camera_name}. Available cameras: {available_cameras}")

    for _ in range(max(steps, 0)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    rgb_out.parent.mkdir(parents=True, exist_ok=True)
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    depth_png_out.parent.mkdir(parents=True, exist_ok=True)
    depth_preview_out.parent.mkdir(parents=True, exist_ok=True)

    rgb_renderer = mujoco.Renderer(model, height=height, width=width)
    depth_renderer = mujoco.Renderer(model, height=height, width=width)

    try:
        rgb_renderer.update_scene(data, camera=cam_id)
        rgb = rgb_renderer.render()

        depth_renderer.enable_depth_rendering()
        depth_renderer.update_scene(data, camera=cam_id)
        depth = depth_renderer.render()
    finally:
        rgb_renderer.close()
        depth_renderer.close()

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    depth_png = np.clip(np.round(depth_mm), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    depth_preview = _normalize_depth_for_preview(depth)

    cv2.imwrite(str(rgb_out), rgb_bgr)
    np.save(str(depth_out), depth)
    cv2.imwrite(str(depth_png_out), depth_png)
    cv2.imwrite(str(depth_preview_out), depth_preview)

    return {
        "xml": str(xml_path),
        "camera": camera_name,
        "rgb_out": str(rgb_out),
        "depth_out": str(depth_out),
        "depth_png_out": str(depth_png_out),
        "depth_preview_out": str(depth_preview_out),
        "width": str(width),
        "height": str(height),
        "steps": str(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RGB and depth images from a MuJoCo scene camera.")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Scene XML path.")
    parser.add_argument("--camera", default="cam1", help="Camera name in the MuJoCo scene.")
    parser.add_argument("--rgb-out", default=None, help="RGB PNG output path.")
    parser.add_argument("--depth-out", default=None, help="Depth .npy output path.")
    parser.add_argument("--depth-png-out", default=None, help="Metric uint16 depth PNG output path in millimeters.")
    parser.add_argument("--depth-preview-out", default=None, help="Normalized depth preview PNG output path.")
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=640, help="Render height.")
    parser.add_argument("--steps", type=int, default=1, help="Simulation steps before rendering.")
    args = parser.parse_args()

    xml_path = _resolve_path(args.xml)
    rgb_out = _resolve_path(args.rgb_out) if args.rgb_out else (DEFAULT_OUTPUT_DIR / f"{args.camera}_rgb.png").resolve()
    depth_out = _resolve_path(args.depth_out) if args.depth_out else (DEFAULT_OUTPUT_DIR / f"{args.camera}_depth.npy").resolve()
    if args.depth_png_out:
        depth_png_out = _resolve_path(args.depth_png_out)
    else:
        depth_png_out = (DEFAULT_OUTPUT_DIR / f"{args.camera}_depth_metric.png").resolve()
    if args.depth_preview_out:
        depth_preview_out = _resolve_path(args.depth_preview_out)
    else:
        depth_preview_out = (DEFAULT_OUTPUT_DIR / f"{args.camera}_depth.png").resolve()

    result = render_scene_capture(
        xml_path=xml_path,
        camera_name=args.camera,
        rgb_out=rgb_out,
        depth_out=depth_out,
        depth_png_out=depth_png_out,
        depth_preview_out=depth_preview_out,
        width=args.width,
        height=args.height,
        steps=args.steps,
    )
    print(result)


if __name__ == "__main__":
    main()
