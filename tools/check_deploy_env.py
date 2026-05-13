#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODULES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("torch", "torch"),
    ("open3d", "open3d"),
    ("mujoco", "mujoco"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("spatialmath", "spatialmath-python"),
    ("py_trees", "py_trees"),
    ("ultralytics", "ultralytics"),
    ("graspnetAPI", "graspnetAPI"),
    ("volcenginesdkarkruntime", "volcenginesdkarkruntime"),
]

OPTIONAL_MODULES = [
    ("openai", "openai"),
    ("trimesh", "trimesh"),
    ("imageio", "imageio"),
    ("gymnasium", "gymnasium"),
    ("pybullet", "pybullet"),
]

REQUIRED_PATHS = [
    "grasp_fastapi_completion_v4.py",
    "main_yoloWorld_sam_completion.py",
    "build_runtime_scene_from_sim_camera.py",
    "calibrate_runtime_pose_from_clouds.py",
    "runtime_pose_calibration.json",
    "camera_pose_calibration.json",
    "yolov8s-world.pth",
    "sam_b.pt",
    "manipulator_grasp/assets/scenes/apple_pear_runtime.xml",
    "manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml",
    "manipulator_grasp/assets/fruit/stl/apple.stl",
    "manipulator_grasp/assets/fruit/stl/pear.stl",
]

OPTIONAL_PATHS = [
    "Grounded-SAM-2",
    "GraspGen",
    "FR5_Reinforcement-learning",
    "GalaxeaManipSim",
]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_paths(root: Path, paths: list[str]) -> list[str]:
    missing: list[str] = []
    for rel in paths:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def check_modules(modules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for module_name, package_name in modules:
        if not has_module(module_name):
            missing.append((module_name, package_name))
    return missing


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check deployment prerequisites for this repository.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to inspect.")
    parser.add_argument("--skip-optional", action="store_true", help="Only check the required runtime set.")
    args = parser.parse_args()

    root = args.root.resolve()

    print(f"Repository root: {root}")
    print(f"Python: {sys.version.split()[0]}")

    required_path_missing = check_paths(root, REQUIRED_PATHS)
    required_module_missing = check_modules(REQUIRED_MODULES)

    optional_path_missing: list[str] = []
    optional_module_missing: list[tuple[str, str]] = []
    if not args.skip_optional:
        optional_path_missing = check_paths(root, OPTIONAL_PATHS)
        optional_module_missing = check_modules(OPTIONAL_MODULES)

    print_section("Required files")
    if required_path_missing:
        for rel in required_path_missing:
            print(f"[MISSING] {rel}")
    else:
        print("[OK] all required files found")

    print_section("Required modules")
    if required_module_missing:
        for module_name, package_name in required_module_missing:
            print(f"[MISSING] {module_name}  (pip install {package_name})")
    else:
        print("[OK] all required modules available")

    if not args.skip_optional:
        print_section("Optional files")
        if optional_path_missing:
            for rel in optional_path_missing:
                print(f"[MISSING] {rel}")
        else:
            print("[OK] all optional paths found")

        print_section("Optional modules")
        if optional_module_missing:
            for module_name, package_name in optional_module_missing:
                print(f"[MISSING] {module_name}  (pip install {package_name})")
        else:
            print("[OK] all optional modules available")

    missing_any = bool(required_path_missing or required_module_missing)
    if missing_any:
        print("\nDeployment check failed.")
        return 1

    print("\nDeployment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
