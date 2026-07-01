from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUQUAN_ROOT = PACKAGE_ROOT / "buquan_runtime"
BUQUAN_INPUTS = BUQUAN_ROOT / "inputs"
BUQUAN_OUTPUTS = PACKAGE_ROOT / "results" / "runtime_scenes" / "real_camera_runtime" / "buquan_completed"
MESH_OUTPUT_DIR = PACKAGE_ROOT / "model" / "meshes" / "runtime_completed"


def _copy_input(src: str | Path, dst: Path) -> None:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Missing buquan input source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    paths = [str(PACKAGE_ROOT)]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False, capture_output=True, text=True)
    return int(proc.returncode), proc.stdout, proc.stderr


def build_buquan_completed_stl(
    *,
    object_name: str,
    color_image_path: str | Path,
    depth_image_path: str | Path,
    mask_path: str | Path,
    camera: str = "left",
) -> dict[str, Any]:
    name = str(object_name).strip().lower().rstrip(".")
    if not name:
        raise ValueError("object_name must be non-empty")

    BUQUAN_INPUTS.mkdir(parents=True, exist_ok=True)
    BUQUAN_OUTPUTS.mkdir(parents=True, exist_ok=True)
    MESH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    color_name = "cleft001.png" if camera == "left" else "cright001.png"
    depth_name = "dleft001.png" if camera == "left" else "dright001.png"
    mask_name = f"{camera}_mask_{name}.png"
    _copy_input(color_image_path, BUQUAN_INPUTS / color_name)
    _copy_input(depth_image_path, BUQUAN_INPUTS / depth_name)
    _copy_input(mask_path, BUQUAN_INPUTS / mask_name)

    runner = BUQUAN_ROOT / "pipeline_zcopy_to_stl_runner_stereo_switch_restore_rotation_pointcloud_v2.py"
    output_dir = BUQUAN_OUTPUTS / name
    restored_stl = output_dir / f"{name}_watertight_voxel_restored.stl"
    restored_report = output_dir / f"{name}_restored_rotation_report.json"
    final_mesh = MESH_OUTPUT_DIR / f"{name}_completed.stl"
    cmd = [
        sys.executable,
        str(runner),
        "--object",
        name,
        "--input-root",
        str(BUQUAN_INPUTS),
        "--calib-pdf",
        "",
        "--extrinsics-json",
        "",
        "--camera",
        camera,
        "--camera-model",
        "sim",
        "--sim-intrinsics-json",
        str(PACKAGE_ROOT / "results" / "runtime_scenes" / "real_camera_runtime" / "buquan_sim_intrinsics.json"),
        "--left-depth-rel",
        depth_name,
        "--left-mask-tpl",
        f"{camera}_mask_{{obj}}.png",
        "--right-depth-rel",
        depth_name,
        "--right-mask-tpl",
        f"{camera}_mask_{{obj}}.png",
        "--output-dir",
        str(output_dir),
        "--restored-stl-out",
        str(restored_stl),
        "--restored-report-out",
        str(restored_report),
    ]
    rc, stdout, stderr = _run(cmd, cwd=PACKAGE_ROOT)
    runner_report = output_dir / f"{name}_runner_report.json"
    if rc != 0:
        return {
            "status": "failed",
            "object_name": name,
            "returncode": rc,
            "stdout": stdout,
            "stderr": stderr,
            "runner_report": str(runner_report.resolve()) if runner_report.exists() else "",
        }
    if not restored_stl.exists():
        raise FileNotFoundError(f"Buquan runner succeeded but restored STL is missing: {restored_stl}")
    shutil.copyfile(restored_stl, final_mesh)
    return {
        "status": "success",
        "object_name": name,
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
        "runner_report": str(runner_report.resolve()),
        "restored_stl": str(restored_stl.resolve()),
        "installed_mesh": str(final_mesh.resolve()),
        "outputs": json.loads(runner_report.read_text(encoding="utf-8")).get("outputs", {}) if runner_report.exists() else {},
    }
