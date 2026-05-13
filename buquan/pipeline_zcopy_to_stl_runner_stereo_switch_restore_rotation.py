#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot runner:
1) image -> robust z-copy filled point cloud
2) z-copy point cloud -> watertight STL
3) inverse-rotate STL back to the original point-cloud orientation

This script keeps the existing stereo-switch scripts unchanged and writes an
additional restored-orientation STL beside the normal outputs, while using a
debug-export Stage1 that also saves raw/centered/aligned point clouds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import trimesh


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, "pipeline_inputs_to_zcopy_robust_stereo_switch_debug.py")
MESH_SCRIPT = os.path.join(SCRIPT_DIR, "zcopy_ply_to_stl_watertight_stereo_switch.py")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return int(proc.returncode), proc.stdout, proc.stderr


def script_supports_arg(script_path: str, flag: str) -> bool:
    try:
        text = open(script_path, "r", encoding="utf-8").read()
    except Exception:
        return False
    return flag in text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run z-copy fill, STL conversion, then restore the STL back to the original point-cloud rotation."
    )

    parser.add_argument("--object", default="pear")
    parser.add_argument("--input-root", default=os.path.join(SCRIPT_DIR, "inputs"))
    parser.add_argument("--calib-pdf", default=os.path.join(SCRIPT_DIR, "inputs", "双ob(1)", "双ob", "标定.pdf"))
    parser.add_argument("--extrinsics-json", default=os.path.join(SCRIPT_DIR, "calib", "refined_stereo_extrinsics.json"))
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--left-depth-rel", default="dleft001.png")
    parser.add_argument("--left-mask-tpl", default="left_mask_{obj}.png")
    parser.add_argument("--right-depth-rel", default="dright001.png")
    parser.add_argument("--right-mask-tpl", default="right_mask_{obj}.png")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc-m", type=float, default=6.0)

    parser.add_argument("--voxel-mm", type=float, default=1.5)
    parser.add_argument("--stat-k", type=int, default=20)
    parser.add_argument("--stat-std", type=float, default=2.0)
    parser.add_argument("--radius-mm", type=float, default=6.0)
    parser.add_argument("--radius-min-neighbors", type=int, default=6)
    parser.add_argument("--cluster-radius-mm", type=float, default=5.0)
    parser.add_argument("--tail-bin-mm", type=float, default=1.2)
    parser.add_argument("--tail-min-support", type=int, default=8)
    parser.add_argument("--tail-max-remove-ratio", type=float, default=0.03)
    parser.add_argument("--center-mode", choices=["mean", "bbox"], default="mean")
    parser.add_argument("--align-axis", choices=["major", "middle", "minor"], default="minor")
    parser.add_argument("--copy-source", choices=["raw", "cleaned_raw", "processed"], default="cleaned_raw")
    parser.add_argument("--copy-times", type=int, default=10)
    parser.add_argument("--copy-step-mm", type=float, default=2.5)
    parser.add_argument("--copy-direction", choices=["neg", "pos", "both"], default="both")
    parser.add_argument("--copy-voxel-mm", type=float, default=0.8)
    parser.add_argument("--pear-y-canonical", choices=["on", "off"], default="on")
    parser.add_argument("--pear-tip-percentile", type=float, default=0.10)

    parser.add_argument("--mesh-voxel-mm", type=float, default=1.2)
    parser.add_argument("--padding-vox", type=int, default=2)
    parser.add_argument("--close-iters", type=int, default=2)
    parser.add_argument("--open-iters", type=int, default=0)
    parser.add_argument("--smooth-iters", type=int, default=8)
    parser.add_argument("--smooth-lambda", type=float, default=0.5)
    parser.add_argument("--target-faces", type=int, default=20000)
    parser.add_argument("--max-split-faces", type=int, default=300000)
    parser.add_argument("--fallback", choices=["convex_hull", "none"], default="convex_hull")
    parser.add_argument("--force-convex-hull", choices=["on", "off"], default="off")
    parser.add_argument("--sample-points", type=int, default=30000)

    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "outputs_pipeline_zcopy_to_stl_restored"))
    parser.add_argument(
        "--restored-stl-out",
        default="",
        help="Optional output STL path after inverse rotation. Default: <output-dir>/<object>_watertight_voxel_restored.stl",
    )
    parser.add_argument(
        "--restored-report-out",
        default="",
        help="Optional inverse-rotation report path. Default: <output-dir>/<object>_restored_rotation_report.json",
    )
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_inverse_rotation(pipeline_report: str) -> np.ndarray:
    payload = _load_json(pipeline_report)
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Pipeline report missing geometry block: {pipeline_report}")
    rotation_to_z = np.asarray(geometry.get("rotation_to_z_3x3", []), dtype=np.float64)
    if rotation_to_z.shape != (3, 3):
        raise ValueError(f"Invalid rotation_to_z_3x3 in pipeline report: {pipeline_report}")
    return rotation_to_z.T


def _restore_stl_orientation(source_stl: str, pipeline_report: str, restored_stl: str) -> np.ndarray:
    inverse_rotation = _load_inverse_rotation(pipeline_report)
    mesh = trimesh.load(source_stl, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh from {source_stl}, got {type(mesh).__name__}")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {source_stl}")

    rotated_vertices = (inverse_rotation @ np.asarray(mesh.vertices, dtype=np.float64).T).T
    restored_mesh = trimesh.Trimesh(
        vertices=rotated_vertices,
        faces=np.asarray(mesh.faces),
        process=False,
    )
    ensure_dir(os.path.dirname(os.path.abspath(restored_stl)))
    restored_mesh.export(restored_stl)
    return inverse_rotation


def main() -> int:
    args = parse_args()
    if not os.path.exists(PIPELINE_SCRIPT):
        raise FileNotFoundError(f"Missing script: {PIPELINE_SCRIPT}")
    if not os.path.exists(MESH_SCRIPT):
        raise FileNotFoundError(f"Missing script: {MESH_SCRIPT}")

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)
    object_name = str(args.object)

    zcopy_ply = os.path.join(output_dir, f"{object_name}_zcopy_filled_aligned.ply")
    zcopy_report = os.path.join(output_dir, f"{object_name}_pipeline_report.json")
    stl_out = os.path.join(output_dir, f"{object_name}_watertight_voxel.stl")
    stl_report = os.path.join(output_dir, f"{object_name}_watertight_voxel_report.json")
    runner_report = os.path.join(output_dir, f"{object_name}_runner_report.json")
    restored_stl = os.path.abspath(
        args.restored_stl_out or os.path.join(output_dir, f"{object_name}_watertight_voxel_restored.stl")
    )
    restored_report = os.path.abspath(
        args.restored_report_out or os.path.join(output_dir, f"{object_name}_restored_rotation_report.json")
    )

    cmd_stage1 = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--object",
        object_name,
        "--input-root",
        os.path.abspath(args.input_root),
        "--calib-pdf",
        os.path.abspath(args.calib_pdf),
        "--extrinsics-json",
        os.path.abspath(args.extrinsics_json),
        "--camera",
        str(args.camera),
        "--left-depth-rel",
        str(args.left_depth_rel),
        "--left-mask-tpl",
        str(args.left_mask_tpl),
        "--right-depth-rel",
        str(args.right_depth_rel),
        "--right-mask-tpl",
        str(args.right_mask_tpl),
        "--depth-scale",
        f"{float(args.depth_scale)}",
        "--depth-trunc-m",
        f"{float(args.depth_trunc_m)}",
        "--voxel-mm",
        f"{float(args.voxel_mm)}",
        "--stat-k",
        f"{int(args.stat_k)}",
        "--stat-std",
        f"{float(args.stat_std)}",
        "--radius-mm",
        f"{float(args.radius_mm)}",
        "--radius-min-neighbors",
        f"{int(args.radius_min_neighbors)}",
        "--cluster-radius-mm",
        f"{float(args.cluster_radius_mm)}",
        "--tail-bin-mm",
        f"{float(args.tail_bin_mm)}",
        "--tail-min-support",
        f"{int(args.tail_min_support)}",
        "--tail-max-remove-ratio",
        f"{float(args.tail_max_remove_ratio)}",
        "--center-mode",
        str(args.center_mode),
        "--align-axis",
        str(args.align_axis),
        "--copy-source",
        str(args.copy_source),
        "--copy-times",
        f"{int(args.copy_times)}",
        "--copy-step-mm",
        f"{float(args.copy_step_mm)}",
        "--copy-direction",
        str(args.copy_direction),
        "--copy-voxel-mm",
        f"{float(args.copy_voxel_mm)}",
        "--output-dir",
        output_dir,
    ]
    if script_supports_arg(PIPELINE_SCRIPT, "--pear-y-canonical"):
        cmd_stage1.extend(["--pear-y-canonical", str(args.pear_y_canonical)])
    if script_supports_arg(PIPELINE_SCRIPT, "--pear-tip-percentile"):
        cmd_stage1.extend(["--pear-tip-percentile", f"{float(args.pear_tip_percentile)}"])

    rc1, out1, err1 = run_cmd(cmd_stage1)
    if rc1 != 0:
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "object": object_name,
            "stage": "zcopy_fill",
            "returncode": rc1,
            "stdout": out1,
            "stderr": err1,
        }
        with open(runner_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ERROR] Stage1 failed. report: {runner_report}")
        return rc1

    if not os.path.exists(zcopy_ply):
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "object": object_name,
            "stage": "zcopy_fill",
            "returncode": 3,
            "error": f"Expected zcopy ply not found: {zcopy_ply}",
            "stdout": out1,
            "stderr": err1,
        }
        with open(runner_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ERROR] Stage1 output missing. report: {runner_report}")
        return 3

    cmd_stage2 = [
        sys.executable,
        MESH_SCRIPT,
        "--input-ply",
        zcopy_ply,
        "--stl-out",
        stl_out,
        "--report-out",
        stl_report,
        "--voxel-mm",
        f"{float(args.mesh_voxel_mm)}",
        "--padding-vox",
        f"{int(args.padding_vox)}",
        "--close-iters",
        f"{int(args.close_iters)}",
        "--open-iters",
        f"{int(args.open_iters)}",
        "--smooth-iters",
        f"{int(args.smooth_iters)}",
        "--smooth-lambda",
        f"{float(args.smooth_lambda)}",
        "--target-faces",
        f"{int(args.target_faces)}",
        "--max-split-faces",
        f"{int(args.max_split_faces)}",
        "--fallback",
        str(args.fallback),
        "--force-convex-hull",
        str(args.force_convex_hull),
        "--sample-points",
        f"{int(args.sample_points)}",
    ]

    rc2, out2, err2 = run_cmd(cmd_stage2)
    payload: Dict[str, object] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "object": object_name,
        "returncode": int(rc2),
        "inputs": {
            "input_root": os.path.abspath(args.input_root),
            "calib_pdf": os.path.abspath(args.calib_pdf),
            "camera": str(args.camera),
            "left_depth_rel": str(args.left_depth_rel),
            "left_mask_tpl": str(args.left_mask_tpl),
            "right_depth_rel": str(args.right_depth_rel),
            "right_mask_tpl": str(args.right_mask_tpl),
            "pear_y_canonical": str(args.pear_y_canonical),
            "pear_tip_percentile": float(args.pear_tip_percentile),
        },
        "stage1": {
            "script": PIPELINE_SCRIPT,
            "returncode": int(rc1),
            "zcopy_ply": zcopy_ply,
            "zcopy_report": zcopy_report if os.path.exists(zcopy_report) else "",
            "stdout": out1,
            "stderr": err1,
        },
        "stage2": {
            "script": MESH_SCRIPT,
            "returncode": int(rc2),
            "stl": stl_out if os.path.exists(stl_out) else "",
            "stl_report": stl_report if os.path.exists(stl_report) else "",
            "stdout": out2,
            "stderr": err2,
            "max_split_faces": int(args.max_split_faces),
            "force_convex_hull": str(args.force_convex_hull),
        },
    }

    if rc2 != 0:
        with open(runner_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ERROR] Stage2 failed. report: {runner_report}")
        return rc2

    try:
        inverse_rotation = _restore_stl_orientation(stl_out, zcopy_report, restored_stl)
        payload["stage3"] = {
            "script": os.path.abspath(__file__),
            "returncode": 0,
            "source_stl": stl_out,
            "pipeline_report": zcopy_report,
            "restored_stl": restored_stl,
            "restored_report": restored_report,
            "inverse_rotation_3x3": inverse_rotation.tolist(),
        }
        restored_payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "object": object_name,
            "returncode": 0,
            "inputs": {
                "source_stl": stl_out,
                "pipeline_report": zcopy_report,
            },
            "geometry": {
                "inverse_rotation_3x3": inverse_rotation.tolist(),
            },
            "outputs": {
                "restored_stl": restored_stl,
                "runner_report": runner_report,
            },
        }
        with open(restored_report, "w", encoding="utf-8") as f:
            json.dump(restored_payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        payload["stage3"] = {
            "script": os.path.abspath(__file__),
            "returncode": 4,
            "source_stl": stl_out if os.path.exists(stl_out) else "",
            "pipeline_report": zcopy_report if os.path.exists(zcopy_report) else "",
            "restored_stl": restored_stl,
            "restored_report": restored_report,
            "error": str(exc),
        }
        with open(restored_report, "w", encoding="utf-8") as f:
            json.dump(payload["stage3"], f, ensure_ascii=False, indent=2)
        with open(runner_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ERROR] Stage3 failed. report: {restored_report}")
        return 4

    payload["outputs"] = {
        "zcopy_ply": zcopy_ply if os.path.exists(zcopy_ply) else "",
        "stl": stl_out if os.path.exists(stl_out) else "",
        "restored_stl": restored_stl if os.path.exists(restored_stl) else "",
        "runner_report": runner_report,
        "restored_report": restored_report,
    }
    with open(runner_report, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] zcopy ply: {zcopy_ply}")
    print(f"[INFO] stl: {stl_out}")
    print(f"[INFO] restored stl: {restored_stl}")
    print(f"[INFO] runner report: {runner_report}")
    print(f"[INFO] restored report: {restored_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
