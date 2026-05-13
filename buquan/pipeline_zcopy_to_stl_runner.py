#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot runner:
1) image -> robust z-copy filled point cloud
2) z-copy point cloud -> watertight STL

This script only orchestrates existing scripts and does not modify them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, "pipeline_inputs_to_zcopy_robust.py")
MESH_SCRIPT = os.path.join(SCRIPT_DIR, "zcopy_ply_to_stl_watertight.py")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return int(proc.returncode), proc.stdout, proc.stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run z-copy fill then STL conversion in sequence.")

    # Stage 1: robust z-copy pipeline args
    parser.add_argument("--object", default="pear")
    parser.add_argument("--input-root", default=os.path.join(SCRIPT_DIR, "inputs"))
    parser.add_argument("--calib-pdf", default=os.path.join(SCRIPT_DIR, "inputs", "双ob(1)", "双ob", "标定.pdf"))
    parser.add_argument("--extrinsics-json", default=os.path.join(SCRIPT_DIR, "calib", "refined_stereo_extrinsics.json"))
    parser.add_argument("--left-depth-rel", default="dleft001.png")
    parser.add_argument("--left-mask-tpl", default="left_mask_{obj}.png")
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

    # Stage 2: mesh conversion args
    parser.add_argument("--mesh-voxel-mm", type=float, default=1.2)
    parser.add_argument("--padding-vox", type=int, default=2)
    parser.add_argument("--close-iters", type=int, default=2)
    parser.add_argument("--open-iters", type=int, default=0)
    parser.add_argument("--smooth-iters", type=int, default=8)
    parser.add_argument("--smooth-lambda", type=float, default=0.5)
    parser.add_argument("--target-faces", type=int, default=20000)
    parser.add_argument("--fallback", choices=["convex_hull", "none"], default="convex_hull")
    parser.add_argument("--sample-points", type=int, default=30000)

    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "outputs_pipeline_zcopy_to_stl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(PIPELINE_SCRIPT):
        raise FileNotFoundError(f"Missing script: {PIPELINE_SCRIPT}")
    if not os.path.exists(MESH_SCRIPT):
        raise FileNotFoundError(f"Missing script: {MESH_SCRIPT}")

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    zcopy_out_dir = output_dir
    object_name = str(args.object)
    zcopy_ply = os.path.join(zcopy_out_dir, f"{object_name}_zcopy_filled_aligned.ply")
    zcopy_report = os.path.join(zcopy_out_dir, f"{object_name}_pipeline_report.json")
    stl_out = os.path.join(zcopy_out_dir, f"{object_name}_watertight_voxel.stl")
    stl_report = os.path.join(zcopy_out_dir, f"{object_name}_watertight_voxel_report.json")
    runner_report = os.path.join(zcopy_out_dir, f"{object_name}_runner_report.json")

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
        "--left-depth-rel",
        str(args.left_depth_rel),
        "--left-mask-tpl",
        str(args.left_mask_tpl),
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
        zcopy_out_dir,
    ]

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
        "--fallback",
        str(args.fallback),
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
            "left_depth_rel": str(args.left_depth_rel),
            "left_mask_tpl": str(args.left_mask_tpl),
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
        },
        "outputs": {
            "zcopy_ply": zcopy_ply if os.path.exists(zcopy_ply) else "",
            "stl": stl_out if os.path.exists(stl_out) else "",
            "runner_report": runner_report,
        },
    }
    with open(runner_report, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if rc2 != 0:
        print(f"[ERROR] Stage2 failed. report: {runner_report}")
        return rc2

    print(f"[INFO] zcopy ply: {zcopy_ply}")
    print(f"[INFO] stl: {stl_out}")
    print(f"[INFO] runner report: {runner_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
