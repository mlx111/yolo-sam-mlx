import os
import sys
import json
import subprocess
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OBJECTS = ["apple", "pear"]
DEFAULT_SCENE_OUT = os.path.join(ROOT_DIR, "scence", "apple_pear_runtime.xml")
DEFAULT_SCENE_REFINED_OUT = os.path.join(
    ROOT_DIR,
    "scence",
    "apple_pear_runtime_refined.xml",
)
DEFAULT_REFINED_POSE_JSON = os.path.join(ROOT_DIR, "runtime_assets", "left_view_refined_pose.json")

from dong2 import generate_scene
from new_runtime.apple_pear_scene import (
    load_mesh_quats_from_selected_reports,
    resolve_meshes,
    selected_runner_reports,
)
from object_pose_runtime import estimate_runtime_object_quats
from pointcloud_v2 import CAMERA_EULER_DEG, point
from calibrate_runtime_pose_from_clouds import DEFAULT_JSON_PATH
from runtime_scene_original import build_original_runtime_scene_inputs
from camera_facing_local_axis import align_object_quats_to_camera
sys.path.append(os.path.join(ROOT_DIR, "Grounded-SAM-2"))


def _run_python_script(script_name: str) -> None:
    script_path = Path(ROOT_DIR) / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, "-u", str(script_path)]
    print({"stage": "run_script", "script": script_name, "cmd": cmd})
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        print(f"[{script_name}] {line}", end="")
    proc.wait()
    if proc.returncode != 0:
        joined = "".join(captured)
        raise RuntimeError(
            f"Failed to run {script_name} (code={proc.returncode}).\noutput:\n{joined}"
        )


def _ensure_runtime_background_point_clouds() -> None:
    outputs_dir = Path(ROOT_DIR) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for flag in ("left", "right"):
        background_ply = outputs_dir / f"{flag}1_background.ply"
        if background_ply.exists():
            continue
        rx, ry, rz = CAMERA_EULER_DEG[flag]
        print({"stage": "background_point_cloud", "camera": flag, "target": str(background_ply)})
        point(flag, rx, ry, rz, None)
        if not background_ply.exists():
            raise FileNotFoundError(f"Failed to generate background point cloud: {background_ply}")


def _build_refined_scene_xml() -> str:
    _run_python_script("left_view_pose_refiner.py")
    _run_python_script("apply_refined_pose_to_scene.py")

    refined_scene = Path(DEFAULT_SCENE_REFINED_OUT)
    refined_pose_json = Path(DEFAULT_REFINED_POSE_JSON)
    if not refined_pose_json.exists():
        raise FileNotFoundError(f"Missing refined pose json after refinement: {refined_pose_json}")
    if not refined_scene.exists():
        raise FileNotFoundError(f"Missing refined scene xml after refinement: {refined_scene}")

    print({"refined_pose_json": str(refined_pose_json.resolve())})
    print({"scene_out_refined": str(refined_scene.resolve())})
    return str(refined_scene.resolve())


def build_runtime_scene(objects=None, scene_out=None, start_server=False):
    from cv_proc import gen_mask
    from grasp_fastapi_completion_v4 import start as start_v4_server

    if objects is None:
        objects = list(DEFAULT_OBJECTS)
    if scene_out is None:
        scene_out = DEFAULT_SCENE_OUT

    gen_mask(objects, recognition_mode="real")
    _ensure_runtime_background_point_clouds()
    scene_inputs = build_original_runtime_scene_inputs(objects=objects)
    calibration = scene_inputs["calibration"]
    Path(DEFAULT_JSON_PATH).write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    result = scene_inputs["object_positions"]
    print(result)

    mesh_quats = {}
    if "apple" in result or "pear" in result:
        target_objects = [name for name in DEFAULT_OBJECTS if name in result]
        mesh_files, reports = resolve_meshes(
            mesh_source="buquan",
            camera="left",
            objects=target_objects,
            force_rebuild=False,
        )
        mesh_quats = load_mesh_quats_from_selected_reports(objects=target_objects, require_all=False)
        print({"runtime_meshes": {name: str(path) for name, path in mesh_files.items()}})
        print({"mesh_reports": {name: str(path) for name, path in reports.items()}})

    camera_poses = scene_inputs["camera_poses"]
    object_quats = estimate_runtime_object_quats(
        camera="left",
        calibration_path=DEFAULT_JSON_PATH,
        pear_strategy="v0_legacy",
    )
    object_quats, camera_facing_debug = align_object_quats_to_camera(
        object_quats,
        result,
        camera_poses,
        camera_name="cam1",
        objects=objects,
    )
    print({"runtime_pose_calibration": calibration})
    print({"runtime_pose_calibration_path": str(Path(DEFAULT_JSON_PATH).resolve())})
    print({"camera_poses": camera_poses})
    print({"object_quats": object_quats})
    print({"camera_facing_local_axis": camera_facing_debug})
    print({"mesh_quats": mesh_quats})
    out_path = generate_scene(
        result,
        camera_poses=camera_poses,
        object_quats=object_quats,
        mesh_quats=mesh_quats,
        scene_out=scene_out,
    )
    print({"scene_out": str(Path(out_path).resolve())})
    refined_scene_path = _build_refined_scene_xml()
    print(
        {
            "stage": "start_server",
            "server_module": "grasp_fastapi_completion_v4",
            "scene_out_refined": refined_scene_path,
            "host": "0.0.0.0",
            "port": 8080,
        }
    )
    start_v4_server()

    return out_path


def generate_scenes(objetcs):
    return build_runtime_scene(objects=objetcs, scene_out=DEFAULT_SCENE_OUT, start_server=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    start_server = "--start-server" in args
    args = [arg for arg in args if arg != "--start-server"]

    if args:
        objects = json.loads(args[0])
    else:
        objects = list(DEFAULT_OBJECTS)

    build_runtime_scene(objects=objects, scene_out=DEFAULT_SCENE_OUT, start_server=start_server)
