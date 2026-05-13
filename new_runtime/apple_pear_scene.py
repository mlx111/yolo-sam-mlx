from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from new_runtime.runtime_mesh_inverse_rotation import (
    export_inverse_rotated_mesh,
    inverse_rotation_install_matches_pipeline_report,
    write_inverse_rotation_install_metadata,
)
from pointcloud_v2 import pos as estimate_positions
from pointcloud_v2 import point as estimate_single_view_point


APPLE = "apple"
PEAR = "pear"
SUPPORTED_OBJECTS = (APPLE, PEAR)


@dataclass(frozen=True)
class SceneArtifacts:
    scene_xml: Path
    mesh_files: Dict[str, Path]
    positions: Dict[str, list[float]]
    reports: Dict[str, Path]


class SceneBuildError(RuntimeError):
    """Raised when scene generation cannot complete."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_scene_template() -> Path:
    return repo_root() / "manipulator_grasp" / "assets" / "scenes" / "scene2.xml"


def default_scene_output() -> Path:
    return repo_root() / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime.xml"


def runtime_mesh_dir() -> Path:
    return repo_root() / "runtime_assets" / "meshes"


def runtime_mesh_metadata_path(object_name: str) -> Path:
    name = str(object_name).strip().lower()
    return runtime_mesh_dir() / f"{name}_inverse_rotation_install.json"


def runtime_report_dir() -> Path:
    return repo_root() / "runtime_assets" / "reports"


def _resolve_target_objects(objects: Iterable[str] | None = None) -> tuple[str, ...]:
    if objects is None:
        return SUPPORTED_OBJECTS
    resolved: list[str] = []
    for value in objects:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_OBJECTS:
            raise SceneBuildError(f"Unsupported object name: {value}")
        if name not in resolved:
            resolved.append(name)
    if not resolved:
        raise SceneBuildError("No valid objects requested.")
    return tuple(resolved)


def selected_runner_report_path(object_name: str) -> Path:
    name = str(object_name).strip().lower()
    if name not in SUPPORTED_OBJECTS:
        raise SceneBuildError(f"Unsupported object name: {object_name}")
    return runtime_report_dir() / f"{name}_selected_runner_report.json"


def selected_runner_reports(objects: Iterable[str] | None = None) -> Dict[str, Path]:
    reports: Dict[str, Path] = {}
    for name in _resolve_target_objects(objects):
        path = selected_runner_report_path(name)
        if path.exists():
            reports[name] = path.resolve()
    return reports


def _import_open3d():
    try:
        import open3d as o3d  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise SceneBuildError("Failed to import open3d for runtime mesh recovery.") from exc
    return o3d


def _cached_buquan_outputs(object_name: str, camera: str) -> tuple[Path, Path] | None:
    root = repo_root()
    candidates = [
        (
            root
            / "runtime_assets"
            / "reports"
            / f"{object_name}_{camera}"
            / f"{object_name}_watertight_voxel.stl",
            root
            / "runtime_assets"
            / "reports"
            / f"{object_name}_{camera}"
            / f"{object_name}_runner_report.json",
        ),
        (
            root / "buquan" / "outputs_eval_lr_20260308" / f"{object_name}_{camera}" / f"{object_name}_watertight_voxel.stl",
            root / "buquan" / "outputs_eval_lr_20260308" / f"{object_name}_{camera}" / f"{object_name}_runner_report.json",
        ),
        (
            root / "buquan" / "outputs_pipeline_zcopy_to_stl_final" / f"{object_name}_watertight_voxel.stl",
            root / "buquan" / "outputs_pipeline_zcopy_to_stl_final" / f"{object_name}_runner_report.json",
        ),
        (
            root / "buquan" / "outputs_pipeline_zcopy_to_stl_restored_pc2_release" / f"{object_name}_watertight_voxel.stl",
            root / "buquan" / "outputs_pipeline_zcopy_to_stl_restored_pc2_release" / f"{object_name}_runner_report.json",
        ),
        (
            root
            / "buquan"
            / "outputs_pipeline_zcopy_to_stl_stereo_switch_combined_test"
            / f"{object_name}_watertight_voxel.stl",
            root
            / "buquan"
            / "outputs_pipeline_zcopy_to_stl_stereo_switch_combined_test"
            / f"{object_name}_runner_report.json",
        ),
    ]
    for stl_path, report_path in candidates:
        if not stl_path.exists() or not report_path.exists():
            continue
        if object_name in {APPLE, PEAR} and not _runner_is_low_poly_mode(report_path):
            print(
                f"[INFO] runtime mesh recovery: skip {object_name} cache without low-poly mode -> {report_path}",
                flush=True,
            )
            continue
        return stl_path, report_path
    return None


def parse_position(text: str, name: str) -> list[float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise SceneBuildError(f"{name} position must be x,y,z.")
    try:
        coords = [float(part) for part in parts]
    except ValueError as exc:
        raise SceneBuildError(f"{name} position must contain numeric values.") from exc
    coords[2] = max(0.0, coords[2])
    return coords


def resolve_positions(
    camera: str = "left",
    apple_pos: str | None = None,
    pear_pos: str | None = None,
) -> Dict[str, list[float]]:
    explicit: Dict[str, list[float]] = {}
    if apple_pos:
        explicit[APPLE] = parse_position(apple_pos, APPLE)
    if pear_pos:
        explicit[PEAR] = parse_position(pear_pos, PEAR)
    if len(explicit) == len(SUPPORTED_OBJECTS):
        return explicit

    # Match zhenghe.py by default: estimate both objects from the shared
    # multi-view helper instead of a single camera approximation.
    estimated = estimate_positions(list(SUPPORTED_OBJECTS))
    resolved: Dict[str, list[float]] = {}
    for name in SUPPORTED_OBJECTS:
        if name in explicit:
            resolved[name] = explicit[name]
            continue
        value = estimated.get(name)
        if not value or len(value) != 3:
            raise SceneBuildError(f"Failed to estimate position for {name}.")
        resolved[name] = [float(value[0]), float(value[1]), max(0.0, float(value[2]))]
    return resolved


def _estimate_positions_from_single_camera(camera: str) -> Dict[str, list[float]]:
    rotations = {
        "left": (-17.08, -40.16, 96.59),
        "right": (-97.480497, -1.078239, -0.070489),
    }
    if camera not in rotations:
        raise SceneBuildError(f"Unsupported camera for position estimation: {camera}")
    rx, ry, rz = rotations[camera]
    estimated: Dict[str, list[float]] = {}
    for name in SUPPORTED_OBJECTS:
        value = estimate_single_view_point(camera, rx, ry, rz, name)
        if not value or len(value) != 3:
            raise SceneBuildError(f"Failed to estimate {name} from {camera} camera.")
        estimated[name] = [float(value[0]), float(value[1]), max(0.0, float(value[2]))]
    return estimated


def _resolve_pipeline_report_from_runner_report(runner_report: Path) -> Path:
    if not runner_report.exists():
        raise SceneBuildError(f"Runner report not found: {runner_report}")

    runner_payload = json.loads(runner_report.read_text(encoding="utf-8"))
    stage1 = runner_payload.get("stage1")
    if not isinstance(stage1, dict):
        raise SceneBuildError(f"Runner report missing stage1 block: {runner_report}")

    zcopy_report_raw = stage1.get("zcopy_report", "")
    zcopy_report = Path(zcopy_report_raw).expanduser()
    candidates: list[Path] = []
    if zcopy_report_raw:
        if zcopy_report.is_absolute():
            candidates.append(zcopy_report)
            candidates.append(runner_report.parent / zcopy_report.name)
        else:
            candidates.append((runner_report.parent / zcopy_report).resolve())
    object_name = str(runner_payload.get("object", "")).strip()
    if object_name:
        candidates.append(runner_report.parent / f"{object_name}_pipeline_report.json")

    camera_hint = ""
    inputs = runner_payload.get("inputs")
    if isinstance(inputs, dict):
        camera_hint = str(inputs.get("camera", "")).strip().lower()
    camera_order: list[str] = []
    if camera_hint in {"left", "right"}:
        camera_order.append(camera_hint)
    for camera_name in ("left", "right"):
        if camera_name not in camera_order:
            camera_order.append(camera_name)

    if object_name:
        root = repo_root()
        for camera_name in camera_order:
            candidates.append(
                root / "buquan" / "outputs_eval_lr_20260308" / f"{object_name}_{camera_name}" / f"{object_name}_pipeline_report.json"
            )
        candidates.append(root / "buquan" / "outputs_pipeline_zcopy_to_stl_stereo_switch_combined_test" / f"{object_name}_pipeline_report.json")
        candidates.append(root / "buquan" / "outputs_pipeline_zcopy_to_stl_final" / f"{object_name}_pipeline_report.json")

    resolved_report = next((path.resolve() for path in candidates if path.exists()), None)
    if resolved_report is None:
        tried = ", ".join(str(path) for path in candidates) if candidates else "<none>"
        raise SceneBuildError(
            f"Pipeline report not found from runner report {runner_report}. Tried: {tried}"
        )

    pipeline_payload = json.loads(resolved_report.read_text(encoding="utf-8"))
    geometry = pipeline_payload.get("geometry")
    if not isinstance(geometry, dict):
        raise SceneBuildError(f"Pipeline report missing geometry block: {resolved_report}")
    if np.asarray(geometry.get("rotation_to_z_3x3", []), dtype=float).shape != (3, 3):
        raise SceneBuildError(f"Invalid rotation_to_z_3x3 in pipeline report: {resolved_report}")
    return resolved_report


def _pear_runner_has_plus_y_tip(runner_report: Path) -> bool:
    if not runner_report.exists():
        return False

    try:
        runner_payload = json.loads(runner_report.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False

    stage1 = runner_payload.get("stage1")
    if isinstance(stage1, dict):
        stage1_script = Path(str(stage1.get("script", ""))).name
        if stage1_script == "pipeline_inputs_to_zcopy_robust_stereo_switch_pear_ycanon.py":
            return True

    try:
        pipeline_report = _resolve_pipeline_report_from_runner_report(runner_report)
        pipeline_payload = json.loads(pipeline_report.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False

    geometry = pipeline_payload.get("geometry")
    if not isinstance(geometry, dict):
        return False
    return bool(geometry.get("applied_pear_y_canonicalization", False)) and str(
        geometry.get("pear_tip_target_axis", "")
    ).strip() == "+Y"


def _runner_is_low_poly_mode(runner_report: Path) -> bool:
    if not runner_report.exists():
        return False
    try:
        runner_payload = json.loads(runner_report.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False

    stage2 = runner_payload.get("stage2")
    if not isinstance(stage2, dict):
        return False
    if str(stage2.get("force_convex_hull", "")).strip().lower() == "on":
        return True

    stl_report_raw = str(stage2.get("stl_report", "")).strip()
    if not stl_report_raw:
        return False
    stl_report = Path(stl_report_raw).expanduser()
    candidates = [stl_report, runner_report.parent / stl_report.name]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if bool(payload.get("fallback_used", False)):
            return True
        config = payload.get("config")
        if isinstance(config, dict) and str(config.get("force_convex_hull", "")).strip().lower() == "on":
            return True
    return False


def buquan_inverse_quats_from_reports(reports: Dict[str, Path]) -> Dict[str, list[float]]:
    quats: Dict[str, list[float]] = {}
    for object_name, runner_report in reports.items():
        pipeline_report = _resolve_pipeline_report_from_runner_report(Path(runner_report))
        payload = json.loads(pipeline_report.read_text(encoding="utf-8"))
        geometry = payload.get("geometry")
        if not isinstance(geometry, dict):
            raise SceneBuildError(f"Pipeline report missing geometry block: {pipeline_report}")
        rotation_to_z = np.asarray(geometry.get("rotation_to_z_3x3", []), dtype=float)
        if rotation_to_z.shape != (3, 3):
            raise SceneBuildError(f"Invalid rotation_to_z_3x3 in pipeline report: {pipeline_report}")
        inverse_rotation = rotation_to_z.T
        quat_xyzw = Rotation.from_matrix(inverse_rotation).as_quat()
        quats[object_name] = [
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        ]
    return quats


def load_mesh_quats_from_selected_reports(
    objects: Iterable[str] | None = None,
    require_all: bool = False,
) -> Dict[str, list[float]]:
    target_objects = _resolve_target_objects(objects)
    reports = selected_runner_reports(target_objects)
    missing = [name for name in target_objects if name not in reports]
    if missing and require_all:
        missing_paths = [str(selected_runner_report_path(name)) for name in missing]
        raise SceneBuildError(
            f"Missing selected runner reports for objects: {', '.join(missing)}. "
            f"Expected: {', '.join(missing_paths)}"
        )
    if not reports:
        return {}

    filtered_reports: Dict[str, Path] = {}
    for object_name, runner_report in reports.items():
        pipeline_report = _resolve_pipeline_report_from_runner_report(Path(runner_report))
        metadata_path = runtime_mesh_metadata_path(object_name)
        if inverse_rotation_install_matches_pipeline_report(metadata_path, pipeline_report):
            print(
                f"[INFO] runtime mesh recovery: skip mesh_quat for {object_name} because installed STL already includes inverse rotation -> {metadata_path}",
                flush=True,
            )
            continue
        filtered_reports[object_name] = runner_report

    if not filtered_reports:
        return {}
    return buquan_inverse_quats_from_reports(filtered_reports)


def _load_runner_payload(runner_report: Path) -> dict:
    try:
        payload = json.loads(runner_report.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SceneBuildError(f"Failed to read runner report: {runner_report}") from exc
    if not isinstance(payload, dict):
        raise SceneBuildError(f"Invalid runner report payload: {runner_report}")
    return payload


def _resolve_restored_stl_from_runner_report(runner_report: Path) -> Path | None:
    runner_payload = _load_runner_payload(runner_report)
    stage3 = runner_payload.get("stage3")
    if not isinstance(stage3, dict):
        return None

    restored_raw = str(stage3.get("restored_stl", "")).strip()
    if not restored_raw:
        return None

    restored_path = Path(restored_raw).expanduser()
    candidates: list[Path] = []
    if restored_path.is_absolute():
        candidates.append(restored_path)
        candidates.append(runner_report.parent / restored_path.name)
    else:
        candidates.append((runner_report.parent / restored_path).resolve())
        candidates.append((runner_report.parent / restored_path.name).resolve())

    return next((path.resolve() for path in candidates if path.exists()), None)


def _install_rotation_recovered_mesh(source_mesh: Path, runner_report: Path, installed_mesh: Path) -> None:
    pipeline_report = _resolve_pipeline_report_from_runner_report(runner_report)
    source_mesh = source_mesh.resolve()
    installed_mesh = installed_mesh.resolve()
    try:
        restored_mesh = _resolve_restored_stl_from_runner_report(runner_report)
        if restored_mesh is not None:
            installed_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(restored_mesh, installed_mesh)
        else:
            export_inverse_rotated_mesh(source_mesh, pipeline_report, installed_mesh)
        write_inverse_rotation_install_metadata(
            runtime_mesh_metadata_path(installed_mesh.stem),
            installed_mesh,
            runner_report,
            pipeline_report,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SceneBuildError(str(exc)) from exc

    print(
        f"[INFO] runtime mesh recovery: runner report -> {runner_report}",
        flush=True,
    )
    print(
        f"[INFO] runtime mesh recovery: pipeline report -> {pipeline_report}",
        flush=True,
    )
    print(
        f"[INFO] runtime mesh recovery: installed inverse-rotated buquan mesh -> {installed_mesh}",
        flush=True,
    )


def resolve_meshes(
    mesh_source: str,
    camera: str,
    objects: Iterable[str] | None = None,
    force_rebuild: bool = False,
    input_root: str | Path | None = None,
    camera_model: str = "pointcloud_v2",
    sim_intrinsics_json: str | Path | None = None,
) -> tuple[Dict[str, Path], Dict[str, Path]]:
    target_objects = _resolve_target_objects(objects)
    if mesh_source == "fixed":
        base = repo_root() / "manipulator_grasp" / "assets" / "fruit" / "stl"
        meshes = {name: (base / f"{name}.stl").resolve() for name in target_objects}
        missing = [str(path) for path in meshes.values() if not path.exists()]
        if missing:
            raise SceneBuildError(f"Missing fixed STL files: {', '.join(missing)}")
        return meshes, {}
    if mesh_source != "buquan":
        raise SceneBuildError(f"Unsupported mesh source: {mesh_source}")

    runner = repo_root() / "buquan" / "pipeline_zcopy_to_stl_runner_stereo_switch_restore_rotation_pointcloud_v2.py"
    if not runner.exists():
        raise SceneBuildError(f"Missing buquan runner: {runner}")

    report_dir = runtime_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    scene_mesh_dir = repo_root() / "manipulator_grasp" / "assets" / "fruit" / "stl"
    scene_mesh_dir.mkdir(parents=True, exist_ok=True)
    runner_input_root = Path(input_root).resolve() if input_root is not None else None
    sim_intrinsics_path = Path(sim_intrinsics_json).resolve() if sim_intrinsics_json is not None else None

    mesh_files: Dict[str, Path] = {}
    reports: Dict[str, Path] = {}
    for name in target_objects:
        attempted_cameras = [camera]
        for candidate in ("left", "right"):
            if candidate not in attempted_cameras:
                attempted_cameras.append(candidate)

        last_error: str | None = None
        for candidate_camera in attempted_cameras:
            cached = None if force_rebuild else _cached_buquan_outputs(name, candidate_camera)
            if cached is not None:
                cached_stl, cached_report = cached
                installed_mesh = scene_mesh_dir / f"{name}.stl"
                stable_report = selected_runner_report_path(name)
                _install_rotation_recovered_mesh(cached_stl, cached_report, installed_mesh)
                if cached_report.exists():
                    shutil.copy2(cached_report, stable_report)
                    reports[name] = stable_report.resolve()
                else:
                    reports[name] = cached_stl.resolve()
                mesh_files[name] = installed_mesh.resolve()
                break

            output_dir = report_dir / f"{name}_{candidate_camera}"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(runner),
                "--object",
                name,
                "--camera",
                candidate_camera,
                "--output-dir",
                str(output_dir),
                "--smooth-iters",
                "0",
                "--target-faces",
                "0",
                "--sample-points",
                "0",
            ]
            if runner_input_root is not None:
                cmd.extend(["--input-root", str(runner_input_root)])
            if camera_model:
                cmd.extend(["--camera-model", str(camera_model)])
            if sim_intrinsics_path is not None:
                cmd.extend(["--sim-intrinsics-json", str(sim_intrinsics_path)])
            if name == PEAR:
                cmd.extend(
                    [
                        "--pear-y-canonical",
                        "off",
                    ]
                )
            if name in {APPLE, PEAR}:
                cmd.extend(
                    [
                        "--force-convex-hull",
                        "on",
                    ]
                )
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                last_error = (
                    f"{candidate_camera} camera failed for {name}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
                continue

            generated_stl = output_dir / f"{name}_watertight_voxel.stl"
            runner_report = output_dir / f"{name}_runner_report.json"
            if not generated_stl.exists():
                last_error = f"Generated STL missing for {name} from {candidate_camera}: {generated_stl}"
                continue
            if not runner_report.exists():
                last_error = f"Runner report missing for {name} from {candidate_camera}: {runner_report}"
                continue
            installed_mesh = scene_mesh_dir / f"{name}.stl"
            stable_report = selected_runner_report_path(name)
            _install_rotation_recovered_mesh(generated_stl, runner_report, installed_mesh)
            shutil.copy2(runner_report, stable_report)
            mesh_files[name] = installed_mesh.resolve()
            reports[name] = stable_report.resolve()
            break
        else:
            raise SceneBuildError(
                f"Failed to generate STL for {name} from preferred cameras {attempted_cameras}. "
                f"Last error:\n{last_error}"
            )
    return mesh_files, reports


def _add_mesh_body(
    spec: mujoco.MjSpec,
    object_name: str,
    body_name: str,
    joint_name: str,
    geom_name: str,
    mesh_name: str,
    mesh_path: Path,
    pos_xyz: list[float],
) -> None:
    spec.add_mesh(file=str(mesh_path), name=mesh_name)
    body = spec.worldbody.add_body(name=body_name, pos=pos_xyz, quat=[1.0, 0.0, 0.0, 0.0])
    body.add_joint(name=joint_name, type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    if object_name == APPLE:
        body.add_geom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name, material="apple_mat")
    elif object_name == PEAR:
        body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=[0.62, 0.78, 0.20, 1.0],
        )
    else:
        raise SceneBuildError(f"Unsupported object: {object_name}")


def build_scene(
    mesh_source: str = "fixed",
    camera: str = "left",
    scene_out: str | Path | None = None,
    apple_pos: str | None = None,
    pear_pos: str | None = None,
) -> SceneArtifacts:
    positions = resolve_positions(camera=camera, apple_pos=apple_pos, pear_pos=pear_pos)
    mesh_files, reports = resolve_meshes(mesh_source=mesh_source, camera=camera)

    template = default_scene_template()
    spec = mujoco.MjSpec.from_file(str(template))
    spec.option.timestep = 0.005
    spec.option.gravity = (0.0, 0.0, -9.81)

    _add_mesh_body(
        spec=spec,
        object_name=APPLE,
        body_name="apple_runtime",
        joint_name="apple_runtime_joint",
        geom_name="apple_runtime_geom",
        mesh_name="apple_runtime_mesh",
        mesh_path=mesh_files[APPLE],
        pos_xyz=positions[APPLE],
    )
    _add_mesh_body(
        spec=spec,
        object_name=PEAR,
        body_name="pear_runtime",
        joint_name="pear_runtime_joint",
        geom_name="pear_runtime_geom",
        mesh_name="pear_runtime_mesh",
        mesh_path=mesh_files[PEAR],
        pos_xyz=positions[PEAR],
    )

    scene_xml = Path(scene_out).resolve() if scene_out else default_scene_output().resolve()
    scene_xml.parent.mkdir(parents=True, exist_ok=True)

    try:
        spec.compile()
    except Exception as exc:  # noqa: BLE001
        raise SceneBuildError(f"MuJoCo compile failed for generated scene: {exc}") from exc

    scene_xml.write_text(spec.to_xml(), encoding="utf-8")
    return SceneArtifacts(scene_xml=scene_xml, mesh_files=mesh_files, positions=positions, reports=reports)


def artifacts_to_json(artifacts: SceneArtifacts) -> str:
    payload = {
        "scene_xml": str(artifacts.scene_xml),
        "mesh_files": {name: str(path) for name, path in artifacts.mesh_files.items()},
        "positions": artifacts.positions,
        "reports": {name: str(path) for name, path in artifacts.reports.items()},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
