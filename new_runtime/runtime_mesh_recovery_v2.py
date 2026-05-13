from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from new_runtime.runtime_mesh_inverse_rotation import (
    export_inverse_rotated_mesh,
    write_inverse_rotation_install_metadata,
)


APPLE = "apple"
PEAR = "pear"
SUPPORTED_OBJECTS = (APPLE, PEAR)


class SceneBuildError(RuntimeError):
    """Raised when runtime mesh recovery cannot complete."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_report_dir() -> Path:
    return repo_root() / "runtime_assets" / "reports"


def runtime_scene_mesh_dir() -> Path:
    return repo_root() / "manipulator_grasp" / "assets" / "fruit" / "stl"


def runtime_mesh_metadata_dir() -> Path:
    return repo_root() / "runtime_assets" / "meshes"


def runtime_mesh_metadata_path(object_name: str) -> Path:
    name = str(object_name).strip().lower()
    return runtime_mesh_metadata_dir() / f"{name}_inverse_rotation_install.json"


def _import_open3d():
    try:
        import open3d as o3d  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise SceneBuildError("Failed to import open3d for runtime mesh recovery v2.") from exc
    return o3d


def _cached_buquan_outputs(object_name: str, camera: str) -> tuple[Path, Path] | None:
    root = repo_root()
    candidates = [
        (
            root / "buquan" / "outputs_eval_lr_20260308" / f"{object_name}_{camera}" / f"{object_name}_watertight_voxel.stl",
            root / "buquan" / "outputs_eval_lr_20260308" / f"{object_name}_{camera}" / f"{object_name}_runner_report.json",
        ),
        (
            root / "buquan" / "outputs_pipeline_zcopy_to_stl_final" / f"{object_name}_watertight_voxel.stl",
            root / "buquan" / "outputs_pipeline_zcopy_to_stl_final" / f"{object_name}_runner_report.json",
        ),
    ]
    for stl_path, report_path in candidates:
        if stl_path.exists():
            return stl_path, report_path
    return None


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

    resolved = next((path.resolve() for path in candidates if path.exists()), None)
    if resolved is None:
        tried = ", ".join(str(path) for path in candidates) if candidates else "<none>"
        raise SceneBuildError(
            f"Pipeline report not found from runner report {runner_report}. Tried: {tried}"
        )
    return resolved


def _load_pipeline_report_from_runner_report(runner_report: Path) -> Path:
    pipeline_report = _resolve_pipeline_report_from_runner_report(runner_report)
    pipeline_payload = json.loads(pipeline_report.read_text(encoding="utf-8"))
    geometry = pipeline_payload.get("geometry")
    if not isinstance(geometry, dict):
        raise SceneBuildError(f"Pipeline report missing geometry block: {pipeline_report}")

    if np.asarray(geometry.get("rotation_to_z_3x3", []), dtype=float).shape != (3, 3):
        raise SceneBuildError(f"Invalid rotation_to_z_3x3 in pipeline report: {pipeline_report}")
    return pipeline_report


def _install_rotation_recovered_mesh(source_mesh: Path, runner_report: Path, installed_mesh: Path) -> None:
    pipeline_report = _load_pipeline_report_from_runner_report(runner_report)
    source_mesh = source_mesh.resolve()
    installed_mesh = installed_mesh.resolve()
    try:
        export_inverse_rotated_mesh(source_mesh, pipeline_report, installed_mesh)
        write_inverse_rotation_install_metadata(
            runtime_mesh_metadata_path(installed_mesh.stem),
            installed_mesh,
            runner_report,
            pipeline_report,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SceneBuildError(str(exc)) from exc

    print(f"[INFO] runtime mesh recovery v2: runner report -> {runner_report}", flush=True)
    print(f"[INFO] runtime mesh recovery v2: pipeline report -> {pipeline_report}", flush=True)
    print(
        f"[INFO] runtime mesh recovery v2: installed inverse-rotated buquan mesh -> {installed_mesh}",
        flush=True,
    )


def resolve_meshes(mesh_source: str, camera: str) -> tuple[dict[str, Path], dict[str, Path]]:
    if mesh_source == "fixed":
        base = runtime_scene_mesh_dir()
        meshes = {name: (base / f"{name}.stl").resolve() for name in SUPPORTED_OBJECTS}
        missing = [str(path) for path in meshes.values() if not path.exists()]
        if missing:
            raise SceneBuildError(f"Missing fixed STL files: {', '.join(missing)}")
        return meshes, {}
    if mesh_source != "buquan":
        raise SceneBuildError(f"Unsupported mesh source: {mesh_source}")

    runner = repo_root() / "buquan" / "pipeline_zcopy_to_stl_runner_stereo_switch.py"
    if not runner.exists():
        raise SceneBuildError(f"Missing buquan runner: {runner}")

    report_dir = runtime_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    scene_mesh_dir = runtime_scene_mesh_dir()
    scene_mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_files: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    for name in SUPPORTED_OBJECTS:
        attempted_cameras = [camera]
        for candidate in ("left", "right"):
            if candidate not in attempted_cameras:
                attempted_cameras.append(candidate)

        last_error: str | None = None
        for candidate_camera in attempted_cameras:
            cached = _cached_buquan_outputs(name, candidate_camera)
            if cached is not None:
                cached_stl, cached_report = cached
                installed_mesh = scene_mesh_dir / f"{name}.stl"
                stable_report = report_dir / f"{name}_selected_runner_report.json"
                _install_rotation_recovered_mesh(cached_stl, cached_report, installed_mesh)
                shutil.copy2(cached_report, stable_report)
                reports[name] = stable_report.resolve()
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
            stable_report = report_dir / f"{name}_selected_runner_report.json"
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
