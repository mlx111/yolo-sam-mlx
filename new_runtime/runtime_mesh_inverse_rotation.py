from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def load_inverse_rotation_from_pipeline_report(pipeline_report: Path) -> np.ndarray:
    payload = json.loads(Path(pipeline_report).read_text(encoding="utf-8"))
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Pipeline report missing geometry block: {pipeline_report}")

    rotation_to_z = np.asarray(geometry.get("rotation_to_z_3x3", []), dtype=np.float64)
    if rotation_to_z.shape != (3, 3):
        raise ValueError(f"Invalid rotation_to_z_3x3 in pipeline report: {pipeline_report}")
    return rotation_to_z.T


def export_inverse_rotated_mesh(source_mesh: Path, pipeline_report: Path, installed_mesh: Path) -> np.ndarray:
    source_mesh = Path(source_mesh).resolve()
    pipeline_report = Path(pipeline_report).resolve()
    installed_mesh = Path(installed_mesh).resolve()

    if not source_mesh.exists():
        raise FileNotFoundError(f"Source STL mesh not found: {source_mesh}")
    if not pipeline_report.exists():
        raise FileNotFoundError(f"Pipeline report not found: {pipeline_report}")

    inverse_rotation = load_inverse_rotation_from_pipeline_report(pipeline_report)
    mesh = trimesh.load(source_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh from {source_mesh}, got {type(mesh).__name__}")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {source_mesh}")

    rotated_vertices = (inverse_rotation @ np.asarray(mesh.vertices, dtype=np.float64).T).T
    restored_mesh = trimesh.Trimesh(
        vertices=rotated_vertices,
        faces=np.asarray(mesh.faces),
        process=False,
    )
    installed_mesh.parent.mkdir(parents=True, exist_ok=True)
    restored_mesh.export(installed_mesh)
    return inverse_rotation


def write_inverse_rotation_install_metadata(
    metadata_path: Path,
    installed_mesh: Path,
    runner_report: Path,
    pipeline_report: Path,
) -> None:
    metadata_path = Path(metadata_path).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inverse_rotation_applied": True,
        "installed_mesh": str(Path(installed_mesh).resolve()),
        "runner_report": str(Path(runner_report).resolve()),
        "pipeline_report": str(Path(pipeline_report).resolve()),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def inverse_rotation_install_matches_pipeline_report(metadata_path: Path, pipeline_report: Path) -> bool:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not bool(payload.get("inverse_rotation_applied", False)):
        return False
    stored = str(payload.get("pipeline_report", "")).strip()
    if not stored:
        return False
    try:
        return Path(stored).resolve() == Path(pipeline_report).resolve()
    except Exception:
        return False
