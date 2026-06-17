"""Compatibility helpers for migrated experience-system tools."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIENCE_SYSTEM = REPO_ROOT / "experience_system"
TOOLS = EXPERIENCE_SYSTEM / "tools"
GALAXEA_ROOT = REPO_ROOT / "galaxea_mujoco"


def _ensure_paths() -> None:
    for path in (str(EXPERIENCE_SYSTEM), str(GALAXEA_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_tool(module_name: str) -> ModuleType:
    _ensure_paths()
    path = TOOLS / f"{module_name}.py"
    if not path.exists():
        raise ModuleNotFoundError(f"experience_system tool not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_experience_system_tools_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experience_system tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def export_tool(module_name: str, namespace: dict[str, object]) -> None:
    module = load_tool(module_name)
    for key, value in module.__dict__.items():
        if key.startswith("__") and key not in {"__doc__", "__all__"}:
            continue
        namespace[key] = value


def run_tool(module_name: str) -> None:
    _ensure_paths()
    path = TOOLS / f"{module_name}.py"
    if not path.exists():
        raise ModuleNotFoundError(f"experience_system tool not found: {path}")
    runpy.run_path(str(path), run_name="__main__")

