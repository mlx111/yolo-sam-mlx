"""Compatibility namespace for migrated experience-system tools.

Modules under ``experience_system/tools`` keep their historical ``source.*``
imports. This namespace resolves those imports to migrated tools first, then to
the original Galaxea MuJoCo robot-specific source modules.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent

__path__ = [
    str(_ROOT / "tools"),
    str(_REPO / "galaxea_mujoco" / "source"),
]

