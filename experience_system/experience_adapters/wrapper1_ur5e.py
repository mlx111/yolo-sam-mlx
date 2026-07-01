"""Compatibility wrapper that re-exports the UR5e core adapter."""

from __future__ import annotations

try:
    from experience_system.ur5e_core import Wrapper1UR5eAdapter
except ModuleNotFoundError:  # pragma: no cover - script-root fallback
    from ur5e_core import Wrapper1UR5eAdapter

__all__ = ["Wrapper1UR5eAdapter"]
