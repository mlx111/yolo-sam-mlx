"""Compatibility wrapper for the root experience_system harder safety stress cases."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "experience_system"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "tools" / "run_harder_safety_stress_cases.py"), run_name="__main__")
