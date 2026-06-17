"""Compatibility wrapper for the root experience_system sandbox calibration ablation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "experience_system"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "tools" / "run_sandbox_calibration_ablation.py"), run_name="__main__")
