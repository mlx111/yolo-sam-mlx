"""Compatibility wrapper for experience_system.tools.run_r1pro_memory_policy_smoke."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source._experience_system_wrapper import export_tool, run_tool

export_tool('run_r1pro_memory_policy_smoke', globals())


if __name__ == "__main__":
    run_tool('run_r1pro_memory_policy_smoke')
