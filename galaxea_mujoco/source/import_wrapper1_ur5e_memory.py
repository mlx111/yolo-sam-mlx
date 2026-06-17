"""Compatibility wrapper for experience_system.tools.import_wrapper1_ur5e_memory."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('import_wrapper1_ur5e_memory', globals())


if __name__ == "__main__":
    run_tool('import_wrapper1_ur5e_memory')
