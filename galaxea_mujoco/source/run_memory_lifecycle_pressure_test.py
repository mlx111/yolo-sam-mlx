"""Compatibility wrapper for experience_system.tools.run_memory_lifecycle_pressure_test."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('run_memory_lifecycle_pressure_test', globals())


if __name__ == "__main__":
    run_tool('run_memory_lifecycle_pressure_test')
