"""Compatibility wrapper for experience_system.tools.apply_universal_critic."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('apply_universal_critic', globals())


if __name__ == "__main__":
    run_tool('apply_universal_critic')
