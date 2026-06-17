"""Compatibility wrapper for experience_system.tools.compare_policy_baseline."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('compare_policy_baseline', globals())


if __name__ == "__main__":
    run_tool('compare_policy_baseline')
