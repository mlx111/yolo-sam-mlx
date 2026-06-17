"""Compatibility wrapper for experience_system.tools.generate_pseudo_real_episodes."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('generate_pseudo_real_episodes', globals())


if __name__ == "__main__":
    run_tool('generate_pseudo_real_episodes')
