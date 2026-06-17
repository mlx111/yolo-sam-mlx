"""Compatibility wrapper for experience_system.tools.build_real_episode_import_report."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('build_real_episode_import_report', globals())


if __name__ == "__main__":
    run_tool('build_real_episode_import_report')
