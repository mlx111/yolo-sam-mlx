"""Compatibility wrapper for experience_system.tools.summarize_universal_experience."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('summarize_universal_experience', globals())


if __name__ == "__main__":
    run_tool('summarize_universal_experience')
