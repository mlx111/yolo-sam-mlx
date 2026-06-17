"""Compatibility wrapper for experience_system.tools.merge_experience_lessons."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool("merge_experience_lessons", globals())


if __name__ == "__main__":
    run_tool("merge_experience_lessons")
