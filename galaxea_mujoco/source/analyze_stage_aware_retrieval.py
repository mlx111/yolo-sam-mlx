"""Compatibility wrapper for experience_system.tools.analyze_stage_aware_retrieval."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool("analyze_stage_aware_retrieval", globals())


if __name__ == "__main__":
    run_tool("analyze_stage_aware_retrieval")
