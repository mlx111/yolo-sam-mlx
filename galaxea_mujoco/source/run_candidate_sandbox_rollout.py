"""Compatibility wrapper for experience_system.tools.run_candidate_sandbox_rollout."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('run_candidate_sandbox_rollout', globals())


if __name__ == "__main__":
    run_tool('run_candidate_sandbox_rollout')
