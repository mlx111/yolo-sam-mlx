"""Compatibility wrapper for experience_system.tools.populate_pseudo_real_keyframes."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('populate_pseudo_real_keyframes', globals())


if __name__ == "__main__":
    run_tool('populate_pseudo_real_keyframes')
