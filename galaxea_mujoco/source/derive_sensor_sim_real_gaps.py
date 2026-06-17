"""Compatibility wrapper for experience_system.tools.derive_sensor_sim_real_gaps."""

from __future__ import annotations

from source._experience_system_wrapper import export_tool, run_tool

export_tool('derive_sensor_sim_real_gaps', globals())


if __name__ == "__main__":
    run_tool('derive_sensor_sim_real_gaps')
