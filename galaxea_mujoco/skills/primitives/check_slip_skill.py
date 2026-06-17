from __future__ import annotations

from pathlib import Path

from skills.primitives.recovery_skills import CheckSlipSkill


def load_skill(path: str | Path | None = None) -> CheckSlipSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return CheckSlipSkill(config_path)
