from pathlib import Path

from skills.primitives.gripper_state_skills import ResyncGrippersSkill


def load_skill(path: str | Path | None = None) -> ResyncGrippersSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return ResyncGrippersSkill(config_path)
