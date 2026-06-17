from pathlib import Path

from skills.primitives.gripper_state_skills import DetectGripperStateSkill


def load_skill(path: str | Path | None = None) -> DetectGripperStateSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return DetectGripperStateSkill(config_path)
