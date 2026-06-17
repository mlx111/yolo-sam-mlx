from pathlib import Path

from skills.primitives.perception_skills import SelectCorrectObjectSkill


def load_skill(path: str | Path | None = None) -> SelectCorrectObjectSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return SelectCorrectObjectSkill(config_path)
