from pathlib import Path

from skills.primitives.place_area_skills import ChooseAlternatePlaceSkill


def load_skill(path: str | Path | None = None) -> ChooseAlternatePlaceSkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return ChooseAlternatePlaceSkill(config_path)
