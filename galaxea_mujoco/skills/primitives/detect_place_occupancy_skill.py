from pathlib import Path

from skills.primitives.place_area_skills import DetectPlaceOccupancySkill


def load_skill(path: str | Path | None = None) -> DetectPlaceOccupancySkill:
    config_path = Path(path) if path is not None else Path(__file__).with_suffix(".json")
    return DetectPlaceOccupancySkill(config_path)
