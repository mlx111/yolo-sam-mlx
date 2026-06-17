"""Build sandbox parameter-sweep profiles from sim-real gap memories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, build_group_sandbox_parameter_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sandbox_parameter_profile_v1 from gap memories.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--condition", default="")
    parser.add_argument("--object-class", default="")
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    profiles_by_key = build_group_sandbox_parameter_profiles(library.entries)
    profiles = []
    for key, profile in profiles_by_key.items():
        robot_type, scenario, condition, object_class = key
        if args.scenario and scenario != args.scenario:
            continue
        if args.condition and condition != args.condition:
            continue
        if args.object_class and object_class != args.object_class:
            continue
        profiles.append(profile.to_dict())
    profiles.sort(key=lambda item: (item.get("group_key", {}).get("scenario_id", ""), item.get("group_key", {}).get("condition_id", "")))
    payload: dict[str, Any]
    if len(profiles) == 1:
        payload = profiles[0]
    else:
        payload = {
            "schema_version": "sandbox_parameter_profile_collection_v1",
            "input": str(args.input),
            "profile_count": len(profiles),
            "profiles": profiles,
        }
    _write_json(args.save, payload)
    print(json.dumps({
        "profile_count": len(profiles),
        "save": str(args.save),
        "profile_ids": [item.get("profile_id", "") for item in profiles],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
