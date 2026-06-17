"""Create a fillable R1Pro real-episode directory template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an R1Pro real-episode template directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-id", default="replace_with_unique_episode_id")
    parser.add_argument("--scenario", default="G1")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--task-name", default="grasp_place_demo")
    parser.add_argument("--robot-id", default="r1pro_real_001")
    parser.add_argument("--object-class", default="cube")
    parser.add_argument("--force", action="store_true", help="overwrite existing episode.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rgb", "depth", "lidar", "force", "keyframes", "video", "logs"):
        (output_dir / name).mkdir(exist_ok=True)

    template_path = ROOT / "templates" / "r1pro_real_episode_template.json"
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["episode_id"] = args.episode_id
    payload["scenario_id"] = args.scenario
    payload["condition_id"] = args.condition
    payload["task_name"] = args.task_name
    payload["robot_id"] = args.robot_id
    payload["object_class"] = args.object_class
    payload["target_object"] = f"target_{args.object_class}"
    payload["robot"]["robot_id"] = args.robot_id
    payload["scenario"]["scenario_id"] = args.scenario
    payload["condition"]["condition_id"] = args.condition
    payload["task"]["name"] = args.task_name
    payload["task"]["object_class"] = args.object_class
    payload["real_episode_ref"]["raw_episode_id"] = args.episode_id
    payload["metadata"]["generated_from"] = str(template_path)

    episode_json = output_dir / "episode.json"
    if episode_json.exists() and not args.force:
        raise SystemExit(f"{episode_json} already exists; use --force to overwrite")
    episode_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "output_dir": str(output_dir),
        "episode_json": str(episode_json),
        "created_dirs": [str(output_dir / name) for name in ("rgb", "depth", "lidar", "force", "keyframes", "video", "logs")],
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
