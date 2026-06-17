"""Import a generic real or pseudo-real episode into universal experience memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import RealEpisodeAdapter
from experience_core import ExperienceLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import real/pseudo-real episode JSON or directory into universal memory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="episode JSON path")
    source.add_argument("--episode-dir", type=Path, help="episode directory containing episode.json/result.json")
    source.add_argument("--batch-dir", type=Path, help="directory containing episode dirs or episode JSON files")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--source", choices=["real", "pseudo_real"], default="real")
    parser.add_argument("--backend", default="real_robot")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-write-policy", action="store_true", help="append entries without write-time gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = RealEpisodeAdapter(default_backend=args.backend)
    library = ExperienceLibrary.load(args.universal_experience_lib)
    imported = []

    if args.batch_dir is not None:
        raw_episodes = adapter.collect_batch_sources(args.batch_dir)
    elif args.episode_dir is not None:
        raw_episodes = [adapter.collect_episode_dir(args.episode_dir)]
    else:
        raw_episodes = [json.loads(args.input.read_text(encoding="utf-8"))]

    for raw_episode in raw_episodes:
        entry = adapter.normalize_episode(raw_episode, source=args.source)
        write_policy = {"decision": "write", "reason": "write_policy_disabled", "stored_experience_id": entry.experience_id}
        if args.no_write_policy:
            library.add(entry)
        else:
            write_policy = library.add_with_policy(entry)
        imported.append({
            "universal_experience_id": entry.experience_id,
            "stored_experience_id": write_policy.get("stored_experience_id", ""),
            "source": entry.source,
            "backend": entry.backend,
            "robot_type": entry.robot.robot_type,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "success": entry.result.get("success", False),
            "memory_partition": entry.memory_partition,
            "write_policy": write_policy,
        })

    library.save(args.universal_experience_lib)

    report = {
        "imported_count": len(imported),
        "entries": imported,
        "output": str(args.universal_experience_lib),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
