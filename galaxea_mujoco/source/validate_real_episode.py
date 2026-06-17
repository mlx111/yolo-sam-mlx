"""Validate real/pseudo-real episode files before importing them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import RealEpisodeAdapter
from experience_core import validate_experience_entry, validate_raw_real_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real/pseudo-real episode JSON or directories.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="episode JSON path")
    source.add_argument("--episode-dir", type=Path, help="episode directory containing episode.json/result.json")
    source.add_argument("--batch-dir", type=Path, help="directory containing episode dirs or episode JSON files")
    parser.add_argument("--source", choices=["real", "pseudo_real"], default="real")
    parser.add_argument("--backend", default="real_robot")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--check-refs", action="store_true", help="check keyframe/log/video paths exist")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if validation errors exist")
    return parser.parse_args()


def _episode_root(raw: dict[str, Any], fallback: Path | None = None) -> Path | None:
    raw_refs = raw.get("raw_refs") if isinstance(raw.get("raw_refs"), dict) else {}
    root = raw.get("_episode_root") or raw_refs.get("episode_dir")
    if root:
        return Path(str(root))
    return fallback


def main() -> None:
    args = parse_args()
    adapter = RealEpisodeAdapter(default_backend=args.backend)
    if args.batch_dir is not None:
        raw_episodes = adapter.collect_batch_sources(args.batch_dir)
        fallback_root = args.batch_dir
    elif args.episode_dir is not None:
        raw_episodes = [adapter.collect_episode_dir(args.episode_dir)]
        fallback_root = args.episode_dir
    else:
        raw_episodes = [json.loads(args.input.read_text(encoding="utf-8"))]
        fallback_root = args.input.parent

    entries = []
    total_errors = 0
    total_warnings = 0
    for raw in raw_episodes:
        raw_report = validate_raw_real_episode(raw, root=_episode_root(raw, fallback_root), check_refs=args.check_refs)
        entry = adapter.normalize_episode(raw, source=args.source)
        entry_issues = validate_experience_entry(entry, check_refs=args.check_refs)
        issues = list(raw_report["issues"]) + [
            {"severity": issue.get("severity", "error"), "code": f"normalized_{issue.get('code', 'issue')}", **issue}
            for issue in entry_issues
        ]
        error_count = sum(1 for issue in issues if issue.get("severity") == "error")
        warning_count = sum(1 for issue in issues if issue.get("severity") == "warning")
        total_errors += error_count
        total_warnings += warning_count
        entries.append({
            "episode_id": raw_report["episode_id"],
            "experience_id": entry.experience_id,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "robot_type": entry.robot.robot_type,
            "object_class": entry.object_state.object_class,
            "skill_count": len(entry.skill_sequence),
            "error_count": error_count,
            "warning_count": warning_count,
            "passed": error_count == 0,
            "issues": issues,
        })

    report = {
        "validated_count": len(entries),
        "error_count": total_errors,
        "warning_count": total_warnings,
        "passed": total_errors == 0,
        "entries": entries,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if args.strict and total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
