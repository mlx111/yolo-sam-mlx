"""Build a real-episode validation, sensor-coverage, and optional import report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_adapters import RealEpisodeAdapter
from experience_core import ExperienceLibrary, sensor_quality_report, validate_experience_entry, validate_raw_real_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real-format episodes, summarize sensor evidence, and optionally import valid episodes."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="single real episode JSON")
    source.add_argument("--episode-dir", type=Path, help="directory containing episode.json/result.json")
    source.add_argument("--batch-dir", type=Path, help="directory containing episode dirs or episode JSON files")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--source", choices=["real", "pseudo_real"], default="real")
    parser.add_argument("--backend", default="real_robot")
    parser.add_argument("--check-refs", action="store_true", help="warn when referenced sensor/log/keyframe files are missing")
    parser.add_argument("--import-valid", action="store_true", help="write valid normalized episodes into the universal library")
    parser.add_argument("--no-write-policy", action="store_true", help="when importing, append/replace entries without write policy")
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def _load_raw_episodes(args: argparse.Namespace, adapter: RealEpisodeAdapter) -> tuple[list[dict[str, Any]], Path]:
    if args.batch_dir is not None:
        return adapter.collect_batch_sources(args.batch_dir), args.batch_dir
    if args.episode_dir is not None:
        return [adapter.collect_episode_dir(args.episode_dir)], args.episode_dir
    return [json.loads(args.input.read_text(encoding="utf-8"))], args.input.parent


def _episode_root(raw: dict[str, Any], fallback: Path) -> Path:
    raw_refs = raw.get("raw_refs") if isinstance(raw.get("raw_refs"), dict) else {}
    root = raw.get("_episode_root") or raw_refs.get("episode_dir")
    return Path(str(root)) if root else fallback


def _issue_counts(issues: list[dict[str, Any]]) -> tuple[int, int]:
    errors = sum(1 for issue in issues if issue.get("severity") == "error")
    warnings = sum(1 for issue in issues if issue.get("severity") == "warning")
    return errors, warnings


def _missing_sensor_ref_count(issues: list[dict[str, Any]]) -> int:
    return sum(1 for issue in issues if issue.get("code") == "missing_sensor_ref")


def _modalities(entry: Any) -> set[str]:
    return {str(item) for item in entry.sensor_evidence.modalities or [] if item}


def _sensor_summary(entry: Any, *, check_refs: bool) -> dict[str, Any]:
    summary = dict(entry.sensor_evidence.summary or {})
    quality = sensor_quality_report(entry, check_refs=check_refs)
    return {
        "modalities": sorted(_modalities(entry)),
        "has_rgb": "rgb" in _modalities(entry) or "rgbd" in _modalities(entry),
        "has_rgbd": "rgbd" in _modalities(entry),
        "has_lidar": "lidar" in _modalities(entry),
        "has_wrist_force": "wrist_force" in _modalities(entry),
        "max_wrist_force_norm": summary.get("max_wrist_force_norm", 0.0),
        "lidar_ray_count": summary.get("lidar_ray_count"),
        "nearest_obstacle_distance": summary.get("nearest_obstacle_distance"),
        "evidence_ref_count": len(entry.sensor_evidence.evidence_refs or {}),
        "quality": quality,
    }


def _episode_report(
    raw: dict[str, Any],
    *,
    adapter: RealEpisodeAdapter,
    fallback_root: Path,
    source: str,
    check_refs: bool,
) -> tuple[dict[str, Any], Any, bool]:
    raw_report = validate_raw_real_episode(raw, root=_episode_root(raw, fallback_root), check_refs=check_refs)
    entry = adapter.normalize_episode(raw, source=source)
    # Raw validation resolves paths relative to the episode root. Avoid running a
    # second reference pass on normalized entries because normalized raw_refs can
    # contain relative paths that are only meaningful under that root.
    normalized_issues = validate_experience_entry(entry, check_refs=False)
    issues = list(raw_report["issues"]) + [
        {"severity": issue.get("severity", "error"), "code": f"normalized_{issue.get('code', 'issue')}", **issue}
        for issue in normalized_issues
    ]
    error_count, warning_count = _issue_counts(issues)
    sensor = _sensor_summary(entry, check_refs=check_refs)
    report = {
        "episode_id": raw_report["episode_id"] or str(raw.get("episode_id") or ""),
        "experience_id": entry.experience_id,
        "scenario_id": entry.scenario_id,
        "condition_id": entry.condition_id,
        "source": entry.source,
        "backend": entry.backend,
        "robot_type": entry.robot.robot_type,
        "object_class": entry.object_state.object_class,
        "success": bool(entry.result.get("success", False)),
        "memory_partition": entry.memory_partition,
        "memory_role": str(entry.memory_tags.get("memory_role") or ""),
        "skill_count": len(entry.skill_sequence),
        "sensor_evidence": sensor,
        "missing_sensor_ref_count": max(_missing_sensor_ref_count(issues), int(sensor["quality"]["sensor_missing_ref_count"])),
        "error_count": error_count,
        "warning_count": warning_count,
        "passed": error_count == 0,
        "issues": issues,
    }
    return report, entry, error_count == 0


def build_real_episode_import_report(args: argparse.Namespace) -> dict[str, Any]:
    adapter = RealEpisodeAdapter(default_backend=args.backend)
    raw_episodes, fallback_root = _load_raw_episodes(args, adapter)
    library = ExperienceLibrary.load(args.universal_experience_lib)
    original_count = len(library.entries)

    episodes = []
    import_records = []
    modality_counter: Counter[str] = Counter()
    validation_counter: Counter[str] = Counter()
    partition_counter: Counter[str] = Counter()
    imported_count = 0

    for raw in raw_episodes:
        episode, entry, valid = _episode_report(
            raw,
            adapter=adapter,
            fallback_root=fallback_root,
            source=args.source,
            check_refs=args.check_refs,
        )
        episodes.append(episode)
        validation_counter[entry.validation_status or ""] += 1
        partition_counter[entry.memory_partition or ""] += 1
        for modality in episode["sensor_evidence"]["modalities"]:
            modality_counter[modality] += 1

        if args.import_valid and valid:
            if args.no_write_policy:
                library.add(entry)
                write_policy = {
                    "decision": "write",
                    "reason": "write_policy_disabled",
                    "stored_experience_id": entry.experience_id,
                }
            else:
                write_policy = library.add_with_policy(entry)
            imported_count += 1 if write_policy.get("decision") != "skip" else 0
            import_records.append({
                "episode_id": episode["episode_id"],
                "experience_id": entry.experience_id,
                "stored_experience_id": write_policy.get("stored_experience_id", ""),
                "memory_partition": entry.memory_partition,
                "write_policy": write_policy,
            })

    if args.import_valid:
        library.save(args.universal_experience_lib)

    error_count = sum(int(item["error_count"]) for item in episodes)
    warning_count = sum(int(item["warning_count"]) for item in episodes)
    rgbd_episode_count = sum(1 for item in episodes if item["sensor_evidence"]["has_rgbd"])
    lidar_episode_count = sum(1 for item in episodes if item["sensor_evidence"]["has_lidar"])
    wrist_force_episode_count = sum(1 for item in episodes if item["sensor_evidence"]["has_wrist_force"])
    rgb_episode_count = sum(1 for item in episodes if item["sensor_evidence"]["has_rgb"])
    sensor_complete_episode_count = sum(
        1
        for item in episodes
        if item["sensor_evidence"]["has_rgbd"]
        and item["sensor_evidence"]["has_lidar"]
        and item["sensor_evidence"]["has_wrist_force"]
    )
    missing_sensor_ref_count = sum(int(item["missing_sensor_ref_count"]) for item in episodes)
    real_success_count = sum(1 for item in episodes if item["success"])
    real_failure_count = len(episodes) - real_success_count
    real_memory_count = sum(1 for item in episodes if item["memory_partition"] == "real_memory")
    failed_memory_count = sum(1 for item in episodes if item["memory_partition"] == "failed_memory")

    return {
        "report_type": "real_episode_import_report",
        "source_mode": args.source,
        "check_refs": bool(args.check_refs),
        "import_valid": bool(args.import_valid),
        "universal_experience_lib": str(args.universal_experience_lib),
        "library_entry_count_before": original_count,
        "library_entry_count_after": len(library.entries),
        "validated_count": len(episodes),
        "valid_real_episode_count": sum(1 for item in episodes if item["passed"]),
        "imported_count": imported_count,
        "error_count": error_count,
        "warning_count": warning_count,
        "real_memory_count": real_memory_count,
        "failed_memory_count": failed_memory_count,
        "rgb_episode_count": rgb_episode_count,
        "rgbd_episode_count": rgbd_episode_count,
        "lidar_episode_count": lidar_episode_count,
        "wrist_force_episode_count": wrist_force_episode_count,
        "sensor_complete_episode_count": sensor_complete_episode_count,
        "missing_sensor_ref_count": missing_sensor_ref_count,
        "real_episode_count": len(episodes),
        "sensor_missing_ref_count": missing_sensor_ref_count,
        "real_success_count": real_success_count,
        "real_failure_count": real_failure_count,
        "real_memory_write_count": sum(1 for item in import_records if item["memory_partition"] == "real_memory"),
        "failed_memory_write_count": sum(1 for item in import_records if item["memory_partition"] == "failed_memory"),
        "sensor_modality_distribution": dict(sorted(modality_counter.items())),
        "validation_status_distribution": dict(sorted(validation_counter.items())),
        "memory_partition_distribution": dict(sorted(partition_counter.items())),
        "imports": import_records,
        "episodes": episodes,
        "safe_paper_wording": (
            "We report the number and sensor coverage of imported real-format episodes "
            "before using them as memory evidence."
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_real_episode_import_report(args)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
