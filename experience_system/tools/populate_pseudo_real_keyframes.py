"""Populate pseudo-real episode keyframes from existing simulation keyframes.

This is a smoke-data utility, not a replacement for real camera collection.
It gives pseudo-real calibration episodes concrete image references so the
generic real episode adapter and visual retrieval pipeline can be exercised
end-to-end before real robot keyframes are available.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_DIR = ROOT / "data/pseudo_real/r1pro_calibration_v1"
DEFAULT_SIM_KEYFRAME_DIR = ROOT / "results/memory/universal_pipeline_calibration_v1/keyframes"

STAGE_ORDER = ["before_task", "after_grasp", "after_lift", "before_place", "after_place"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy simulation keyframes into pseudo-real episode directories.")
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--sim-keyframe-dir", type=Path, default=DEFAULT_SIM_KEYFRAME_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _candidate_source_dirs(episode: dict[str, Any]) -> list[str]:
    scenario_id = str((episode.get("scenario") or {}).get("scenario_id") or "")
    condition_id = str((episode.get("condition") or {}).get("condition_id") or "")
    if scenario_id == "G3":
        if condition_id == "place_occupied":
            return ["G3_place_occupied_g3_default_1", "G3_clean_g3_default_1"]
        return ["G3_clean_g3_default_1", "G3_place_occupied_g3_default_1"]
    if scenario_id == "G4":
        if condition_id == "place_occupied":
            return ["G4_place_occupied_g4_default_1", "G4_clean_g4_default_1"]
        return ["G4_clean_g4_default_1", "G4_place_occupied_g4_default_1"]
    return []


def _select_source_dir(sim_keyframe_dir: Path, episode: dict[str, Any]) -> Path:
    for name in _candidate_source_dirs(episode):
        path = sim_keyframe_dir / name
        if path.is_dir() and any((path / f"{stage}.png").exists() for stage in STAGE_ORDER):
            return path
    episode_id = episode.get("episode_id", "<unknown>")
    raise FileNotFoundError(f"no usable simulation keyframe source for {episode_id}")


def _copy_keyframes(source_dir: Path, target_dir: Path, *, dry_run: bool) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for stage in STAGE_ORDER:
        source = source_dir / f"{stage}.png"
        if not source.exists():
            continue
        target = target_dir / source.name
        if not dry_run:
            shutil.copy2(source, target)
        frames.append({
            "stage": stage,
            "image_path": f"keyframes/{source.name}",
            "description": f"pseudo-real smoke keyframe copied from {source_dir.name}/{source.name}",
            "used_for_retrieval": True,
            "source": "simulation_keyframe_copy",
            "source_image_path": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
        })
    if not frames:
        raise FileNotFoundError(f"no keyframe images found in {source_dir}")
    return frames


def populate_episode(episode_path: Path, sim_keyframe_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    episode = _load_json(episode_path)
    source_dir = _select_source_dir(sim_keyframe_dir, episode)
    episode_dir = episode_path.parent
    keyframe_dir = episode_dir / "keyframes"
    frames = _copy_keyframes(source_dir, keyframe_dir, dry_run=dry_run)
    robot_log = episode_dir / "robot_log.jsonl"
    if not dry_run and not robot_log.exists():
        robot_log.write_text("", encoding="utf-8")

    episode["keyframes"] = frames
    real_ref = dict(episode.get("real_episode_ref") or {})
    real_ref["keyframe_dir"] = str(keyframe_dir)
    real_ref["robot_log_path"] = str(robot_log)
    episode["real_episode_ref"] = real_ref

    metadata = dict(episode.get("metadata") or {})
    metadata["pseudo_real_keyframe_source"] = str(source_dir.relative_to(ROOT) if source_dir.is_relative_to(ROOT) else source_dir)
    metadata["pseudo_real_keyframe_note"] = "Smoke calibration images copied from matching simulation run; replace with real camera keyframes for real deployment."
    episode["metadata"] = metadata

    if not dry_run:
        _write_json(episode_path, episode)

    return {
        "episode_id": episode.get("episode_id"),
        "scenario_id": (episode.get("scenario") or {}).get("scenario_id"),
        "condition_id": (episode.get("condition") or {}).get("condition_id"),
        "source_dir": metadata["pseudo_real_keyframe_source"],
        "keyframe_count": len(frames),
        "episode_path": str(episode_path.relative_to(ROOT) if episode_path.is_relative_to(ROOT) else episode_path),
    }


def main() -> None:
    args = parse_args()
    episode_dir = args.episode_dir if args.episode_dir.is_absolute() else ROOT / args.episode_dir
    sim_keyframe_dir = args.sim_keyframe_dir if args.sim_keyframe_dir.is_absolute() else ROOT / args.sim_keyframe_dir
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode-dir does not exist: {episode_dir}")
    if not sim_keyframe_dir.is_dir():
        raise FileNotFoundError(f"sim-keyframe-dir does not exist: {sim_keyframe_dir}")

    reports = []
    for episode_path in sorted(episode_dir.glob("*/episode.json")):
        reports.append(populate_episode(episode_path, sim_keyframe_dir, dry_run=args.dry_run))

    report = {
        "episode_dir": str(episode_dir.relative_to(ROOT) if episode_dir.is_relative_to(ROOT) else episode_dir),
        "sim_keyframe_dir": str(sim_keyframe_dir.relative_to(ROOT) if sim_keyframe_dir.is_relative_to(ROOT) else sim_keyframe_dir),
        "dry_run": args.dry_run,
        "episode_count": len(reports),
        "keyframe_count": sum(item["keyframe_count"] for item in reports),
        "episodes": reports,
    }
    report_path = episode_dir / "keyframe_population_report.json"
    if not args.dry_run:
        _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
