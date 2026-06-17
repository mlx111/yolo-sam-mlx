"""Derive conservative sim-real gap fields from stored sensor evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, apply_sensor_sim_real_gaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill sensor-derived sim-real gaps for real/pseudo-real entries.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace existing sim_real_gap fields on eligible entries")
    return parser.parse_args()


def _avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    before = {entry.experience_id: entry.sim_real_gap.gap_id for entry in library.entries}
    updated_entries = apply_sensor_sim_real_gaps(library.entries, overwrite=args.overwrite)
    library.entries = updated_entries
    library.save(args.output)

    gap_type = Counter()
    gap_scores: list[float] = []
    uncertainties: list[float] = []
    updated_ids: list[str] = []
    perception_count = 0
    contact_count = 0
    scene_count = 0
    timing_count = 0

    for entry in library.entries:
        old_gap_id = before.get(entry.experience_id, "")
        new_gap_id = entry.sim_real_gap.gap_id
        if entry.source not in {"real", "pseudo_real"}:
            continue
        if not new_gap_id:
            continue
        if not args.overwrite and old_gap_id:
            continue
        updated_ids.append(entry.experience_id)
        gap_type[str(entry.sim_real_gap.outcome_gap.get("type") or "")] += 1
        gap_scores.append(float(entry.sim_real_gap.gap_score or 0.0))
        uncertainties.append(float(entry.sim_real_gap.uncertainty or 0.0))
        perception_count += int(bool(entry.sim_real_gap.perception_gap))
        contact_count += int(bool(entry.sim_real_gap.contact_gap))
        scene_count += int(bool(entry.sim_real_gap.scene_reconstruction_gap))
        timing_count += int(bool(entry.sim_real_gap.timing_gap))

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "overwrite": bool(args.overwrite),
        "entry_count": len(library.entries),
        "sensor_gap_entry_count": len(updated_ids),
        "perception_gap_from_rgbd_count": perception_count,
        "contact_gap_from_wrist_force_count": contact_count,
        "scene_gap_from_lidar_count": scene_count,
        "timing_gap_count": timing_count,
        "sensor_gap_confidence_avg": round(max(0.0, 1.0 - _avg(uncertainties)), 4) if uncertainties else 0.0,
        "gap_score_avg": _avg(gap_scores),
        "gap_uncertainty_avg": _avg(uncertainties),
        "gap_type_distribution": dict(gap_type),
        "updated_experience_ids": updated_ids,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
