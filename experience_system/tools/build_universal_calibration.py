"""Build sandbox calibration records from universal sim-real gap memories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, apply_sandbox_calibration, compute_group_calibrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped sandbox calibration from paired sim-real gap memories.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    calibrations = compute_group_calibrations(library.entries)
    library.entries = apply_sandbox_calibration(library.entries)
    library.save(args.output)

    groups = []
    for key, calibration in sorted(calibrations.items(), key=lambda item: item[0]):
        groups.append({
            "robot_type": key[0],
            "scenario_id": key[1],
            "condition_id": key[2],
            "object_class": key[3],
            "calibration_id": calibration.calibration_id,
            "source_gap_ids": calibration.source_gap_ids,
            "object_pose_bias": calibration.object_pose_bias,
            "perception_noise_bias": calibration.perception_noise_bias,
            "contact_success_bias": calibration.contact_success_bias,
            "slip_risk_bias": calibration.slip_risk_bias,
            "calibration_confidence": calibration.calibration_confidence,
            "details": calibration.details,
        })

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "entry_count": len(library.entries),
        "group_count": len(groups),
        "groups": groups,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"entry_count": len(library.entries), "group_count": len(groups), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
