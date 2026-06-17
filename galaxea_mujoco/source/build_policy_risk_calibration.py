"""Build policy risk-transfer calibration from universal experience memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, build_policy_risk_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy risk-transfer calibration JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    calibration = build_policy_risk_calibration(library.entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "group_count": calibration.get("group_count", 0),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
