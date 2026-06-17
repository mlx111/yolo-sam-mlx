"""Consolidate duplicate low-risk universal experience entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, consolidate_experiences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate duplicate low-risk universal experience entries.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    library.entries, report = consolidate_experiences(library.entries)
    library.save(args.output)
    report = {"input": str(args.input), "output": str(args.output), **report}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input_count": report["input_count"],
        "output_count": report["output_count"],
        "removed_count": report["removed_count"],
        "merged_group_count": report["merged_group_count"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
