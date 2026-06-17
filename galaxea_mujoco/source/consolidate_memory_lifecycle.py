"""Apply STM/LTM lifecycle consolidation to a universal experience library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, consolidate_memory_lifecycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote STM entries to LTM and evict low-value STM overflow.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--stm-capacity", type=int, default=30)
    parser.add_argument("--min-retrieval-count", type=int, default=3)
    parser.add_argument("--min-write-score", type=float, default=0.65)
    parser.add_argument("--evict-batch-size", type=int, default=5)
    parser.add_argument("--no-promote-real", action="store_true")
    parser.add_argument("--no-promote-failures", action="store_true")
    parser.add_argument("--no-promote-validated-success", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    library.entries, report = consolidate_memory_lifecycle(
        library.entries,
        stm_capacity=args.stm_capacity,
        min_retrieval_count=args.min_retrieval_count,
        min_write_score=args.min_write_score,
        promote_real=not args.no_promote_real,
        promote_failures=not args.no_promote_failures,
        promote_validated_success=not args.no_promote_validated_success,
        evict_batch_size=args.evict_batch_size,
    )
    library.save(args.output)
    report = {"input": str(args.input), "output": str(args.output), **report}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input_count": report["input_count"],
        "output_count": report["output_count"],
        "removed_count": report["removed_count"],
        "stm_count": report["stm_count"],
        "ltm_count": report["ltm_count"],
        "promoted_count": len(report["promoted"]),
        "evicted_count": len(report["evicted"]),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
