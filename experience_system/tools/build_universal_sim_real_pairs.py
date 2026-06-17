"""Build sim-real pairs and gap signatures for a universal experience library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, apply_pair_and_gap, pair_sim_real_experiences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair simulation and real/pseudo-real entries in a universal memory library.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pair-score", type=float, default=0.55)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    pairs = pair_sim_real_experiences(library.entries, min_pair_score=args.min_pair_score)
    library.entries = apply_pair_and_gap(library.entries, pairs)
    library.save(args.output)
    by_id = {entry.experience_id: entry for entry in library.entries}
    report_pairs = []
    for pair in pairs:
        real_entry = by_id.get(str(pair.get("real_experience_id")))
        gap_score = real_entry.sim_real_gap.gap_score if real_entry is not None else pair.get("gap_score", 0.0)
        item = dict(pair)
        item["gap_score"] = gap_score
        if real_entry is not None:
            item["gap_type"] = real_entry.sim_real_gap.outcome_gap.get("type", "")
            item["gap_uncertainty"] = real_entry.sim_real_gap.uncertainty
        report_pairs.append(item)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "entry_count": len(library),
        "pair_count": len(pairs),
        "pairs": report_pairs,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"entry_count": len(library), "pair_count": len(pairs), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
