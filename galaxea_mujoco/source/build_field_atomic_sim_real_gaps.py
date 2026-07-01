from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE_ROOT = ROOT.parent / "experience_system"
for path in (ROOT, EXPERIENCE_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from experience_core import (  # noqa: E402
    ExperienceLibrary,
    GALAXEA_R1PRO_TORSO_NAMESPACE,
    apply_pair_and_gap,
    is_field_atomic_entry,
    pair_sim_real_experiences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Galaxea field-atomic sim-real gap memories.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-pair-score", type=float, default=0.55)
    return parser.parse_args()


def _field_atomic_entries(library: ExperienceLibrary):
    return [
        entry for entry in library.entries
        if is_field_atomic_entry(entry) and entry.skill_namespace == GALAXEA_R1PRO_TORSO_NAMESPACE
    ]


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.input)
    field_entries = _field_atomic_entries(library)
    pairs = pair_sim_real_experiences(field_entries, min_pair_score=float(args.min_pair_score))
    updated_field = apply_pair_and_gap(field_entries, pairs)
    by_id = {entry.experience_id: entry for entry in updated_field}
    updated_entries = [by_id.get(entry.experience_id, entry) for entry in library.entries]
    library.entries = updated_entries
    library.save(args.output)

    report_pairs = []
    by_updated = {entry.experience_id: entry for entry in library.entries}
    for pair in pairs:
        real_entry = by_updated.get(str(pair.get("real_experience_id")))
        sim_entry = by_updated.get(str(pair.get("sim_experience_id")))
        gap = real_entry.sim_real_gap if real_entry is not None else None
        report_pairs.append({
            **pair,
            "sim_source": sim_entry.source if sim_entry is not None else "",
            "real_source": real_entry.source if real_entry is not None else "",
            "gap_id": gap.gap_id if gap is not None else "",
            "gap_score": gap.gap_score if gap is not None else 0.0,
            "gap_uncertainty": gap.uncertainty if gap is not None else 0.0,
            "gap_type": gap.outcome_gap.get("type", "") if gap is not None else "",
        })
    report = {
        "schema_version": "galaxea_field_atomic_sim_real_gap_report_v1",
        "input": str(args.input),
        "output": str(args.output),
        "entry_count": len(library.entries),
        "field_atomic_entry_count": len(field_entries),
        "pair_count": len(pairs),
        "pairs": report_pairs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"field_atomic_entry_count": len(field_entries), "pair_count": len(pairs), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
