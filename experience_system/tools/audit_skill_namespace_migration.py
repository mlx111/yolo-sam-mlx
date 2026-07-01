from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit inferred skill namespaces in an experience library.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.library)
    counts = Counter(entry.skill_namespace or "unknown" for entry in library.entries)
    unknown_entries = [
        {
            "experience_id": entry.experience_id,
            "robot_type": entry.robot.robot_type,
            "backend": entry.backend,
            "actions": [item.name for item in entry.skill_sequence if item.name],
        }
        for entry in library.entries
        if (entry.skill_namespace or "unknown") == "unknown"
    ]
    report = {
        "schema_version": "skill_namespace_migration_audit_v1",
        "library": str(args.library),
        "entry_count": len(library.entries),
        "skill_catalog_namespaces": sorted(library.skill_catalogs),
        "namespace_distribution": dict(sorted(counts.items())),
        "unknown_count": len(unknown_entries),
        "unknown_entries": unknown_entries,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
