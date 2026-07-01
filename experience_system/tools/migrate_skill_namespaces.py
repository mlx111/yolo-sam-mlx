from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an experience library with inferred skill namespaces and default skill catalogs.")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace: {args.output}")
    library = ExperienceLibrary.load(args.library)
    library.save(args.output)
    print(f"wrote {len(library.entries)} entries with {len(library.skill_catalogs)} skill catalogs to {args.output}")


if __name__ == "__main__":
    main()
