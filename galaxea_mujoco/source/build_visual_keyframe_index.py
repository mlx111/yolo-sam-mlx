"""Build a CLIP+FAISS visual keyframe index for a universal experience library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, VisualRetrievalIndex, image_paths_from_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual keyframe index for universal experience memory.")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None, help="base path for relative keyframe paths")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def build_visual_index(
    library: ExperienceLibrary,
    *,
    index_dir: Path,
    base_dir: Path,
    model_name: str | None = None,
) -> dict:
    index = VisualRetrievalIndex(model_name=model_name)

    entries = []
    indexed_count = 0
    image_count = 0
    for entry in library.entries:
        image_paths = image_paths_from_entry(entry, base_dir=base_dir)
        added = index.add(entry.experience_id, image_paths)
        if added:
            indexed_count += 1
            image_count += added
        entries.append({
            "experience_id": entry.experience_id,
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "image_count": len(image_paths),
            "indexed_image_count": added,
            "image_paths": image_paths,
        })

    index.save(index_dir)
    return {
        "index_dir": str(index_dir),
        "model_name": index.model_name,
        "device": index.device,
        "entry_count": len(library.entries),
        "indexed_entry_count": indexed_count,
        "indexed_image_count": image_count,
        "faiss_size": index.size,
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    base_dir = args.base_dir or args.universal_experience_lib.parent
    report = {
        "input": str(args.universal_experience_lib),
        **build_visual_index(
            library,
            index_dir=args.index_dir,
            base_dir=base_dir,
            model_name=args.model_name,
        ),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("index_dir", "indexed_entry_count", "indexed_image_count", "faiss_size")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
