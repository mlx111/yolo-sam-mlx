"""Query a universal visual keyframe index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, RetrievalQuery, VisualRetrievalIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query visual keyframe similarity for universal experience memory.")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query-image", type=Path, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--condition", default="")
    parser.add_argument("--robot-type", default="")
    parser.add_argument("--object-class", default="")
    parser.add_argument("--visual-weight", type=float, default=0.12)
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    by_id = {entry.experience_id: entry for entry in library.entries}
    index = VisualRetrievalIndex()
    index.load(args.index_dir)
    query_images = [str(path.resolve()) for path in args.query_image]
    raw_results = index.search(query_images, top_k=args.top_k)
    visual_scores = {experience_id: score for experience_id, score in raw_results}

    results = []
    for experience_id, score in raw_results:
        entry = by_id.get(experience_id)
        results.append({
            "experience_id": experience_id,
            "visual_similarity": round(float(score), 4),
            "found_in_library": entry is not None,
            "source": entry.source if entry else "",
            "scenario_id": entry.scenario_id if entry else "",
            "condition_id": entry.condition_id if entry else "",
            "robot_type": entry.robot.robot_type if entry else "",
            "success": bool(entry.result.get("success", False)) if entry else None,
            "failure_type": entry.failure_taxonomy.get("failure_type", "") if entry else "",
            "memory_role": entry.memory_tags.get("memory_role", "") if entry else "",
        })

    report = {
        "library": str(args.universal_experience_lib),
        "index_dir": str(args.index_dir),
        "query_images": query_images,
        "top_k": args.top_k,
        "result_count": len(results),
        "results": results,
    }
    if any((args.scenario, args.condition, args.robot_type, args.object_class)):
        structured = library.query_structured(
            RetrievalQuery(
                scenario_id=args.scenario,
                condition_id=args.condition,
                robot_type=args.robot_type,
                object_class=args.object_class,
                require_scenario=bool(args.scenario),
                visual_scores=visual_scores,
                visual_weight=args.visual_weight,
                top_k=args.top_k,
            )
        )
        report["structured_results"] = [
            {
                "experience_id": match.entry.experience_id,
                "score": match.score,
                "visual_similarity": round(float(visual_scores.get(match.entry.experience_id, 0.0)), 4),
                "scenario_id": match.entry.scenario_id,
                "condition_id": match.entry.condition_id,
                "source": match.entry.source,
                "success": bool(match.entry.result.get("success", False)),
                "explanation": match.explanation,
            }
            for match in structured
        ]
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
