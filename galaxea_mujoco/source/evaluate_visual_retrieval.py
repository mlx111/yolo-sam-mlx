"""Evaluate visual keyframe retrieval for a universal experience library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import ExperienceLibrary, RetrievalQuery, VisualRetrievalIndex, image_paths_from_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP/FAISS visual retrieval over universal experience keyframes.")
    parser.add_argument("--universal-experience-lib", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visual-weight", type=float, default=0.12)
    parser.add_argument("--all-keyframes", action="store_true", help="query every keyframe instead of first keyframe per entry")
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args()


def _rank(matches: list, experience_id: str) -> int:
    for index, match in enumerate(matches, 1):
        if match.entry.experience_id == experience_id:
            return index
    return 0


def _condition_hit(results: list[tuple[str, float]], by_id: dict, entry, top_k: int) -> bool:
    for experience_id, _score in results[:top_k]:
        match = by_id.get(experience_id)
        if match is None:
            continue
        if match.scenario_id == entry.scenario_id and match.condition_id == entry.condition_id:
            return True
    return False


def _first_condition_hit(results: list[tuple[str, float]], by_id: dict, entry) -> bool:
    if not results:
        return False
    match = by_id.get(results[0][0])
    return bool(match and match.scenario_id == entry.scenario_id and match.condition_id == entry.condition_id)


def evaluate_visual_retrieval(
    library: ExperienceLibrary,
    *,
    index_dir: Path,
    base_dir: Path,
    top_k: int = 5,
    visual_weight: float = 0.12,
    all_keyframes: bool = False,
) -> dict:
    index = VisualRetrievalIndex()
    index.load(index_dir)
    by_id = {entry.experience_id: entry for entry in library.entries}

    entries_with_keyframes = []
    keyframe_count = 0
    for entry in library.entries:
        paths = image_paths_from_entry(entry, base_dir=base_dir)
        keyframe_count += len(paths)
        if paths:
            entries_with_keyframes.append((entry, paths))

    queries = []
    rank_deltas = []
    for entry, paths in entries_with_keyframes:
        query_paths = paths if all_keyframes else paths[:1]
        for image_path in query_paths:
            visual_results = index.search([image_path], top_k=top_k)
            visual_scores = {experience_id: score for experience_id, score in visual_results}
            structured_without = library.query_structured(
                RetrievalQuery(
                    scenario_id=entry.scenario_id,
                    condition_id=entry.condition_id,
                    robot_type=entry.robot.robot_type,
                    object_class=entry.object_state.object_class,
                    top_k=max(len(library.entries), top_k),
                )
            )
            structured_with = library.query_structured(
                RetrievalQuery(
                    scenario_id=entry.scenario_id,
                    condition_id=entry.condition_id,
                    robot_type=entry.robot.robot_type,
                    object_class=entry.object_state.object_class,
                    visual_scores=visual_scores,
                    visual_weight=visual_weight,
                    top_k=max(len(library.entries), top_k),
                )
            )
            rank_without = _rank(structured_without, entry.experience_id)
            rank_with = _rank(structured_with, entry.experience_id)
            if rank_without and rank_with:
                rank_deltas.append(rank_without - rank_with)
            top_ids = [experience_id for experience_id, _score in visual_results[:top_k]]
            queries.append({
                "experience_id": entry.experience_id,
                "scenario_id": entry.scenario_id,
                "condition_id": entry.condition_id,
                "query_image": image_path,
                "visual_top1": top_ids[0] if top_ids else "",
                "visual_topk": top_ids,
                "top1_self_hit": bool(top_ids and top_ids[0] == entry.experience_id),
                "topk_self_hit": entry.experience_id in top_ids,
                "top1_same_condition": _first_condition_hit(visual_results, by_id, entry),
                "topk_same_condition": _condition_hit(visual_results, by_id, entry, top_k),
                "rank_without_visual": rank_without,
                "rank_with_visual": rank_with,
                "rank_delta": (rank_without - rank_with) if rank_without and rank_with else 0,
            })

    query_count = len(queries)

    def rate(key: str) -> float:
        if not query_count:
            return 0.0
        return round(sum(1 for item in queries if item.get(key)) / query_count, 4)

    indexed_entry_ids = set(index._eid_to_ids.keys())
    report = {
        "entry_count": len(library.entries),
        "entries_with_keyframes": len(entries_with_keyframes),
        "keyframe_count": keyframe_count,
        "indexed_entry_count": len(indexed_entry_ids),
        "indexed_image_count": index.size,
        "indexed_keyframe_coverage": round(index.size / keyframe_count, 4) if keyframe_count else 0.0,
        "top_k": top_k,
        "visual_weight": visual_weight,
        "query_count": query_count,
        "top1_self_hit_rate": rate("top1_self_hit"),
        "topk_self_hit_rate": rate("topk_self_hit"),
        "top1_same_condition_rate": rate("top1_same_condition"),
        "topk_same_condition_rate": rate("topk_same_condition"),
        "rank_delta_avg": round(mean(rank_deltas), 4) if rank_deltas else 0.0,
        "queries": queries,
    }
    return report


def main() -> None:
    args = parse_args()
    library = ExperienceLibrary.load(args.universal_experience_lib)
    report = {
        "library": str(args.universal_experience_lib),
        "index_dir": str(args.index_dir),
        **evaluate_visual_retrieval(
            library,
            index_dir=args.index_dir,
            base_dir=args.base_dir or args.universal_experience_lib.parent,
            top_k=args.top_k,
            visual_weight=args.visual_weight,
            all_keyframes=args.all_keyframes,
        ),
    }
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "entry_count",
        "entries_with_keyframes",
        "keyframe_count",
        "indexed_entry_count",
        "indexed_image_count",
        "query_count",
        "top1_self_hit_rate",
        "topk_self_hit_rate",
        "top1_same_condition_rate",
        "topk_same_condition_rate",
        "rank_delta_avg",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
