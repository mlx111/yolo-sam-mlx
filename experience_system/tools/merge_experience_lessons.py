"""Merge and maintain LLM-generated experience lesson libraries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import normalize_lesson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge, deduplicate, filter, and sort LLM experience lessons.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="lesson JSON file; can be repeated")
    parser.add_argument("--input-glob", action="append", default=[], help="glob pattern for lesson JSON files")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--dedupe-by", choices=["semantic", "lesson_id"], default="semantic")
    parser.add_argument("--sort-by", choices=["confidence", "support", "lesson_id"], default="confidence")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"lesson input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _input_paths(inputs: list[Path], patterns: list[str]) -> list[Path]:
    paths = list(inputs)
    for pattern in patterns:
        paths.extend(Path(item) for item in sorted(Path().glob(pattern)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _raw_lessons(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    lessons = payload.get("lessons") if isinstance(payload, dict) else payload
    if not isinstance(lessons, list):
        raise ValueError("lesson input must be a JSON list or object containing lessons")
    return [item for item in lessons if isinstance(item, dict)]


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _lesson_key(lesson: dict[str, Any], *, mode: str) -> str:
    if mode == "lesson_id":
        return str(lesson.get("lesson_id") or "")
    return "|".join([
        _norm_text(lesson.get("if")),
        _norm_text(lesson.get("then")),
        _norm_text(lesson.get("avoid")),
        _norm_text(lesson.get("lesson")),
    ])


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*left, *right]:
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _merge_lesson(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    existing_confidence = float(existing.get("confidence") or 0.0)
    incoming_confidence = float(incoming.get("confidence") or 0.0)
    if incoming_confidence > existing_confidence:
        for key in ("lesson", "if", "then", "avoid", "lesson_id"):
            merged[key] = incoming.get(key, merged.get(key, ""))
        merged["confidence"] = incoming_confidence
    else:
        merged["confidence"] = existing_confidence
    merged["evidence_ids"] = _merge_unique(existing.get("evidence_ids", []), incoming.get("evidence_ids", []))
    merged["supporting_candidate_ids"] = _merge_unique(
        existing.get("supporting_candidate_ids", []),
        incoming.get("supporting_candidate_ids", []),
    )
    merged["source_paths"] = _merge_unique(existing.get("source_paths", []), incoming.get("source_paths", []))
    return merged


def _sort_key(lesson: dict[str, Any], *, sort_by: str) -> tuple[Any, ...]:
    if sort_by == "support":
        return (
            -len(lesson.get("evidence_ids", [])),
            -len(lesson.get("supporting_candidate_ids", [])),
            -float(lesson.get("confidence") or 0.0),
            str(lesson.get("lesson_id") or ""),
        )
    if sort_by == "lesson_id":
        return (str(lesson.get("lesson_id") or ""),)
    return (
        -float(lesson.get("confidence") or 0.0),
        -len(lesson.get("evidence_ids", [])),
        str(lesson.get("lesson_id") or ""),
    )


def merge_lessons(paths: list[Path], *, min_confidence: float, dedupe_by: str, sort_by: str) -> dict[str, Any]:
    merged_by_key: dict[str, dict[str, Any]] = {}
    raw_count = 0
    filtered_count = 0
    for path in paths:
        payload = _load_json(path)
        for raw in _raw_lessons(payload):
            raw_count += 1
            lesson = normalize_lesson(raw, source_path=str(path))
            if float(lesson.get("confidence") or 0.0) < min_confidence:
                filtered_count += 1
                continue
            lesson["source_paths"] = [str(path)]
            key = _lesson_key(lesson, mode=dedupe_by)
            if not key:
                key = _lesson_key(lesson, mode="semantic")
            if key in merged_by_key:
                merged_by_key[key] = _merge_lesson(merged_by_key[key], lesson)
            else:
                merged_by_key[key] = lesson
    lessons = sorted(merged_by_key.values(), key=lambda item: _sort_key(item, sort_by=sort_by))
    return {
        "schema_version": "experience_lessons_library_v1",
        "generator": "merge_experience_lessons.py",
        "input_files": [str(path) for path in paths],
        "merge_policy": {
            "dedupe_by": dedupe_by,
            "sort_by": sort_by,
            "min_confidence": min_confidence,
            "confidence": "keep highest-confidence text for duplicate lessons",
            "evidence_ids": "union",
            "supporting_candidate_ids": "union",
        },
        "raw_lesson_count": raw_count,
        "filtered_lesson_count": filtered_count,
        "duplicate_count": max(raw_count - filtered_count - len(lessons), 0),
        "lesson_count": len(lessons),
        "lessons": lessons,
    }


def main() -> None:
    args = parse_args()
    paths = _input_paths(args.input, args.input_glob)
    if not paths:
        raise ValueError("provide at least one --input or --input-glob")
    report = merge_lessons(
        paths,
        min_confidence=max(0.0, min(float(args.min_confidence), 1.0)),
        dedupe_by=args.dedupe_by,
        sort_by=args.sort_by,
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input_count": len(paths),
        "raw_lesson_count": report["raw_lesson_count"],
        "filtered_lesson_count": report["filtered_lesson_count"],
        "duplicate_count": report["duplicate_count"],
        "lesson_count": report["lesson_count"],
        "save": str(args.save),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
