"""Storage and simple structured retrieval for universal experience entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .retrieval import RetrievalMatch, RetrievalQuery, matches_to_tuples, retrieve_experiences
from .schema import ExperienceEntry, SkillCatalog, coerce_skill_catalogs, utc_now
from .write_policy import apply_write_decision, should_write_entry


class ExperienceLibrary:
    def __init__(self, entries: list[ExperienceEntry] | None = None, skill_catalogs: dict[str, SkillCatalog] | None = None) -> None:
        self.entries = entries or []
        self.skill_catalogs = coerce_skill_catalogs(skill_catalogs)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: ExperienceEntry) -> None:
        for index, existing in enumerate(self.entries):
            if existing.experience_id == entry.experience_id:
                entry.updated_at = utc_now()
                self.entries[index] = entry
                return
        self.entries.append(entry)

    def add_with_policy(
        self,
        entry: ExperienceEntry,
        *,
        strict_quality: bool = True,
        merge_duplicates: bool = True,
    ) -> dict[str, Any]:
        decision = should_write_entry(
            entry,
            self.entries,
            strict_quality=strict_quality,
            merge_duplicates=merge_duplicates,
        )
        self.entries, written_entry = apply_write_decision(self.entries, entry, decision)
        if written_entry is not None:
            decision["stored_experience_id"] = written_entry.experience_id
        return decision

    def query(
        self,
        *,
        scenario_id: str = "",
        condition_id: str = "",
        robot_type: str = "",
        backend: str = "",
        skill_namespace: str = "",
        task_stage: str = "",
        top_k: int = 5,
        include_failed: bool = True,
    ) -> list[tuple[ExperienceEntry, float]]:
        matches = self.query_structured(
            RetrievalQuery(
                scenario_id=scenario_id,
                condition_id=condition_id,
                robot_type=robot_type,
                backend=backend,
                skill_namespace=skill_namespace,
                task_stage=task_stage,
                top_k=top_k,
                include_failed=include_failed,
            )
        )
        return matches_to_tuples(matches)

    def query_structured(self, query: RetrievalQuery | dict[str, Any]) -> list[RetrievalMatch]:
        return retrieve_experiences(self.entries, query)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "universal_experience_library_v2",
            "updated_at": utc_now(),
            "entry_count": len(self.entries),
            "skill_catalogs": {key: catalog.__dict__ for key, catalog in self.skill_catalogs.items()},
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceLibrary":
        source = Path(path)
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries", []) if isinstance(payload, dict) else payload
        entries = [ExperienceEntry(**item) for item in raw_entries if isinstance(item, dict)]
        skill_catalogs = payload.get("skill_catalogs", {}) if isinstance(payload, dict) else {}
        return cls(entries, skill_catalogs=skill_catalogs)
