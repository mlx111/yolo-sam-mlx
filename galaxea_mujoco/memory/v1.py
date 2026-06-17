"""Minimal legacy R1Pro task-chain memory schema.

The canonical experience system lives in ``experience_system/experience_core``.
This module only keeps older ``run_r1pro_task_chain.py --experience-lib`` flows
working without depending on the removed ``experience_system/memory`` package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskChainResult:
    scenario_id: str
    condition_id: str
    control_mode: str
    model_path: str
    success: bool
    task_success: bool
    selected_place_site: str
    target_object: str
    object_start: list[float]
    object_final: list[float]
    skill_trace: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    keyframes: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    memory_id: str
    created_at: str
    scenario_id: str
    condition_id: str
    control_mode: str
    success: bool
    task_success: bool
    selected_place_site: str
    target_object: str
    object_start: list[float]
    object_final: list[float]
    skill_trace: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    keyframes: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEntry":
        return cls(
            memory_id=str(payload.get("memory_id") or uuid4()),
            created_at=str(payload.get("created_at") or _utc_now()),
            scenario_id=str(payload.get("scenario_id") or ""),
            condition_id=str(payload.get("condition_id") or ""),
            control_mode=str(payload.get("control_mode") or ""),
            success=bool(payload.get("success")),
            task_success=bool(payload.get("task_success")),
            selected_place_site=str(payload.get("selected_place_site") or ""),
            target_object=str(payload.get("target_object") or ""),
            object_start=list(payload.get("object_start") or []),
            object_final=list(payload.get("object_final") or []),
            skill_trace=list(payload.get("skill_trace") or []),
            metrics=dict(payload.get("metrics") or {}),
            keyframes=list(payload.get("keyframes") or []),
            failure_reason=str(payload.get("failure_reason") or ""),
        )


class MemoryLibrary:
    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self.entries = entries or []

    @classmethod
    def load(cls, path: str | Path) -> "MemoryLibrary":
        path = Path(path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = payload.get("entries", payload if isinstance(payload, list) else [])
        return cls([MemoryEntry.from_dict(item) for item in raw_entries])

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "r1pro_legacy_memory_v1",
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def make_memory_entry(result: TaskChainResult | dict[str, Any]) -> MemoryEntry:
    payload = asdict(result) if is_dataclass(result) else dict(result)
    return MemoryEntry(
        memory_id=f"{payload.get('scenario_id', 'r1pro')}_{payload.get('condition_id', 'unknown')}_{uuid4().hex[:8]}",
        created_at=_utc_now(),
        scenario_id=str(payload.get("scenario_id") or ""),
        condition_id=str(payload.get("condition_id") or ""),
        control_mode=str(payload.get("control_mode") or ""),
        success=bool(payload.get("success")),
        task_success=bool(payload.get("task_success")),
        selected_place_site=str(payload.get("selected_place_site") or ""),
        target_object=str(payload.get("target_object") or ""),
        object_start=list(payload.get("object_start") or []),
        object_final=list(payload.get("object_final") or []),
        skill_trace=list(payload.get("skill_trace") or []),
        metrics=dict(payload.get("metrics") or {}),
        keyframes=list(payload.get("keyframes") or []),
        failure_reason=str(payload.get("failure_reason") or ""),
    )
