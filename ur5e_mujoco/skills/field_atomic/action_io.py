from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_action_steps(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        steps = payload["steps"]
    else:
        raise ValueError('action JSON must be a JSON object with a "steps" list')
    return [dict(item) for item in steps if isinstance(item, dict)]


def result_to_dict(result: Any, *, index: int | None = None) -> dict[str, Any]:
    raw = result.to_dict() if hasattr(result, "to_dict") else {
        "action": getattr(result, "action", ""),
        "success": getattr(result, "success", False),
        "status": getattr(result, "status", ""),
        "message": getattr(result, "message", ""),
        "parameters": getattr(result, "parameters", {}),
        "raw_result": getattr(result, "raw_result", {}),
    }
    record = dict(raw)
    if index is not None:
        record["index"] = int(index)
    return record
