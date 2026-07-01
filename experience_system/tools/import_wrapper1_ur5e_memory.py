"""Import ur5e_mujoco MemoryV3+ entries into universal memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from experience_system.ur5e_core import Wrapper1UR5eAdapter
except ModuleNotFoundError:  # pragma: no cover - script-root fallback
    from ur5e_core import Wrapper1UR5eAdapter
from experience_core import ExperienceLibrary


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return [item for item in payload["entries"] if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"Unsupported wrapper1 memory JSON shape: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import wrapper1 UR5e MemoryV3+ JSON into universal experience memory.")
    parser.add_argument("--input", type=Path, required=True, help="wrapper1 MemoryV3+ JSON path")
    parser.add_argument("--universal-experience-lib", type=Path, required=True, help="output universal memory JSON")
    parser.add_argument("--limit", type=int, default=0, help="max entries to import; 0 means all")
    parser.add_argument("--robot-id", default="ur5e_wrapper1_mujoco")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-write-policy", action="store_true", help="append entries without write-time gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_entries = _load_entries(args.input)
    if args.limit > 0:
        raw_entries = raw_entries[: args.limit]

    adapter = Wrapper1UR5eAdapter(robot_id=args.robot_id)
    library = ExperienceLibrary.load(args.universal_experience_lib)

    imported = []
    for raw in raw_entries:
        entry = adapter.normalize_entry(raw)
        write_policy = {"decision": "write", "reason": "write_policy_disabled", "stored_experience_id": entry.experience_id}
        if args.no_write_policy:
            library.add(entry)
        else:
            write_policy = library.add_with_policy(entry)
        imported.append({
            "source_experience_id": raw.get("experience_id", ""),
            "universal_experience_id": entry.experience_id,
            "stored_experience_id": write_policy.get("stored_experience_id", ""),
            "scenario_id": entry.scenario_id,
            "condition_id": entry.condition_id,
            "success": entry.result.get("success", False),
            "memory_partition": entry.memory_partition,
            "write_policy": write_policy,
        })

    library.save(args.universal_experience_lib)
    report = {
        "input": str(args.input),
        "output": str(args.universal_experience_lib),
        "imported_count": len(imported),
        "entries": imported,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"imported_count": len(imported), "output": str(args.universal_experience_lib)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
