"""Build a runtime MuJoCo sandbox scene from structured scene observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experience_core import RuntimeSandboxScene, write_runtime_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate runtime MuJoCo XML from scene observation JSON.")
    parser.add_argument("--scene", type=Path, required=True, help="runtime_sandbox_scene_v1 JSON")
    parser.add_argument("--output", type=Path, required=True, help="output MuJoCo XML")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.scene.read_text(encoding="utf-8"))
    scene = RuntimeSandboxScene.from_dict(payload)
    report = write_runtime_scene(scene, args.output)
    report.update({
        "input": str(args.scene),
        "schema_version": scene.schema_version,
        "metadata": scene.metadata,
    })
    if args.report is not None:
        _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
