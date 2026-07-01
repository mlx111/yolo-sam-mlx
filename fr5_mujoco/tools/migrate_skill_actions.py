"""Migrate historical UR5e action names to the field-atomic skill names.

Usage:
    python ur5e_mujoco/tools/migrate_skill_actions.py --dry-run ur5e_mujoco/configs
    python ur5e_mujoco/tools/migrate_skill_actions.py ur5e_mujoco/configs ur5e_mujoco/acknowledge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTION_MAP = {
    "camera-image": "camera_rgbd_save",
    "detect-object": "detect_object_pose",
    "create-grasp": "create_fixed_vertical_grasp",
    "move-pregrasp": "move_to_pregrasp",
    "move-grasp": "approach_object",
    "vertical-grasp": "lift",
    "execute-grasp2": "move_lifted_object_to",
    "execute-init": "go_home",
}

TEXT_REPLACEMENTS = {
    **ACTION_MAP,
    '"action": "gripper-action", "state": 1': '"action": "close_gripper"',
    '"action": "gripper-action", "state": 0': '"action": "open_gripper"',
    '\\"action\\": \\"gripper-action\\", \\"state\\": 1': '\\"action\\": \\"close_gripper\\"',
    '\\"action\\": \\"gripper-action\\", \\"state\\": 0': '\\"action\\": \\"open_gripper\\"',
    "gripper-action state=1": "close_gripper",
    "gripper-action state=0": "open_gripper",
    "gripper-action": "close_gripper",
}


def _convert_action(action: str, params: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    if action == "gripper-action":
        state = params.get("state")
        if int(state) == 1:
            return "close_gripper", {}, True
        return "open_gripper", {}, True
    if action in ACTION_MAP:
        return ACTION_MAP[action], params, True
    return action, params, False


def _walk(value: Any) -> tuple[Any, int]:
    count = 0
    if isinstance(value, str):
        converted = value
        for old, new in TEXT_REPLACEMENTS.items():
            converted = converted.replace(old, new)
        return converted, int(converted != value)
    if isinstance(value, list):
        converted = []
        for item in value:
            new_item, item_count = _walk(item)
            count += item_count
            converted.append(new_item)
        return converted, count
    if isinstance(value, dict):
        payload = {}
        action = value.get("action")
        params = value.get("parameters") if isinstance(value.get("parameters"), dict) else {}
        changed_action = False
        if isinstance(action, str):
            new_action, new_params, changed_action = _convert_action(action, dict(params))
            payload["action"] = new_action
            if "parameters" in value or changed_action:
                payload["parameters"] = new_params
            count += int(changed_action)
        for key, item in value.items():
            if key == "action" or (key == "parameters" and changed_action):
                continue
            new_item, item_count = _walk(item)
            payload[key] = new_item
            count += item_count
        return payload, count
    return value, count


def _json_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0
    for path in _json_files(args.paths):
        try:
            original = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        converted, count = _walk(original)
        if count == 0:
            continue
        total += count
        print(f"{path}: {count} action(s)")
        if not args.dry_run:
            path.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n")
    print(f"total converted actions: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
