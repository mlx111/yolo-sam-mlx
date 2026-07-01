"""Generate pseudo-real R1Pro episodes for universal memory calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-real R1Pro episode.json files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/pseudo_real/r1pro_calibration_v1")
    return parser.parse_args()


def _skill(name: str, success: bool = True, *, message: str = "", error: float | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "success": success}
    if message:
        item["message"] = message
    if error is not None:
        item["error"] = error
    return item


def _g3_skills(failure_type: str = "") -> list[dict[str, Any]]:
    skills = [
        _skill("detect_multiple_objects"),
        _skill("select_correct_object"),
        _skill("move_to_pregrasp"),
        _skill("approach_object", failure_type != "grasp_miss", message="tcp stopped before stable grasp" if failure_type == "grasp_miss" else ""),
        _skill("left_gripper_close", failure_type != "grasp_miss", message="no contact detected" if failure_type == "grasp_miss" else ""),
        _skill("verify_grasp", failure_type not in {"grasp_miss", "grasp_slip", "object_not_lifted"}, message=failure_type if failure_type else ""),
        _skill("left_vertical_lift", failure_type not in {"grasp_slip", "object_not_lifted"}, message="object slipped during lift" if failure_type == "grasp_slip" else ""),
        _skill("detect_place_occupancy"),
        _skill("choose_alternate_place"),
        _skill("place_object", failure_type == ""),
        _skill("open_gripper_release", failure_type == ""),
        _skill("verify_place_zone", failure_type == ""),
    ]
    return skills


def _g4_skills(failure_type: str = "") -> list[dict[str, Any]]:
    return [
        _skill("base_move_to_region"),
        _skill("torso_set_height"),
        _skill("dual_arm_pregrasp"),
        _skill("dual_arm_approach", failure_type != "dual_arm_mismatch", message="dual arm target mismatch" if failure_type == "dual_arm_mismatch" else ""),
        _skill("dual_gripper_close", failure_type != "grasp_miss", message="right gripper contact missing" if failure_type == "grasp_miss" else ""),
        _skill("dual_arm_synchronized_lift", failure_type not in {"grasp_miss", "dual_arm_mismatch"}, message=failure_type if failure_type else ""),
        _skill("dual_arm_level_object", failure_type != "dual_arm_mismatch", message="height mismatch exceeded threshold" if failure_type == "dual_arm_mismatch" else ""),
        _skill("segmented_transport", failure_type != "transport_collision", message="transport collision risk" if failure_type == "transport_collision" else ""),
        _skill("detect_place_occupancy"),
        _skill("choose_alternate_place"),
        _skill("base_move_to_place"),
        _skill("dual_arm_place", failure_type not in {"place_occupied", "transport_collision", "dual_arm_mismatch"}, message="place zone occupied" if failure_type == "place_occupied" else ""),
        _skill("dual_gripper_release", failure_type == ""),
        _skill("verify_place_zone", failure_type == ""),
    ]


def _episode(
    *,
    episode_id: str,
    scenario_id: str,
    condition_id: str,
    object_class: str,
    target_object: str,
    success: bool,
    failure_type: str = "",
    selected_place_site: str = "place_zone_site",
    object_start: list[float] | None = None,
    object_final: list[float] | None = None,
) -> dict[str, Any]:
    object_start = object_start or [0.12, 0.0, 0.78]
    object_final = object_final or ([0.28, 0.25, 0.88] if success else [0.15, 0.02, 0.79])
    skills = _g3_skills(failure_type) if scenario_id == "G3" else _g4_skills(failure_type)
    occupancy = condition_id == "place_occupied"
    return {
        "episode_id": episode_id,
        "source": "pseudo_real",
        "backend": "pseudo_real_robot",
        "validation_status": "pseudo_real_executed",
        "domain": "r1pro_pseudo_real_calibration",
        "robot": {
            "robot_id": "r1pro_pseudo_real_001",
            "robot_type": "mobile_dual_arm",
            "embodiment_tags": ["mobile_base", "torso", "dual_arm", "gripper"],
        },
        "scenario": {
            "scenario_id": scenario_id,
            "name": f"R1Pro {scenario_id} pseudo-real calibration",
        },
        "condition": {
            "condition_id": condition_id,
            "name": condition_id,
        },
        "task": {
            "name": f"r1pro_{scenario_id.lower()}_task_chain",
            "stage": "task_chain",
            "object_class": object_class,
        },
        "object_state": {
            "target_object": target_object,
            "object_class": object_class,
            "objects": {
                target_object: {
                    "observed_position": object_start,
                    "final_position": object_final,
                }
            },
            "occupancy": {
                "place_occupied": occupancy,
                "selected_place_site": selected_place_site,
            },
        },
        "skill_sequence": skills,
        "result": {
            "success": success,
            "task_success": success,
            "failure_reason": failure_type,
            "attempt_count": 1,
        },
        "failure_taxonomy": {"failure_type": failure_type} if failure_type else {},
        "execution_feedback": {
            "selected_place_site": selected_place_site,
            "object_lift": round(float(object_final[2] - object_start[2]), 4),
            "metrics": {
                "place_occupied": occupancy,
                "attach_mode": "pseudo_real_contact",
                "failure_type": failure_type,
            },
        },
        "sensor_summary": {
            "sensor_modalities": ["joint_state", "rgb", "depth", "contact"],
            "gripper_state": {"left": "open", "right": "open"},
            "contact_state": {"left": success, "right": success if scenario_id == "G4" else False},
        },
        "real_episode_ref": {
            "raw_episode_id": episode_id,
            "robot_log_path": "robot_log.jsonl",
            "keyframe_dir": "keyframes",
        },
        "memory_tags": {
            "memory_type": "episodic",
            "memory_scope": "condition",
            "memory_role": "pseudo_real_success_prior" if success else "pseudo_real_failure_case",
        },
    }


def build_episodes() -> list[dict[str, Any]]:
    return [
        _episode(
            episode_id="pseudo_real_g3_clean_success_001",
            scenario_id="G3",
            condition_id="clean",
            object_class="sortable_object",
            target_object="target_cube",
            success=True,
            selected_place_site="place_zone_site",
        ),
        _episode(
            episode_id="pseudo_real_g3_place_occupied_success_001",
            scenario_id="G3",
            condition_id="place_occupied",
            object_class="sortable_object",
            target_object="target_cube",
            success=True,
            selected_place_site="alternate_place_zone_site",
        ),
        _episode(
            episode_id="pseudo_real_g3_grasp_miss_001",
            scenario_id="G3",
            condition_id="grasp_miss",
            object_class="sortable_object",
            target_object="target_cube",
            success=False,
            failure_type="grasp_miss",
            object_final=[0.12, 0.0, 0.78],
        ),
        _episode(
            episode_id="pseudo_real_g3_grasp_slip_001",
            scenario_id="G3",
            condition_id="grasp_slip",
            object_class="sortable_object",
            target_object="target_cube",
            success=False,
            failure_type="grasp_slip",
            object_final=[0.14, 0.02, 0.80],
        ),
        _episode(
            episode_id="pseudo_real_g4_clean_success_001",
            scenario_id="G4",
            condition_id="clean",
            object_class="large_object",
            target_object="large_object_body",
            success=True,
            selected_place_site="place_zone_site",
        ),
        _episode(
            episode_id="pseudo_real_g4_place_occupied_failure_001",
            scenario_id="G4",
            condition_id="place_occupied",
            object_class="large_object",
            target_object="large_object_body",
            success=False,
            failure_type="place_occupied",
            selected_place_site="place_zone_site",
            object_final=[0.24, 0.18, 0.86],
        ),
        _episode(
            episode_id="pseudo_real_g4_transport_collision_001",
            scenario_id="G4",
            condition_id="transport_collision",
            object_class="large_object",
            target_object="large_object_body",
            success=False,
            failure_type="transport_collision",
            object_final=[0.18, 0.08, 0.84],
        ),
        _episode(
            episode_id="pseudo_real_g4_dual_arm_mismatch_001",
            scenario_id="G4",
            condition_id="dual_arm_mismatch",
            object_class="large_object",
            target_object="large_object_body",
            success=False,
            failure_type="dual_arm_mismatch",
            object_final=[0.12, 0.0, 0.80],
        ),
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = build_episodes()
    for episode in episodes:
        episode_dir = output_dir / str(episode["episode_id"])
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "episode.json").write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "episode_count": len(episodes),
        "episodes": [
            {
                "episode_id": episode["episode_id"],
                "scenario_id": episode["scenario"]["scenario_id"],
                "condition_id": episode["condition"]["condition_id"],
                "success": episode["result"]["success"],
                "failure_type": episode["result"]["failure_reason"],
                "path": str(output_dir / str(episode["episode_id"]) / "episode.json"),
            }
            for episode in episodes
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "episode_count": len(episodes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
