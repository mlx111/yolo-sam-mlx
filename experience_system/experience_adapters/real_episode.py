"""Generic real/pseudo-real episode adapter for universal experience memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_core import (
    ExperienceEntry,
    MemoryGate,
    ObjectState,
    RobotState,
    SensorEvidence,
    SensorSummary,
    SkillTraceItem,
    compute_memory_gate,
    attach_sensor_sim_real_gap,
    enrich_memory_gate_with_sensor_quality,
    standardize_failure_taxonomy,
)


class RealEpisodeAdapter:
    """Normalize file-based real robot episodes into universal ExperienceEntry."""

    def __init__(self, *, default_backend: str = "real_robot") -> None:
        self.default_backend = default_backend

    def collect_episode_dir(self, episode_dir: str | Path) -> dict[str, Any]:
        root = Path(episode_dir)
        if not root.is_dir():
            raise ValueError(f"episode_dir is not a directory: {root}")

        episode = self._read_first_json(root, ["episode.json", "real_episode.json", "result.json"])
        if not episode:
            raise ValueError(f"no episode.json, real_episode.json, or result.json found in {root}")

        experience_after = self._read_first_json(root, ["experience_after.json", "experience.json"])
        if experience_after:
            episode = self._merge_dicts(episode, experience_after)

        sensor_summary = self._read_first_json(root, ["sensor_summary.json", "sensors.json"])
        if sensor_summary:
            episode.setdefault("sensor_summary", sensor_summary)

        keyframe_dir = self._first_existing_dir(root, ["keyframes", "frames"])
        if keyframe_dir is not None and not episode.get("keyframes"):
            episode["keyframes"] = self._collect_keyframes(keyframe_dir)

        raw_refs = dict(episode.get("raw_refs") or {})
        for name in ("episode.hdf5", "episode.h5", "robot_log.jsonl", "robot_log.json"):
            path = root / name
            if path.exists():
                raw_refs[name.replace(".", "_")] = str(path)
        for dirname in ("video", "videos", "rgb", "camera", "keyframes", "frames"):
            path = root / dirname
            if path.exists():
                raw_refs[f"{dirname}_dir"] = str(path)
        raw_refs["episode_dir"] = str(root)
        episode["raw_refs"] = raw_refs
        episode.setdefault("_episode_root", str(root))
        return episode

    def normalize_episode(self, raw_episode: dict[str, Any], *, source: str | None = None) -> ExperienceEntry:
        source_value = source or str(raw_episode.get("source") or "real")
        if source_value not in {"real", "pseudo_real"}:
            source_value = "real"

        scenario = self._section(raw_episode, "scenario")
        condition = self._section(raw_episode, "condition")
        task = self._section(raw_episode, "task")
        result = self._result(raw_episode)
        sensor_summary = self._sensor_summary(raw_episode)
        sensor_evidence = self._sensor_evidence(raw_episode, raw_refs=self._raw_refs(raw_episode))
        object_state = self._object_state(raw_episode)
        validation_status = str(raw_episode.get("validation_status") or ("real_executed" if source_value == "real" else "pseudo_real_executed"))
        skill_sequence = [
            self._skill_trace_item(item)
            for item in self._list_value(raw_episode, "skill_sequence", "executed_recovery_steps", "recovery_steps")
        ]
        raw_refs = self._raw_refs(raw_episode)
        keyframes = self._normalize_keyframes(raw_episode.get("keyframes") or [], raw_refs.get("episode_dir", ""))
        metrics = {
            "condition_id": condition.get("condition_id", ""),
            "skill_trace": raw_episode.get("skill_sequence") or raw_episode.get("executed_recovery_steps") or [],
            "anomaly_detected": bool(raw_episode.get("anomaly") or raw_episode.get("anomaly_state") or condition.get("condition_id")),
        }
        metrics.update(raw_episode.get("metrics") or {})
        memory_gate = compute_memory_gate(
            metrics,
            task_success=bool(result.get("task_success", False)),
            validation_status=validation_status,
            sim_real_gap=raw_episode.get("sim_real_gap") if isinstance(raw_episode.get("sim_real_gap"), dict) else None,
        )

        entry = ExperienceEntry(
            experience_id=str(raw_episode.get("experience_id") or raw_episode.get("episode_id") or ""),
            source=source_value,
            domain=str(raw_episode.get("domain") or "real_robot_episode"),
            backend=str(raw_episode.get("backend") or self.default_backend),
            validation_status=validation_status,
            robot=self._robot_state(raw_episode),
            embodiment=dict(raw_episode.get("embodiment") or {}),
            scenario=scenario,
            condition=condition,
            task=task,
            anomaly=dict(raw_episode.get("anomaly") or raw_episode.get("anomaly_state") or {}),
            skill_sequence=skill_sequence,
            action_trace=list(raw_episode.get("action_trace") or []),
            observation_trace=list(raw_episode.get("observation_trace") or []),
            state_before=dict(raw_episode.get("state_before") or raw_episode.get("before_state") or {}),
            state_after=dict(raw_episode.get("state_after") or raw_episode.get("after_state") or {}),
            sensor_summary=sensor_summary,
            sensor_evidence=sensor_evidence,
            spatial_state=self._spatial_state(raw_episode),
            object_state=object_state,
            result=result,
            execution_feedback=dict(raw_episode.get("execution_feedback") or {}),
            key_slices=list(raw_episode.get("key_slices") or []),
            keyframes=keyframes,
            memory_gate=MemoryGate(**memory_gate),
            failure_taxonomy=dict(raw_episode.get("failure_taxonomy") or ({"failure_type": result.get("failure_reason", "")} if result.get("failure_reason") else {})),
            memory_tags=dict(raw_episode.get("memory_tags") or {
                "memory_type": "episodic",
                "memory_scope": "condition",
                "memory_role": "real_success_prior" if result.get("success") else "real_failure_case",
            }),
            real_episode_ref=self._real_episode_ref(raw_episode),
            raw_refs=raw_refs,
            metadata={
                "raw_episode_id": raw_episode.get("episode_id", ""),
                "operator": raw_episode.get("operator", ""),
                "device": raw_episode.get("device", ""),
                "source_format": "generic_real_episode",
            },
        )
        entry = enrich_memory_gate_with_sensor_quality(entry, check_refs=False)
        entry = attach_sensor_sim_real_gap(entry, overwrite=False)
        return standardize_failure_taxonomy(entry)

    def load_episode(self, source: str | Path) -> dict[str, Any]:
        path = Path(source)
        if path.is_dir():
            return self.collect_episode_dir(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def collect_batch_sources(self, source_root: str | Path) -> list[dict[str, Any]]:
        root = Path(source_root)
        if root.is_file():
            return [self.load_episode(root)]

        episodes: list[dict[str, Any]] = []
        for child in sorted(root.iterdir()):
            if child.is_dir():
                if any((child / name).exists() for name in ("episode.json", "real_episode.json", "result.json")):
                    episodes.append(self.collect_episode_dir(child))
                continue
            if child.suffix.lower() != ".json":
                continue
            if child.name in {"manifest.json", "report.json", "summary.json"}:
                continue
            if child.stem.endswith(("_report", "_summary")):
                continue
            episodes.append(self.load_episode(child))
        return episodes

    @staticmethod
    def _read_first_json(root: Path, names: list[str]) -> dict[str, Any]:
        for name in names:
            path = root / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _first_existing_dir(root: Path, names: list[str]) -> Path | None:
        for name in names:
            path = root / name
            if path.is_dir():
                return path
        return None

    @staticmethod
    def _collect_keyframes(path: Path) -> list[dict[str, Any]]:
        frames = []
        for image in sorted(path.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            frames.append({
                "stage": image.stem,
                "image_path": str(image),
                "description": image.stem,
                "used_for_retrieval": True,
            })
        return frames

    @staticmethod
    def _merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in extra.items():
            if key not in merged:
                merged[key] = value
                continue
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = RealEpisodeAdapter._merge_dicts(merged[key], value)
                continue
            if merged[key] in ({}, [], "", None):
                merged[key] = value
        return merged

    @staticmethod
    def _list_value(raw: dict[str, Any], *keys: str) -> list[Any]:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
        value = raw.get(key)
        if isinstance(value, dict):
            return dict(value)
        if key == "scenario":
            return {
                "scenario_id": str(raw.get("scenario_id") or raw.get("scene_id") or ""),
                "name": str(raw.get("scenario_name") or raw.get("scene_name") or ""),
            }
        if key == "condition":
            return {
                "condition_id": str(raw.get("condition_id") or raw.get("anomaly_id") or ""),
                "name": str(raw.get("condition_name") or raw.get("anomaly_name") or ""),
            }
        if key == "task":
            return {
                "name": str(raw.get("task_name") or raw.get("task_id") or "real_robot_task"),
                "stage": str(raw.get("task_stage") or raw.get("stage") or ""),
                "object_class": str(raw.get("object_class") or raw.get("target_class") or ""),
            }
        return {}

    @staticmethod
    def _result(raw: dict[str, Any]) -> dict[str, Any]:
        value = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        task_success = bool(value.get("task_success", raw.get("task_success", value.get("success", raw.get("success", False)))))
        success = bool(value.get("success", raw.get("success", task_success)))
        return {
            "success": success,
            "task_success": task_success,
            "failure_reason": str(value.get("failure_reason") or raw.get("failure_reason") or ""),
            "attempt_count": value.get("attempt_count", raw.get("attempt_count", raw.get("attempts", 1))),
            **{k: v for k, v in value.items() if k not in {"success", "task_success", "failure_reason"}},
        }

    @staticmethod
    def _robot_state(raw: dict[str, Any]) -> RobotState:
        robot = raw.get("robot") if isinstance(raw.get("robot"), dict) else {}
        robot_type = str(robot.get("robot_type") or raw.get("robot_type") or "unknown_robot")
        backend = str(raw.get("backend") or "real_robot")
        return RobotState(
            robot_id=str(robot.get("robot_id") or raw.get("robot_id") or "real_robot_unknown"),
            robot_type=robot_type,
            embodiment_tags=list(robot.get("embodiment_tags") or raw.get("embodiment_tags") or []),
            backend=backend,
            kinematic_groups=dict(robot.get("kinematic_groups") or {}),
            end_effectors=dict(robot.get("end_effectors") or {}),
            mobile_base=dict(robot.get("mobile_base") or {}),
            torso=dict(robot.get("torso") or {}),
            grippers=dict(robot.get("grippers") or {}),
            joints=dict(robot.get("joints") or {}),
            metadata=dict(robot.get("metadata") or {}),
        )

    @staticmethod
    def _sensor_summary(raw: dict[str, Any]) -> SensorSummary:
        value = raw.get("sensor_summary") if isinstance(raw.get("sensor_summary"), dict) else {}
        if "ee_pose" in value and "end_effector_pose" not in value:
            value = {**value, "end_effector_pose": value.get("ee_pose")}
        if "contacts" in value and "contact_state" not in value:
            value = {**value, "contact_state": value.get("contacts")}
        modalities = list(value.get("sensor_modalities") or [])
        for field_name, modality in (
            ("visual_observation", "rgbd"),
            ("rgbd_observation", "rgbd"),
            ("lidar_observation", "lidar"),
            ("wrist_force_observation", "wrist_force"),
            ("force_torque_observation", "wrist_force"),
        ):
            if isinstance(raw.get(field_name), dict) and modality not in modalities:
                modalities.append(modality)
        if modalities:
            value = {**value, "sensor_modalities": sorted(set(str(item) for item in modalities if item))}
        if "wrist_force_observation" in raw and "force_torque" not in value:
            value = {**value, "force_torque": raw.get("wrist_force_observation")}
        return SensorSummary(**{key: item for key, item in value.items() if key in SensorSummary.__dataclass_fields__})

    @staticmethod
    def _sensor_evidence(raw: dict[str, Any], *, raw_refs: dict[str, Any]) -> SensorEvidence:
        visual = raw.get("visual_observation") or raw.get("rgbd_observation") or {}
        lidar = raw.get("lidar_observation") or {}
        wrist_force = raw.get("wrist_force_observation") or raw.get("force_torque_observation") or {}
        evidence_refs = dict(raw.get("sensor_evidence_refs") or {})
        for key, value in raw_refs.items():
            if any(token in key for token in ("rgb", "depth", "camera", "lidar", "force", "wrist")):
                evidence_refs.setdefault(key, value)
        modalities = []
        if isinstance(visual, dict) and visual:
            modalities.append("rgbd" if visual.get("depth_image_path") or visual.get("depth") else "rgb")
        if isinstance(lidar, dict) and lidar:
            modalities.append("lidar")
        if isinstance(wrist_force, dict) and wrist_force:
            modalities.append("wrist_force")
        summary = {
            "has_visual": bool(visual),
            "has_lidar": bool(lidar),
            "has_wrist_force": bool(wrist_force),
            "visual_camera": str(visual.get("camera_name") or "") if isinstance(visual, dict) else "",
            "lidar_ray_count": lidar.get("ray_count") if isinstance(lidar, dict) else None,
            "max_wrist_force_norm": _max_wrist_force_norm(wrist_force) if isinstance(wrist_force, dict) else 0.0,
        }
        return SensorEvidence(
            visual_observation=dict(visual) if isinstance(visual, dict) else {},
            lidar_observation=dict(lidar) if isinstance(lidar, dict) else {},
            wrist_force_observation=dict(wrist_force) if isinstance(wrist_force, dict) else {},
            evidence_refs=evidence_refs,
            modalities=sorted(set(modalities)),
            summary=summary,
        )

    @staticmethod
    def _object_state(raw: dict[str, Any]) -> ObjectState:
        value = raw.get("object_state") if isinstance(raw.get("object_state"), dict) else {}
        if value:
            return ObjectState(**{key: item for key, item in value.items() if key in ObjectState.__dataclass_fields__})
        target = str(raw.get("target_object") or raw.get("object_class") or raw.get("target_name") or "")
        observed_pos = (
            raw.get("observed_pos")
            or raw.get("object_pose")
            or ((raw.get("execution_feedback") or {}).get("observed_pos") if isinstance(raw.get("execution_feedback"), dict) else None)
        )
        objects = {target: {"observed_position": observed_pos}} if target else {}
        return ObjectState(
            objects=objects,
            target_object=target,
            object_class=str(raw.get("object_class") or raw.get("target_class") or target),
        )

    @staticmethod
    def _skill_trace_item(raw: dict[str, Any]) -> SkillTraceItem:
        name = str(raw.get("name") or raw.get("skill") or raw.get("action") or raw.get("type") or "")
        params = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else raw.get("params", {})
        return SkillTraceItem(
            name=name,
            primitive_type=str(raw.get("primitive_type") or ""),
            phase=str(raw.get("phase") or ""),
            inputs=params if isinstance(params, dict) else {},
            outputs=dict(raw.get("outputs") or {}),
            success=bool(raw.get("success", True)),
            error=raw.get("error"),
            duration=raw.get("duration"),
            message=str(raw.get("message") or ""),
            raw=dict(raw),
        )

    @staticmethod
    def _raw_refs(raw: dict[str, Any]) -> dict[str, Any]:
        refs = dict(raw.get("raw_refs") or {})
        real_episode_ref = raw.get("real_episode_ref") if isinstance(raw.get("real_episode_ref"), dict) else {}
        if real_episode_ref:
            refs.setdefault("real_episode_ref", real_episode_ref)
            for key in ("raw_episode_id", "hdf5_path", "video_dir", "keyframe_dir", "robot_log_path"):
                if key in real_episode_ref and key not in refs:
                    refs[key] = real_episode_ref[key]
        if raw.get("_episode_root"):
            refs.setdefault("episode_dir", raw["_episode_root"])
        return refs

    @staticmethod
    def _real_episode_ref(raw: dict[str, Any]) -> dict[str, Any]:
        value = raw.get("real_episode_ref") if isinstance(raw.get("real_episode_ref"), dict) else {}
        ref = dict(value)
        if raw.get("episode_id") and not ref.get("raw_episode_id"):
            ref["raw_episode_id"] = raw.get("episode_id")
        if raw.get("_episode_root") and not ref.get("episode_dir"):
            ref["episode_dir"] = raw.get("_episode_root")
        for key in ("hdf5_path", "video_dir", "keyframe_dir", "robot_log_path", "operator", "device"):
            if key not in ref and raw.get(key) is not None:
                ref[key] = raw.get(key)
        return ref

    @staticmethod
    def _normalize_keyframes(raw_keyframes: list[dict[str, Any]], episode_root: str) -> list[dict[str, Any]]:
        root = Path(episode_root) if episode_root else None
        normalized = []
        for item in raw_keyframes:
            if not isinstance(item, dict):
                continue
            frame = dict(item)
            image_path = frame.get("image_path")
            if image_path and root is not None:
                path = Path(str(image_path))
                if not path.is_absolute():
                    frame["image_path"] = str((root / path).resolve())
            normalized.append(frame)
        return normalized

    @staticmethod
    def _spatial_state(raw: dict[str, Any]) -> dict[str, Any]:
        value = raw.get("spatial_state") if isinstance(raw.get("spatial_state"), dict) else {}
        if value:
            return dict(value)
        state = {}
        for key in ("scene_name", "workspace_id", "table_zone", "place_zone", "selected_place_site"):
            if key in raw:
                state[key] = raw[key]
        return state


def _max_wrist_force_norm(value: dict[str, Any]) -> float:
    candidates: list[float] = []
    for key in ("force_norm", "max_force_norm", "peak_force_norm"):
        if value.get(key) is not None:
            candidates.append(float(value[key]))
    for side in ("left", "right"):
        side_value = value.get(side)
        if isinstance(side_value, dict):
            for key in ("force_norm", "max_force_norm", "peak_force_norm"):
                if side_value.get(key) is not None:
                    candidates.append(float(side_value[key]))
    samples = value.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if isinstance(sample, dict) and sample.get("force_norm") is not None:
                candidates.append(float(sample["force_norm"]))
    return max(candidates) if candidates else 0.0
