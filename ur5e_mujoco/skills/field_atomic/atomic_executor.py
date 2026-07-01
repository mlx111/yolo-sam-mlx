from __future__ import annotations

from typing import Any, Callable

from skills.context import Ur5eSkillContext

from .atomic_registry import field_atomic_skill_registry
from .atomic_schema import Ur5eFieldAtomicResult


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


class Ur5eFieldAtomicSkillExecutor:
    """Field-atomic dispatcher for the UR5e MuJoCo wrapper."""

    def __init__(self, runtime: Ur5eSkillContext | Any, *, default_pregrasp_height: float = 0.127) -> None:
        self.context = runtime if isinstance(runtime, Ur5eSkillContext) else Ur5eSkillContext(runtime, default_pregrasp_height=default_pregrasp_height)
        self._registry = field_atomic_skill_registry()
        self._loaders: dict[str, Callable[[], Any]] = self._build_loaders()

    def can_execute(self, action: str) -> bool:
        return str(action) in self._registry

    def execute(self, action: str, parameters: dict[str, Any] | None = None) -> Ur5eFieldAtomicResult:
        action = str(action)
        params = parameters if isinstance(parameters, dict) else {}
        if action not in self._registry:
            return Ur5eFieldAtomicResult(action=action, success=False, status="unsupported_action", message=f"unsupported UR5e action: {action}", parameters=dict(params))
        skill_results_before = self._skill_result_count()
        try:
            raw = self._loaders[action]().execute_recovery_action(self.context, params)
        except Exception as exc:
            return Ur5eFieldAtomicResult(action=action, success=False, status="failed", message=str(exc), parameters=dict(params), raw_result={})
        raw_result = _jsonable(raw)
        records = self._new_skill_results(skill_results_before)
        failed_record = self._last_failed_record(records)
        if failed_record is not None:
            return Ur5eFieldAtomicResult(
                action=action,
                success=False,
                status=str(failed_record.get("reason") or "failed"),
                message=str(failed_record.get("reason") or f"{action} failed"),
                parameters=dict(params),
                raw_result={
                    "result": raw_result if not isinstance(raw_result, dict) else raw_result,
                    "skill_record": failed_record,
                },
            )
        ok_record = records[-1] if records else None
        return Ur5eFieldAtomicResult(
            action=action,
            success=True,
            status=str((ok_record or {}).get("reason") or "ok"),
            message=str((ok_record or {}).get("reason") or "executed"),
            parameters=dict(params),
            raw_result={
                **(raw_result if isinstance(raw_result, dict) else {"result": raw_result}),
                **({"skill_record": ok_record} if ok_record else {}),
            },
        )

    def execute_plan(self, steps: list[dict[str, Any]]) -> list[Ur5eFieldAtomicResult]:
        results: list[Ur5eFieldAtomicResult] = []
        for step in steps:
            if not isinstance(step, dict):
                results.append(Ur5eFieldAtomicResult(action="", success=False, status="invalid_step", message="step is not a dict"))
                continue
            results.append(self.execute(str(step.get("action", "")), step.get("parameters") if isinstance(step.get("parameters"), dict) else {}))
        return results

    def _skill_result_count(self) -> int:
        metrics = getattr(self.context.experiment, "metrics", {}) or {}
        results = metrics.get("skill_results") if isinstance(metrics, dict) else None
        return len(results) if isinstance(results, list) else 0

    def _new_skill_results(self, before: int) -> list[dict[str, Any]]:
        metrics = getattr(self.context.experiment, "metrics", {}) or {}
        results = metrics.get("skill_results") if isinstance(metrics, dict) else None
        if not isinstance(results, list):
            return []
        return [dict(item) for item in results[before:] if isinstance(item, dict)]

    @staticmethod
    def _last_failed_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        for record in reversed(records):
            if record.get("success") is False:
                return record
        return None

    @staticmethod
    def _build_loaders() -> dict[str, Callable[[], Any]]:
        from skills.primitives.approach_object_skill import load_skill as load_approach_object
        from skills.primitives.camera_rgbd_save_skill import load_skill as load_camera_rgbd_save
        from skills.primitives.close_gripper_skill import load_skill as load_close_gripper
        from skills.primitives.create_fixed_vertical_grasp_skill import load_skill as load_create_fixed_vertical_grasp
        from skills.primitives.detect_object_pose_skill import load_skill as load_detect_object_pose
        from skills.primitives.go_home_skill import load_skill as load_go_home
        from skills.primitives.lift_skill import load_skill as load_lift
        from skills.primitives.move_lifted_object_to_skill import load_skill as load_move_lifted_object_to
        from skills.primitives.move_to_pregrasp_skill import load_skill as load_move_to_pregrasp
        from skills.primitives.open_gripper_skill import load_skill as load_open_gripper

        return {
            "camera_rgbd_save": load_camera_rgbd_save,
            "detect_object_pose": load_detect_object_pose,
            "create_fixed_vertical_grasp": load_create_fixed_vertical_grasp,
            "move_to_pregrasp": load_move_to_pregrasp,
            "approach_object": load_approach_object,
            "close_gripper": load_close_gripper,
            "open_gripper": load_open_gripper,
            "lift": load_lift,
            "move_lifted_object_to": load_move_lifted_object_to,
            "go_home": load_go_home,
        }
