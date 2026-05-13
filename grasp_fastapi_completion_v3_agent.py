from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from fastapi import HTTPException
from langchain.agents import create_agent
from pydantic import BaseModel, Field

import grasp_fastapi_completion_v3 as base
from exception_handling_agent import (
    _extract_json_text,
    _history_images_for_recovery,
    _history_records,
    _history_summary_text,
    _json_dumps,
    _normalize_recovery_step,
    get_exception_handling_agent,
)


app = base.app


def _dump_model(instance: BaseModel) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    return instance.dict()


class AgentPlanStep(BaseModel):
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentPlanResponse(BaseModel):
    status: str
    target: str
    raw_response: str
    steps: list[AgentPlanStep] = Field(default_factory=list)
    source: str


class AgentExecutionResponse(BaseModel):
    status: str
    target: str
    executed_steps: list[dict[str, Any]] = Field(default_factory=list)
    plan: list[AgentPlanStep] = Field(default_factory=list)
    message: str = ""
    score: dict[str, Any] | None = None


class AgentRecoveryRequest(BaseModel):
    target_name: Optional[str] = None
    place_x: Optional[float] = None
    place_y: Optional[float] = None


class AgentRunTaskRequest(BaseModel):
    target_name: str = "apple"
    place_x: float = 0.5
    place_y: float = 0.5
    max_recovery_rounds: int = 2


class RecoveryPlannerAgent:
    def __init__(self) -> None:
        self.backend = get_exception_handling_agent()
        self._graph = None

    def _get_graph(self):
        if self._graph is not None:
            return self._graph
        model = self.backend._get_chat_model(
            model_name=self.backend.settings.recovery_model,
            timeout_seconds=self.backend.settings.recovery_timeout_seconds,
            max_tokens=self.backend.settings.max_recovery_tokens,
        )
        self._graph = create_agent(
            model=model,
            tools=[],
            system_prompt=(
                "你是机械臂抓取异常恢复 agent。"
                "你的职责只有一件事：根据当前失败上下文，输出最小且可执行的恢复动作 JSON 数组。"
                "不要解释，不要 Markdown，不要代码块。"
            ),
            name="grasp_recovery_planner",
        )
        return self._graph

    def _extract_text(self, result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        if not messages:
            return "[]"
        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        if isinstance(content, str):
            return content
        return _json_dumps(content)

    def _fallback_plan(self, target_name: str, place_target_xy: list[float] | None = None) -> AgentPlanResponse:
        steps = self.backend.plan_recovery(
            base.content_list_all,
            base.task_list,
            target=target_name,
            place_target_xy=place_target_xy,
        )
        return AgentPlanResponse(
            status="success" if steps else "failure",
            target=target_name,
            raw_response=_json_dumps([step.to_payload() for step in steps]),
            steps=[AgentPlanStep(**step.to_payload()) for step in steps],
            source="fallback",
        )

    def plan(
        self,
        target_name: Optional[str] = None,
        place_target_xy: list[float] | None = None,
    ) -> AgentPlanResponse:
        resolved_target = str(target_name or base.target or "apple")
        history_records = _history_records(base.content_list_all)
        resolved_place_target = place_target_xy or base.get_current_place_target_xy()
        place_hint = ""
        if resolved_place_target is not None:
            place_hint = (
                f"\n指定放置坐标: x={float(resolved_place_target[0]):.4f}, "
                f"y={float(resolved_place_target[1]):.4f}"
            )
        content_blocks = _history_images_for_recovery(history_records, self.backend.settings)
        content_blocks.append(
            {
                "type": "text",
                "text": f"""
当前目标物体: {resolved_target}
当前技能轨迹: {base.task_list}
历史动作摘要:
{_history_summary_text(history_records)}
{place_hint}

你需要针对当前失败上下文输出恢复动作序列。
可用动作只有：
camera-image, detect-object, create-cloud, create-grasp, move-pregrasp, move-grasp,
vertical-grasp, gripper-action, execute-grasp2, execute-init

输出约束：
1. 只输出 JSON 数组
2. 每个元素形如 {{"action":"camera-image","parameters":{{}}}}
3. detect-object 的参数名必须是 target_class
4. gripper-action 的参数名必须是 flag，值只能是 0 或 1
5. execute-grasp2 的参数必须包含 x 和 y，并使用给定的指定放置坐标
6. 如果最后失败动作需要重新执行，应直接把该动作写进恢复序列
7. 如果物体位置明显变化，应先 execute-init，再重新感知和生成抓取
8. 如果只是局部失败，优先局部回退，不要整套重来
9. 在抓取前必须保证夹爪打开
""",
            }
        )

        try:
            result = self._get_graph().invoke(
                {"messages": [{"role": "user", "content": content_blocks}]}
            )
            raw_text = self._extract_text(result)
            normalized_text = _extract_json_text(raw_text, prefer_array=True)
            payload = json.loads(normalized_text)
            if not isinstance(payload, list):
                return self._fallback_plan(resolved_target, resolved_place_target)

            steps: list[AgentPlanStep] = []
            for raw_step in payload:
                normalized = _normalize_recovery_step(
                    raw_step,
                    target=resolved_target,
                    place_target_xy=resolved_place_target,
                )
                if normalized is None:
                    continue
                steps.append(AgentPlanStep(**normalized.to_payload()))

            if not steps:
                return self._fallback_plan(resolved_target, resolved_place_target)

            return AgentPlanResponse(
                status="success",
                target=resolved_target,
                raw_response=raw_text,
                steps=steps,
                source="agent",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] recovery planner agent fallback due to: {exc}")
            return self._fallback_plan(resolved_target, resolved_place_target)


class RecoveryActionExecutor:
    def __init__(self) -> None:
        self.action_map = {
            "camera-image": base.get_camera_image,
            "detect-object": base.detect,
            "create-cloud": base.cloud,
            "create-grasp": base.create_grasp,
            "move-pregrasp": base.move_pregrasp,
            "move-grasp": base.move_grasp,
            "vertical-grasp": base.vertical,
            "gripper-action": base.gripper,
            "execute-grasp2": base.execute_grasp_2,
            "execute-init": base.execute_init,
        }

    async def execute_step(self, step: AgentPlanStep) -> dict[str, Any]:
        func = self.action_map.get(step.action)
        if func is None:
            return {
                "status": "failure",
                "action": step.action,
                "parameters": step.parameters,
                "message": f"未知恢复动作: {step.action}",
            }
        call_parameters = dict(step.parameters)
        if step.action == "gripper-action" and "state" in call_parameters and "flag" not in call_parameters:
            call_parameters["flag"] = call_parameters.pop("state")
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**call_parameters)
            else:
                result = func(**call_parameters)
        except HTTPException as exc:
            return {
                "status": "failure",
                "action": step.action,
                "parameters": step.parameters,
                "message": str(exc.detail),
                "error_type": "http_exception",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failure",
                "action": step.action,
                "parameters": step.parameters,
                "message": str(exc),
                "error_type": "exception",
            }

        if not isinstance(result, dict):
            return {
                "status": "failure",
                "action": step.action,
                "parameters": step.parameters,
                "message": f"动作返回值非法: {type(result).__name__}",
            }

        payload = dict(result)
        payload.setdefault("action", step.action)
        payload.setdefault("parameters", step.parameters)
        return payload

    async def execute_plan(self, steps: list[AgentPlanStep], *, target_name: str) -> AgentExecutionResponse:
        executed_steps: list[dict[str, Any]] = []
        for step in steps:
            result = await self.execute_step(step)
            executed_steps.append(result)
            if result.get("status") != "success":
                return AgentExecutionResponse(
                    status="failure",
                    target=target_name,
                    executed_steps=executed_steps,
                    plan=steps,
                    message=f"恢复动作失败: {step.action}",
                )
        return AgentExecutionResponse(
            status="success",
            target=target_name,
            executed_steps=executed_steps,
            plan=steps,
            message="恢复动作执行完成",
        )


class GraspExceptionHandlingAgentRunner:
    def __init__(self) -> None:
        self.planner = RecoveryPlannerAgent()
        self.executor = RecoveryActionExecutor()

    async def plan_recovery(
        self,
        target_name: Optional[str] = None,
        place_target_xy: list[float] | None = None,
    ) -> AgentPlanResponse:
        return self.planner.plan(target_name, place_target_xy)

    async def execute_recovery(
        self,
        target_name: Optional[str] = None,
        place_target_xy: list[float] | None = None,
    ) -> AgentExecutionResponse:
        plan = self.planner.plan(target_name, place_target_xy)
        if plan.status != "success" or not plan.steps:
            return AgentExecutionResponse(
                status="failure",
                target=plan.target,
                executed_steps=[],
                plan=plan.steps,
                message="agent 未生成有效恢复计划",
            )
        result = await self.executor.execute_plan(plan.steps, target_name=plan.target)
        return result

    async def run_task(
        self,
        target_name: str,
        *,
        place_target_xy: list[float],
        max_recovery_rounds: int = 2,
    ) -> AgentExecutionResponse:
        base.place_target_xy = [float(place_target_xy[0]), float(place_target_xy[1])]
        workflow = [
            AgentPlanStep(action="camera-image"),
            AgentPlanStep(action="detect-object", parameters={"target_class": target_name}),
            AgentPlanStep(action="create-cloud"),
            AgentPlanStep(action="create-grasp"),
            AgentPlanStep(action="move-pregrasp"),
            AgentPlanStep(action="gripper-action", parameters={"flag": 0}),
            AgentPlanStep(action="move-grasp"),
            AgentPlanStep(action="gripper-action", parameters={"flag": 1}),
            AgentPlanStep(action="vertical-grasp"),
            AgentPlanStep(
                action="execute-grasp2",
                parameters={"x": float(place_target_xy[0]), "y": float(place_target_xy[1])},
            ),
            AgentPlanStep(action="gripper-action", parameters={"flag": 0}),
            AgentPlanStep(action="execute-init"),
        ]

        executed_steps: list[dict[str, Any]] = []
        recovery_round = 0

        for step in workflow:
            result = await self.executor.execute_step(step)
            executed_steps.append(result)
            if result.get("status") == "success":
                continue

            if recovery_round >= max_recovery_rounds:
                return AgentExecutionResponse(
                    status="failure",
                    target=target_name,
                    executed_steps=executed_steps,
                    plan=[],
                    message=f"动作失败且超过最大恢复次数: {step.action}",
                )

            recovery_round += 1
            recovery = await self.execute_recovery(target_name, place_target_xy)
            executed_steps.append(
                {
                    "status": recovery.status,
                    "action": "agent-recovery",
                    "parameters": {"round": recovery_round},
                    "message": recovery.message,
                    "recovery_plan": [_dump_model(step) for step in recovery.plan],
                    "recovery_steps": recovery.executed_steps,
                }
            )
            if recovery.status != "success":
                return AgentExecutionResponse(
                    status="failure",
                    target=target_name,
                    executed_steps=executed_steps,
                    plan=[],
                    message=f"agent 恢复失败，主流程终止，失败动作: {step.action}",
                )

        score: dict[str, Any] | None = None
        try:
            score = base.get_sorce(
                base.content_list_all,
                base.task_list,
                place_target_xy=base.get_current_place_target_xy(),
            )
        except Exception as exc:  # noqa: BLE001
            score = {"status": "failure", "score": "0", "reason": f"score error: {exc}"}

        return AgentExecutionResponse(
            status="success",
            target=target_name,
            executed_steps=executed_steps,
            plan=[],
            message="主流程执行完成",
            score=score,
        )


runner = GraspExceptionHandlingAgentRunner()


@app.post("/agent/plan-recovery", description="使用 LangChain agent 生成恢复计划")
async def agent_plan_recovery(request: AgentRecoveryRequest) -> dict[str, Any]:
    place_target_xy = None
    if request.place_x is not None and request.place_y is not None:
        place_target_xy = [float(request.place_x), float(request.place_y)]
    plan = await runner.plan_recovery(request.target_name, place_target_xy)
    return _dump_model(plan)


@app.post("/agent/execute-recovery", description="使用 LangChain agent 生成并执行恢复计划")
async def agent_execute_recovery(request: AgentRecoveryRequest) -> dict[str, Any]:
    place_target_xy = None
    if request.place_x is not None and request.place_y is not None:
        place_target_xy = [float(request.place_x), float(request.place_y)]
    result = await runner.execute_recovery(request.target_name, place_target_xy)
    return _dump_model(result)


@app.post("/agent/run-task", description="执行抓取主流程，并在失败时使用 agent 做异常恢复")
async def agent_run_task(request: AgentRunTaskRequest) -> dict[str, Any]:
    if not base.grasp_service.env_initialized:
        raise HTTPException(status_code=500, detail="机器人环境未初始化")
    result = await runner.run_task(
        request.target_name,
        place_target_xy=[float(request.place_x), float(request.place_y)],
        max_recovery_rounds=max(0, int(request.max_recovery_rounds)),
    )
    return _dump_model(result)
