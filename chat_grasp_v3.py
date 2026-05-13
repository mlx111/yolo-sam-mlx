import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, parse, request


BASE_URL = os.environ.get("CHAT_GRASP_BASE_URL", "http://localhost:8080")
TASK_FILE = "renwu.json"
RECOVERY_FILE = "yichang.json"
MAX_RECOVERY_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 60

SUPPORTED_ACTIONS = {
    "reset-environment",
    "camera-image",
    "detect-object",
    "create-cloud",
    "create-grasp",
    "move-pregrasp",
    "gripper-action",
    "move-grasp",
    "vertical-grasp",
    "execute-grasp2",
    "execute-init",
    "gen-sorce",
}

ACTION_ALIASES = {
    "execute-grasp1": [
        {"action": "move-pregrasp", "parameters": {}},
        {"action": "move-grasp", "parameters": {}},
        {"action": "gripper-action", "parameters": {"state": 1}},
        {"action": "vertical-grasp", "parameters": {}},
    ]
}


def make_local_failure(message: str, *, action: str, error_type: str = "local") -> Dict[str, Any]:
    return {
        "ok": False,
        "http_status": None,
        "status": "failure",
        "message": message,
        "payload": {
            "status": "failure",
            "message": message,
            "error_type": error_type,
            "action": action,
        },
        "raw_text": "",
    }


def load_task_file(path: str) -> List[Dict[str, Any]]:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"任务文件不存在: {task_path}")

    with task_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"任务文件顶层必须是 JSON 数组: {task_path}")
    return data


def load_normalized_task_file(path: str) -> List[Dict[str, Any]]:
    raw_items = load_task_file(path)
    return normalize_task_items(raw_items, source_name=path)


def expand_action_alias(action: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    if action not in ACTION_ALIASES:
        return [{"action": action, "parameters": parameters}]

    expanded: List[Dict[str, Any]] = []
    for item in ACTION_ALIASES[action]:
        expanded.append(
            {
                "action": item["action"],
                "parameters": dict(item.get("parameters", {})),
            }
        )
    return expanded


def normalize_task_items(task_items: List[Dict[str, Any]], *, source_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(task_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{source_name} 第 {index} 项必须是对象。")

        action = item.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"{source_name} 第 {index} 项缺少合法 action。")
        action = action.strip()

        parameters = item.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError(f"{source_name} 第 {index} 项的 parameters 必须是对象。")

        expanded_items = expand_action_alias(action, parameters)
        for expanded in expanded_items:
            expanded_action = expanded["action"]
            if expanded_action not in SUPPORTED_ACTIONS:
                raise ValueError(f"{source_name} 第 {index} 项包含不支持的动作: {expanded_action}")
            normalized.append(
                {
                    "action": expanded_action,
                    "parameters": dict(expanded.get("parameters", {})),
                    "source_index": index,
                    "source_action": action,
                }
            )
    return normalized


def build_action_request(
    action: str,
    parameters: Dict[str, Any],
    *,
    auto_recover: bool = True,
) -> tuple[str, Dict[str, Any]]:
    if action == "detect-object":
        target_class = parameters.get("object")
        if not isinstance(target_class, str) or not target_class.strip():
            raise ValueError("detect-object 缺少 parameters.object")
        return action, {"target_class": target_class.strip()}

    if action == "gripper-action":
        state = parameters.get("state", parameters.get("flag"))
        if state not in {0, 1}:
            raise ValueError("gripper-action 的 parameters.state/flag 必须为 0 或 1")
        return action, {"flag": int(state), "auto_recover": str(bool(auto_recover)).lower()}

    if action == "execute-grasp2":
        try:
            x = float(parameters["x"])
            y = float(parameters["y"])
        except KeyError as exc:
            raise ValueError("execute-grasp2 缺少 parameters.x 或 parameters.y") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("execute-grasp2 的 parameters.x/y 必须是数字") from exc
        return action, {"x": x, "y": y, "auto_recover": str(bool(auto_recover)).lower()}

    if action in {"move-pregrasp", "move-grasp", "vertical-grasp", "execute-init"}:
        return action, {"auto_recover": str(bool(auto_recover)).lower()}

    if action in SUPPORTED_ACTIONS:
        return action, {}

    raise ValueError(f"不支持的动作: {action}")


def parse_response_payload(raw_text: str) -> Dict[str, Any] | None:
    if not raw_text.strip():
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def call_grasp_action(
    action: str,
    parameters: Dict[str, Any],
    *,
    base_url: str,
    auto_recover: bool = True,
) -> Dict[str, Any]:
    try:
        endpoint, query_params = build_action_request(action, parameters, auto_recover=auto_recover)
    except ValueError as exc:
        return make_local_failure(str(exc), action=action)

    query_string = parse.urlencode(query_params)
    url = f"{base_url.rstrip('/')}/{endpoint}"
    if query_string:
        url = f"{url}?{query_string}"

    req = request.Request(
        url,
        data=b"",
        headers={"accept": "application/json"},
        method="POST",
    )

    raw_text = ""
    http_status = None
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            http_status = response.getcode()
            raw_text = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        http_status = exc.code
        raw_text = exc.read().decode("utf-8", errors="replace")
        payload = parse_response_payload(raw_text)
        message = ""
        if payload is not None:
            message = str(payload.get("message") or payload.get("detail") or raw_text)
        else:
            message = raw_text or f"HTTP {http_status}"
        return {
            "ok": False,
            "http_status": http_status,
            "status": "failure",
            "message": message,
            "payload": payload if payload is not None else {"detail": raw_text},
            "raw_text": raw_text,
        }
    except error.URLError as exc:
        return {
            "ok": False,
            "http_status": None,
            "status": "failure",
            "message": f"无法连接 grasp 服务: {exc}",
            "payload": None,
            "raw_text": "",
        }

    payload = parse_response_payload(raw_text)
    if payload is None:
        return {
            "ok": False,
            "http_status": http_status,
            "status": "failure",
            "message": f"{action} 返回了非 JSON 响应",
            "payload": None,
            "raw_text": raw_text,
        }

    status = payload.get("status")
    if status not in {"success", "failure", "finished"}:
        return {
            "ok": False,
            "http_status": http_status,
            "status": "failure",
            "message": f"{action} 响应缺少合法 status",
            "payload": payload,
            "raw_text": raw_text,
        }

    message = str(payload.get("message") or "")
    return {
        "ok": http_status is not None and http_status < 400,
        "http_status": http_status,
        "status": status,
        "message": message,
        "payload": payload,
        "raw_text": raw_text,
    }


def should_stop_for_failure(result: Dict[str, Any]) -> bool:
    return (not result["ok"]) or result["status"] == "failure"


def print_action_result(step_no: int, total_steps: int, mode: str, action: str, response: Dict[str, Any]) -> None:
    print(f"[{mode}] step {step_no}/{total_steps} action={action}")
    print(f"  http_status={response['http_status']} status={response['status']}")
    if response.get("message"):
        print(f"  message={response['message']}")
    payload = response.get("payload")
    if isinstance(payload, dict):
        if "recovery_status" in payload:
            print(f"  recovery_status={payload.get('recovery_status')}")
        if "recovery_message" in payload:
            print(f"  recovery_message={payload.get('recovery_message')}")
        if "best_grasp" in payload:
            print("  best_grasp=available")


def _extract_place_target(task_items: List[Dict[str, Any]]) -> Dict[str, float] | None:
    for item in task_items:
        if item.get("action") != "execute-grasp2":
            continue
        parameters = item.get("parameters", {})
        try:
            return {
                "x": float(parameters["x"]),
                "y": float(parameters["y"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _is_gripper_open_step(item: Dict[str, Any]) -> bool:
    if item.get("action") != "gripper-action":
        return False
    state = item.get("parameters", {}).get("state", item.get("parameters", {}).get("flag"))
    return state == 0


def _ensure_recovery_completion(
    task_items: List[Dict[str, Any]],
    *,
    place_target: Dict[str, float] | None,
    failed_action: str | None,
) -> List[Dict[str, Any]]:
    enriched = [
        {
            **item,
            "parameters": dict(item.get("parameters", {})),
        }
        for item in task_items
    ]

    has_execute_grasp2 = False

    for item in enriched:
        if item.get("action") == "execute-grasp2":
            has_execute_grasp2 = True
            if place_target is not None:
                item["parameters"].setdefault("x", place_target["x"])
                item["parameters"].setdefault("y", place_target["y"])

    needs_place_tail = failed_action in {"vertical-grasp", "execute-grasp2"}
    if needs_place_tail and place_target is not None and not has_execute_grasp2:
        enriched.append(
            {
                "action": "execute-grasp2",
                "parameters": {"x": place_target["x"], "y": place_target["y"]},
            }
        )
        has_execute_grasp2 = True

    execute_grasp2_index = next(
        (index for index, item in enumerate(enriched) if item.get("action") == "execute-grasp2"),
        None,
    )
    if execute_grasp2_index is not None:
        tail_items = enriched[execute_grasp2_index + 1:]
        has_release_after_place = any(_is_gripper_open_step(item) for item in tail_items)
        has_execute_init_after_place = any(item.get("action") == "execute-init" for item in tail_items)
        if not has_release_after_place:
            enriched.append({"action": "gripper-action", "parameters": {"state": 0}})
        if not has_execute_init_after_place:
            enriched.append({"action": "execute-init", "parameters": {}})

    return enriched


def run_task_items(task_items: List[Dict[str, Any]], *, mode: str, source_name: str, base_url: str) -> Dict[str, Any]:

    total_steps = len(task_items)
    if total_steps == 0:
        return {
            "success": True,
            "mode": mode,
            "path": source_name,
            "failed_action": None,
            "failed_step": None,
            "response": None,
        }

    for step_no, item in enumerate(task_items, start=1):
        action = item["action"]
        parameters = item["parameters"]
        print(f"[{mode}] execute step {step_no}/{total_steps}: action={action}, parameters={parameters}")
        response = call_grasp_action(
            action,
            parameters,
            base_url=base_url,
            auto_recover=(mode != "recovery"),
        )
        print_action_result(step_no, total_steps, mode, action, response)
        if should_stop_for_failure(response):
            return {
                "success": False,
                "mode": mode,
                "path": source_name,
                "failed_action": action,
                "failed_step": step_no,
                "response": response,
            }

    return {
        "success": True,
        "mode": mode,
        "path": source_name,
        "failed_action": None,
        "failed_step": None,
        "response": None,
    }


def run_task_file(path: str, *, mode: str, base_url: str) -> Dict[str, Any]:
    try:
        task_items = load_normalized_task_file(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "mode": mode,
            "path": path,
            "failed_action": None,
            "failed_step": None,
            "response": make_local_failure(str(exc), action="load-task"),
        }
    return run_task_items(task_items, mode=mode, source_name=path, base_url=base_url)


def score(*, base_url: str) -> Dict[str, Any]:
    print("[score] execute action=gen-sorce")
    response = call_grasp_action("gen-sorce", {}, base_url=base_url)
    print_action_result(1, 1, "score", "gen-sorce", response)
    return response


def run_pipeline(task_file: str, recovery_file: str, *, base_url: str) -> int:
    recovery_attempts = 0
    setup_result = None
    print("[setup] execute action=reset-environment")
    setup_result = call_grasp_action("reset-environment", {}, base_url=base_url)
    print_action_result(1, 1, "setup", "reset-environment", setup_result)
    if should_stop_for_failure(setup_result):
        print("\n=== Summary ===")
        print("setup_success=False")
        print(f"setup_message={setup_result.get('message')}")
        print("main_success=False")
        print("recovery_triggered=False")
        print("score_status=skipped")
        print("score_message=reset-environment failed")
        return 1

    try:
        main_task_items = load_normalized_task_file(task_file)
    except Exception as exc:  # noqa: BLE001
        main_result = {
            "success": False,
            "mode": "main",
            "path": task_file,
            "failed_action": None,
            "failed_step": None,
            "response": make_local_failure(str(exc), action="load-task"),
        }
        main_place_target = None
    else:
        main_place_target = _extract_place_target(main_task_items)
        main_result = run_task_items(main_task_items, mode="main", source_name=task_file, base_url=base_url)
    recovery_result = None

    if not main_result["success"] and recovery_attempts < MAX_RECOVERY_ATTEMPTS:
        recovery_attempts += 1
        print(
            f"[main] failed at step={main_result['failed_step']} action={main_result['failed_action']}, "
            f"start recovery attempt {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}"
        )
        try:
            recovery_task_items = load_normalized_task_file(recovery_file)
            recovery_task_items = _ensure_recovery_completion(
                recovery_task_items,
                place_target=main_place_target,
                failed_action=main_result["failed_action"],
            )
        except Exception as exc:  # noqa: BLE001
            recovery_result = {
                "success": False,
                "mode": "recovery",
                "path": recovery_file,
                "failed_action": None,
                "failed_step": None,
                "response": make_local_failure(str(exc), action="load-task"),
            }
        else:
            recovery_result = run_task_items(
                recovery_task_items,
                mode="recovery",
                source_name=recovery_file,
                base_url=base_url,
            )

    score_result = score(base_url=base_url)

    print("\n=== Summary ===")
    print("setup_success=True")
    print(f"main_success={main_result['success']}")
    if not main_result["success"]:
        response = main_result["response"] or {}
        print(f"main_failed_action={main_result['failed_action']}")
        print(f"main_failed_message={response.get('message')}")
    print(f"recovery_triggered={recovery_result is not None}")
    if recovery_result is not None:
        print(f"recovery_success={recovery_result['success']}")
        if not recovery_result["success"]:
            response = recovery_result["response"] or {}
            print(f"recovery_failed_action={recovery_result['failed_action']}")
            print(f"recovery_failed_message={response.get('message')}")
    print(f"score_status={score_result.get('status')}")
    print(f"score_message={score_result.get('message')}")

    if main_result["success"]:
        return 0
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON-driven chat client for grasp_fastapi_completion_v3")
    parser.add_argument("--task-file", default=TASK_FILE, help="主任务 JSON 文件")
    parser.add_argument("--recovery-file", default=RECOVERY_FILE, help="异常任务 JSON 文件")
    parser.add_argument("--base-url", default=BASE_URL, help="grasp 服务地址")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(args.task_file, args.recovery_file, base_url=args.base_url)


if __name__ == "__main__":
    sys.exit(main())
