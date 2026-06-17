# 双臂协调层设计备忘

## 状态

暂缓实现。当前真机实验不一定需要双臂协调，因此先保留为后续扩展设计，不进入当前必须实现路径。

## 为什么需要协调层

双臂任务不建议直接恢复一套固定的 `G4/dual_arm_*` 技能。更通用的做法是复用已有左右单臂基础技能，在上层增加一个协调层，用来描述双臂动作之间的同步关系、安全检查和失败处理。

单臂技能负责具体动作，例如：

```text
move_to_pregrasp(side=left/right)
approach_object(side=left/right)
left_gripper_close / right_gripper_close
left_vertical_lift / right_vertical_lift
place_object(side=left/right)
open_gripper_release(side=left/right)
```

协调层负责这些动作之间的关系，例如：

```text
左右臂是否同时执行
左右 TCP 是否对齐
左右高度差是否过大
两臂路径是否可能互相碰撞
物体是否倾斜或滑移
一侧失败时是否停止另一侧
```

## 推荐抽象

协调层不直接控制电机，也不直接替代左右臂技能。它输出结构化的双臂计划，再由真机执行器或 MuJoCo 沙盒展开为已有单臂技能。

示例：

```json
{
  "action": "bimanual_step",
  "mode": "sync",
  "left": {
    "skill": "approach_object",
    "parameters": {"side": "left"}
  },
  "right": {
    "skill": "approach_object",
    "parameters": {"side": "right"}
  },
  "checks": [
    "tcp_alignment",
    "height_difference",
    "collision_risk"
  ]
}
```

执行时可以展开为：

```text
left: approach_object
right: approach_object
check_bimanual_alignment
check_bimanual_collision_risk
```

## 最小协调动作集合

如果后续真机需要双臂，可以先支持以下抽象动作：

```text
bimanual_pregrasp
bimanual_approach
bimanual_gripper_close
bimanual_lift
bimanual_transport
bimanual_place
bimanual_release
check_bimanual_alignment
check_bimanual_force_balance
check_object_slip
```

这些动作不是底层技能，而是计划层动作。它们应被展开为左右臂单臂技能和检查步骤。

## 与经验库的关系

经验库应保存两类信息：

```text
1. 单臂技能执行经验
2. 双臂协调约束经验
```

例如，真机或沙盒中出现“双臂高度不同步导致物体倾斜”，应写入为协调层风险经验，而不是简单归因到某个单臂技能失败。

可记录字段：

```text
bimanual_mode
left_skill
right_skill
sync_constraint
max_height_error
tcp_alignment_error
force_balance_error
object_slip_distance
failure_side
recovery_strategy
```

## 与 LLM 计划的关系

LLM 可以生成双臂协调计划，但必须满足两个限制：

```text
1. 不直接生成未注册的底层机器人控制 API
2. 只生成可被协调层展开的结构化动作
```

推荐输出：

```json
{
  "goal": "use both arms to stabilize and lift the object",
  "steps": [
    {
      "action": "bimanual_pregrasp",
      "parameters": {
        "left_skill": "move_to_pregrasp",
        "right_skill": "move_to_pregrasp",
        "mode": "sync"
      }
    },
    {
      "action": "check_bimanual_alignment",
      "parameters": {
        "max_tcp_error": 0.03,
        "max_height_error": 0.02
      }
    },
    {
      "action": "bimanual_gripper_close",
      "parameters": {
        "mode": "sync",
        "stop_if_one_arm_fails": true
      }
    }
  ]
}
```

## 与沙盒验证的关系

沙盒验证不应只检查任务成功，还要检查协调层风险：

```text
left_right_tcp_error
left_right_height_error
object_tilt_proxy
object_slip_distance
contact_lost_side
collision_between_arms
force_balance_proxy
```

如果这些指标超阈值，即使任务最终成功，也应给 LLM rewrite 提供反馈。

## 后续实现位置

如果后续确认真机需要双臂，建议实现位置为：

```text
experience_system/experience_core/bimanual_coordination.py
experience_system/tools/build_bimanual_robot_plan.py
```

其中：

```text
bimanual_coordination.py
  定义协调层 schema、计划展开、同步约束校验

build_bimanual_robot_plan.py
  把 LLM 生成的双臂协调计划转换为 validated_robot_plan
```

MuJoCo 或真机执行器只负责执行展开后的单臂技能和检查步骤。

## 当前结论

双臂协调层可以复用单臂技能，但不能简单把左臂和右臂脚本串起来。真正需要补的是计划层同步约束、失败联动和协调风险 critic。当前先不实现，等真机任务确认需要双臂后再接入。
