# RoboCritics 对 Sim-Real 双源经验库的启发

## 1. 阅读对象

本文档记录对以下论文的分析：

1. `RoboCritics1.pdf`

重点关注该论文对当前研究方向 **Sim-Real 双源经验库与异常恢复沙盒推演** 的启发，尤其是执行前安全检查、motion-level critic、结构化反馈、候选恢复动作修正和 LLM 规划闭环。

## 2. 论文的核心内容

RoboCritics 面向 LLM 机器人编程中的安全和可靠性问题。LLM 可以根据自然语言生成机器人程序，但这些程序往往难以验证，可能存在碰撞、关节速度过快、末端姿态不安全、夹爪运动风险等问题。

论文提出的核心做法是：

```text
LLM 生成机器人程序
→ 在仿真中执行
→ motion-level critics 分析执行轨迹
→ 输出结构化问题反馈
→ LLM 根据反馈修正程序
→ 用户验证
→ 真实机器人执行
```

其中 critic 不是只看代码文本，而是分析运动轨迹。论文中包括的 critic 类型有：

1. **Collision critic**：检查夹爪或机器人是否与环境发生碰撞或过近。
2. **Joint speed critic**：检查关节速度是否超过安全阈值。
3. **End-effector pose critic**：检查末端姿态和运动方向是否存在风险。
4. **Pinch-point critic**：检查机器人结构之间是否形成夹点风险。
5. **Space-usage critic**：检查机器人运动占用空间是否过大。

该工作说明：机器人程序的可靠性不能只靠 LLM 自我反思或 prompt 规则，必须引入外部、可执行、基于轨迹证据的 critic。

## 3. 对当前研究的总体启发

当前研究中的沙盒推演不能只判断：

```text
候选恢复动作是否成功
```

还必须判断：

```text
候选恢复动作是否安全、稳定、可解释、可迁移
```

这正是 RoboCritics 可以补强的地方。对于异常恢复任务，恢复动作通常发生在偏离正常轨迹之后，风险比普通任务更高。因此，沙盒推演阶段应增加 critic 层：

```text
candidate recovery action
→ sandbox rollout
→ motion-level critic
→ score / reject / rewrite
→ pseudo-real or real execution
```

这能把当前系统从“仿真验证成功”升级为“仿真验证成功且通过安全 critic”。

## 4. 对沙盒评分函数的启发

当前研究文档中已有：

```text
sandbox_score =
  a1 * predicted_success
  - a2 * collision_risk
  - a3 * pose_error
  - a4 * recovery_time
  - a5 * sim_real_uncertainty
```

结合 RoboCritics，可以把 `safety_risk` 拆成更具体的 motion-level critic：

```text
critic_score {
  collision_risk,
  joint_speed_risk,
  joint_limit_risk,
  end_effector_pose_risk,
  pinch_point_risk,
  workspace_usage_risk,
  object_contact_risk
}
```

最终候选动作评分可以写成：

```text
candidate_score =
  recovery_success_score
  + retrieved_success_prior
  - failed_memory_risk
  - sim_real_gap_uncertainty
  - critic_risk_score
```

其中：

```text
critic_risk_score =
  c1 * collision_risk
  + c2 * joint_speed_risk
  + c3 * end_effector_pose_risk
  + c4 * pinch_point_risk
  + c5 * workspace_usage_risk
```

这样可以避免某个候选动作虽然能抓起物体，但路径明显不安全。

## 5. 对经验库字段的启发

当前经验库已有：

```text
critic_result
failure_taxonomy
execution_feedback
validation_evidence
```

结合 RoboCritics，建议把 critic 结果结构化保存：

```text
critic_result {
  overall_status: pass / warning / reject,
  collision: {
    status,
    min_distance,
    involved_objects,
    evidence_timestep
  },
  joint_speed: {
    status,
    max_speed,
    joint_name,
    threshold
  },
  end_effector_pose: {
    status,
    pose_score,
    risky_segment
  },
  pinch_point: {
    status,
    min_link_distance,
    evidence_timestep
  },
  suggested_fix
}
```

失败经验也可以将 critic 诊断写入：

```text
failure_taxonomy {
  failure_type,
  root_cause,
  critic_flags,
  failed_predicates
}
```

这会让失败经验更可解释，而不是只有 `failure_reason` 字符串。

## 6. 对 LLM 规划修正的启发

RoboCritics 的一个关键点是：critic 不只是打分，还要生成结构化反馈，让 LLM 能修正程序。

当前研究也可以采用类似闭环：

```text
LLM 生成恢复计划
→ sandbox rollout
→ critic 发现问题
→ 将 critic feedback 写入 prompt
→ LLM 重写恢复计划
→ 再次沙盒验证
```

例如：

```text
critic feedback:
  该恢复计划在 move-grasp 阶段与障碍物距离过近，
  且 close-gripper 前没有完成姿态对齐。
  请加入 move-pregrasp 安全中间点，并降低 approach 速度。
```

这比简单提示“计划失败，请重试”更有效。

## 7. 对没有真机阶段的启发

当前没有真机时，RoboCritics 更有价值。因为你可以先在 `experiment-sim-wrapper` 中建立：

```text
simulation execution
→ critic verification
→ pseudo-real validation
```

也就是说，不是所有仿真成功都进入高置信经验库，只有：

```text
recovery_success = true
critic_status = pass or warning acceptable
virtual_validation_success = true
```

的经验，才能晋升为 `simulation_validated` 或 pseudo-real high-confidence memory。

这样能减少“仿真中侥幸成功”的经验污染经验库。

## 8. 对现有 experiment-sim-wrapper 的对应关系

当前系统已经有若干可承接 critic 的字段和逻辑：

1. `execution_feedback`
2. `failure_taxonomy`
3. `validation_evidence`
4. `recovery_success_criteria`
5. `task_success_criteria`
6. `sim_wrapper` 虚拟验证
7. `invalid_plan_count`
8. `failed_plan_rewrite`

因此，RoboCritics 可以作为一个升级方向：

```text
sim_wrapper validation
→ sim_wrapper validation + motion-level critic
```

第一阶段不需要实现复杂 critic，可以先做规则化 critic：

```text
collision_count
min_pinch_distance
max_joint_delta
end_effector_pose_error
gripper_contact_pattern
object_drop_risk
```

这些都可以从 MuJoCo 执行轨迹中计算。

## 9. 对实验设计的启发

可以设置如下对比：

```text
No critic:
  只看恢复是否成功

Prompt-only critic:
  只在 prompt 中提醒 LLM 注意安全

Motion-level critic:
  执行沙盒 rollout 后用规则 critic 检查

Motion-level critic + rewrite:
  critic 反馈给 LLM 重新生成计划
```

评价指标：

```text
recovery_success_rate
collision_count
unsafe_motion_count
invalid_plan_count
rewrite_success_rate
critic_rejection_rate
task_completion_rate
```

其中最能体现 RoboCritics 启发的是：

```text
unsafe_motion_count
rewrite_success_rate
```

## 10. 与已有启发文档的关系

RoboCritics 与前面几篇文档的关系如下：

1. RISE 提供 imagined rollout 和 value 评分思想。
2. Learning From Failure 提供失败经验风险先验。
3. Plan in Sandbox 提供沙盒经验生成。
4. RoboCritics 提供执行前安全和轨迹级 critic。

组合后，当前系统可以形成：

```text
候选动作生成
→ 失败经验风险过滤
→ 沙盒 rollout
→ motion-level critic
→ LLM rewrite
→ pseudo-real execution
→ 经验回写
```

## 11. 不能直接照搬的部分

RoboCritics 主要面向 end-user robot programming，而当前研究是异常恢复经验库。因此不适合直接照搬：

1. Web UI 和用户研究部分。
2. 一键修复交互界面。
3. UR3e 具体实现。
4. 面向普通用户的机器人编程流程。

当前研究应吸收的是它的 **motion-level external critic** 思想。

## 12. 可写入当前研究方案的关键表述

后续整合到总方案时，可以使用如下表述：

```text
RoboCritics 表明，LLM 生成的机器人程序不能仅依靠文本规则或模型自我反思验证，其安全性和可靠性必须通过外部 motion-level critics 在执行轨迹上进行检查。该思想对异常恢复尤为重要，因为恢复动作通常发生在偏离正常状态之后，碰撞、关节速度、末端姿态和夹点风险更高。

因此，本研究在沙盒推演阶段引入 critic 检查，将候选恢复动作的成功率、失败经验风险和运动安全风险共同纳入评分，并将 critic 反馈用于 LLM 恢复计划重写。
```

## 13. 阶段性结论

该论文对当前研究的核心启发可以概括为：

```text
沙盒推演不能只问“能不能成功”，
还必须问“这个成功是不是安全、稳定、可解释”。
```

具体而言，当前研究可以吸收以下内容：

1. 引入 motion-level critic 作为沙盒验证的后置检查。
2. 将碰撞、关节速度、末端姿态、夹点风险写入 critic_result。
3. 用 critic feedback 驱动 LLM 重写恢复计划。
4. 在没有真机阶段，用 critic 提高 pseudo-real memory 的可信度。

因此，该论文非常适合作为当前研究中 **安全 critic、执行前轨迹检查、LLM 计划修正和高置信经验晋升** 的支撑材料。
