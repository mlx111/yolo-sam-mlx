# 无真机阶段剩余工作清单

这份文档只列当前**不依赖真机**还能继续推进的内容。

## 当前状态

### P0 物理沙盒调稳

状态：**基础目标已完成，后续只做增强和报告化。**

目标：

```text
让 physical-actuator sandbox 成为默认执行路径，并在 G3 clean 上稳定可跑。
```

主要工作：

- 调小步长和速度。
- 增加 settle steps。
- 补 actuator tracking 诊断。
- 记录关节限位、workspace 越界、contact/slip 风险。
- direct-position 仅保留为显式调试/对照，不作为默认实验路径。

已完成：

- `physical-actuator` 已作为默认执行路径。
- `direct_qpos` 默认已收紧为 `false`，只保留显式调试开关。
- G3 clean physical 完整链路已通过。
- field_atomic 中 base、torso、gripper 等基础动作已按物理路径 smoke 通过。
- 已有 physical sandbox perturbation report，可统计 pose/delay/gain/gripper 扰动下的结果。
- 已有 `physical_default_audit_report`，当前核心文件 `direct_qpos=true` 默认计数为 0。

还能做：

- 后续新增技能时同步更新 `physical_default_audit_report`。
- 扩展 actuator tracking 诊断，例如 base/torso/arm/gripper 分组误差。

### P1 动态 runtime scene

状态：**观测 schema 已固定，具体观测到 XML 的生成需要等现场数据格式确定后再实现。**

目标：

```text
把结构化观测继续完善为 runtime_sandbox_scene_v1。
```

主要工作：

- 固定现场观测 JSON 结构。
- 明确 RGB-D / LiDAR / 机器人状态需要转成哪些字段。
- 明确对象、障碍物、放置区如何映射到 runtime XML。
- 现场拿到真实感知输出后，再实现具体转换器。

已有基础：

- `experience_system/templates/field_runtime_scene_observation_template.json`
- `experience_system/docs/runtime_scene_observation_schema_zh.md`

计划实现，但暂不直接写死：

```text
field_runtime_scene_observation_v1
-> runtime_sandbox_scene_v1
-> runtime_scene.xml
-> MuJoCo sandbox rollout
```

后续工具建议：

```text
experience_system/tools/build_runtime_sandbox_scene.py
```

建议输入：

```text
--observation scene_observation.json
--base-model galaxea_mujoco/r1pro_g3_sorting_scene.xml
--save-scene runtime_scene.xml
--save-report runtime_scene_report.json
```

建议输出报告：

```json
{
  "schema_version": "runtime_scene_build_report_v1",
  "object_count": 0,
  "obstacle_count": 0,
  "place_zone_count": 0,
  "target_object_mapped": false,
  "runtime_xml_valid": false,
  "field_atomic_smoke_success": false
}
```

暂不实现原因：

- 现场 RGB-D 相机、LiDAR、物体检测、坐标系外参的实际输出格式还没有确定。
- 如果现在强行写死映射规则，后面很可能和现场数据不一致。
- 当前阶段只需要把 schema、字段含义、工具入口和报告格式固定，具体生成逻辑等现场数据回来后再补。

### P2 LLM 计划重写闭环

状态：**真实 LLM sequential rewrite 链路已跑通，但还没有成功收敛证据。**

目标：

```text
让 critic 反馈和失败记忆真正进入 prompt。
```

主要工作：

- 接入 stage retrieval。
- 接入 planner_input。
- 接入 sandbox critic 反馈。
- 接入失败经验和 gap 经验。

已完成：

- LLM 可以生成 field_atomic 计划。
- planner_input 已能读取 field_atomic 成功/失败经验和参数先验。
- sandbox 验证后可写回成功/失败经验。

已补报告：

- `results/memory/universal_pipeline_calibration_v1/rewrite_loop_ablation_report.json`
- `results/memory/universal_pipeline_calibration_v1/rewrite_loop_ablation_report.md`
- `results/memory/universal_pipeline_calibration_v1/rewrite_loop_g3_clean_real_report.json`
- `results/memory/universal_pipeline_calibration_v1/rewrite_loop_g3_clean_real_validated_plan.json`

当前报告结论：

- single-plan baseline 已有：真实 LLM 生成单个计划，经过语义校验、sandbox critic 和 validated_robot_plan 导出。
- multi-candidate critic ranking 已有：多个 LLM 候选计划进入 sandbox 并由 critic 排序。
- failure memory 证据已有：memory-aware ranking 中失败经验参与风险证据，candidate changed rate 为 0.375。
- parameter priors 证据已有：field_atomic writeback 能在下一轮 planner_input 中提供参数先验。
- 真实 LLM sequential rewrite 证据已有：`rewrite_rounds=1`，`critic_feedback_history` 非空，说明 critic feedback 已进入下一轮 rewrite。
- 成功收敛证据还没有：真实 LLM rewrite 后仍为 `final_sandbox_status=reject`，不能声称 critic feedback 已经提升恢复成功率。

当前真实 LLM rewrite 失败原因：

- 第 0 轮生成 `retry_lift_after_grasp_check`，但缺少前置事实 `object_grasped`。
- 第 1 轮生成 `recover_from_joint_limit` 和 `slow_cartesian_approach`，但 `recover_from_joint_limit` 缺少前置事实 `target_object_selected`。
- 这说明闭环机制有效，但经验库中还缺少“失败 -> 合法重写 -> sandbox 接受”的高质量示例。

后续如果要补强论文，需要积累或构造成功收敛样本：

```text
round 0: LLM plan -> sandbox critic reject/warn
round 1: critic feedback + failure memory + parameter priors -> LLM rewrite
round 1 sandbox accept
```

新增指标：

- `rewrite_rounds > 0`（已具备）
- `critic_feedback_history` 非空（已具备）
- `final_sandbox_status=accept`（未具备）
- `critic_risk_delta`
- `sandbox_status_changed`
- `parameter_changed_count`
- `repeated_failure_rate`

当前不建议为了让 demo 通过而显著加强 prompt 约束，因为这会污染 baseline。更合理的路线是：

```text
保持 prompt 中性
-> 真机/仿真失败经验写回
-> 失败恢复成功样本进入经验库
-> LLM 从经验检索和参数先验中自然获得更好的 rewrite 行为
```

### P3 沙盒写回策略细化

状态：**基础实现和构造压力测试均已完成，后续只需按新实验补报告。**

目标：

```text
把哪些仿真经验该写、哪些该跳过，规则说清楚并落到报告里。
```

主要工作：

- 保留失败和高风险经验。
- 合并重复低风险成功。
- 跳过低价值候选。
- 让 write_policy 报告更可解释。

已完成：

- 成功和失败 field_atomic 经验都能写回。
- sandbox 验证过的仿真经验可按 write_policy 进入经验库。
- 已有 `write_policy_audit_report`，能统计 write / skip / merge / reject 及原因。
- 已有 `run_write_policy_pressure_test.py` 构造测试，能稳定触发 write / merge / skip / reject 四类决策。

还能做：

- 后续每次新增经验类型时，把该类型补进 pressure test。
- 将 pressure test 结果复制到正式 results 目录，供论文 appendix 固定引用。

### P4 论文和证据对齐

状态：**paper evidence summary 已更新，但文档仍需继续同步。**

目标：

```text
让文档和论文只描述当前已实现内容。
```

主要工作：

- 更新 paper evidence summary。
- 更新 appendix。
- 清理旧 G4 / 双臂主线表述。
- 把 sandbox / writeback 边界写清楚。

已完成：

- paper evidence summary 已包含 field_atomic、physical perturbation、runtime schema、write_policy audit、真实 LLM rewrite ablation 等证据项。
- 当前 paper evidence summary 为 `claim_count=24`，`supported_claim_count=23`，`missing_report_count=0`。
- 已明确不能声称真机成功率提升。

还能做：

- 更新 appendix 文档，把最新 physical-first 默认、field_atomic 写回、runtime scene schema、真实 LLM rewrite 失败分析加进去。
- 清理或标注旧 G4 / 双臂 / direct-position 表述。
- 把“当前能证明什么 / 不能证明什么”单独整理成论文安全表述表。

## 暂时不做

- 真机自动下发执行器
- 真机传感器自动建场景
- 真机执行日志自动写回
- 真机成功率提升结论

## 推荐顺序

```text
P1 runtime scene schema 等现场数据
-> P2 LLM rewrite 成功收敛样本等经验积累
-> P3 write_policy pressure/audit cases
-> P4 docs/paper sync
-> P0 physical audit report
```
