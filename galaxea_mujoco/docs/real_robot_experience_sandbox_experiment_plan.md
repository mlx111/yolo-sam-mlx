# 真机经验库与沙盒验证实验流程

## 目标

本实验目标是把真机异常恢复流程接入经验库和 MuJoCo 沙盒：

```text
真机场景采集
-> 构建沙盒场景
-> 真机执行错误技能序列并出现异常
-> 经验库检索真机/仿真经验
-> LLM 生成异常处理技能序列
-> 沙盒物理驱动验证
-> critic 评估与必要重写
-> 输出 validated_robot_plan
-> 真机执行恢复计划
-> 真机经验与仿真经验写回经验库
```

## 当前可以支持的部分

### 1. 真机 episode 写入经验库

已有能力：

```text
experience_system/experience_adapters/real_episode.py
source/import_real_episode.py
source/validate_real_episode.py
docs/real_episode_import_format.md
docs/real_episode_template.json
```

可以把真机日志、图片、深度图、激光雷达、腕部受力、技能执行结果整理成 real episode，然后导入统一经验库。

当前状态：

```text
可做，但需要现场按照模板保存数据。
```

### 2. 结构化场景生成 MuJoCo XML

已有基础：

```text
experience_system/templates/field_runtime_scene_observation_template.json
experience_system/docs/runtime_scene_observation_schema_zh.md
```

目标是从现场结构化观测 JSON 生成 MuJoCo 沙盒 XML。

当前状态：

```text
现场观测 schema 已固定。
具体 JSON -> XML 转换器暂不写死，需要等现场 RGB-D/LiDAR/外参输出格式确定后再实现。
还缺真机传感器 -> 结构化 JSON 的自动抽取。
```

### 3. LLM 生成异常处理计划

已有能力：

```text
experience_system/experience_core/llm_provider.py
experience_system/experience_core/recovery_plan.py
experience_system/tools/generate_recovery_plan_llm.py
experience_system/tools/run_recovery_plan_sandbox_loop.py
```

经验库会先检索相关成功经验、失败经验、sim-real gap 和 stage context，整理为 `planner_input`，再放入 LLM prompt。

当前状态：

```text
可做。
```

### 4. 计划语义校验、沙盒推演、critic、重写

已有能力：

```text
experience_system/experience_core/skill_semantics.py
experience_system/tools/candidate_sandbox.py
experience_system/tools/run_recovery_plan_sandbox_loop.py
experience_system/experience_core/critic.py
```

作用：

```text
检查技能顺序是否合法
在 MuJoCo 中执行候选计划
评估任务成功、接触、滑移、轨迹风险
失败时把 critic feedback 反馈给 LLM 重写
```

当前状态：

```text
单臂/G3 主线可用。
G4/旧双臂仿真已移到 legacy，不作为当前主线。
```

### 5. validated_robot_plan 输出

已有能力：

```text
experience_system/experience_core/robot_plan_executor.py
experience_system/tools/run_validated_robot_plan_dry_run.py
```

系统可以输出经过校验和沙盒验证的结构化计划：

```text
validated_robot_plan_v1
```

当前状态：

```text
可以 dry-run。
不能直接控制真机，因为缺真机 executor。
```

## 当前缺失的关键部分

### 1. 真机场景感知到 runtime_sandbox_scene_v1

需要把现场传感器输出转换为：

```json
{
  "schema_version": "runtime_sandbox_scene_v1",
  "scene_id": "real_scene_xxx",
  "table_pose": [0.0, 0.0, 0.0],
  "target_object": "target_object",
  "objects": [],
  "obstacles": [],
  "place_zones": []
}
```

需要输入：

```text
RGB/RGB-D 图像
目标检测结果
mask 或 bbox
深度估计或点云
相机到机器人 base 的外参
桌面高度
目标物体尺寸估计
放置区位置
```

当前缺口：

```text
还没有自动脚本把真机图片/深度/雷达直接变成 runtime_sandbox_scene_v1。
```

### 2. validated_robot_plan 到真机执行器

需要真机侧提供技能 API 映射：

```text
move_to_pregrasp -> 真机接口
approach_object -> 真机接口
left_gripper_close -> 真机接口
left_vertical_lift -> 真机接口
place_object -> 真机接口
open_gripper_release -> 真机接口
```

还需要：

```text
坐标系定义
安全限位
速度限制
力/力矩限制
技能返回格式
失败原因格式
急停/人工确认机制
```

当前缺口：

```text
经验库只能输出 validated_robot_plan，不能直接下发真机。
```

### 3. 在线写回闭环

目标闭环：

```text
真机执行恢复计划
-> 保存图片、深度、雷达、腕力、关节状态、技能结果
-> 生成 real episode
-> import_real_episode
-> 写入 universal_experience_library
```

当前缺口：

```text
导入工具已有，但还缺现场自动保存和一键写回脚本。
```

### 4. 仿真经验写回

沙盒验证会先产出可写回的仿真经验候选，但是否入库由 `write_policy` 决定：

```text
sandbox rollout result
-> R1ProMujocoAdapter
-> ExperienceEntry
-> critic_result
-> write_policy
-> ExperienceLibrary.add_with_policy
-> universal_experience_library
```

当前状态：

```text
沙盒 rollout 默认只产出证据，不自动写库；
只有通过上层 pipeline / smoke / writeback 脚本显式调用时，才会写入经验库。
```

## 经验如何发挥作用

经验库不是直接控制机器人，而是通过四种方式影响计划：

```text
1. 写入 LLM prompt
2. 参与候选计划打分
3. 用 sim-real gap 校准沙盒
4. 用失败经验和 critic 风险过滤候选
```

流程：

```text
真机/仿真经验
-> 检索相关经验
-> 构造 planner_input
-> LLM 生成恢复计划
-> 语义校验
-> 沙盒物理驱动验证
-> critic 评分
-> validated_robot_plan
```

## 仿真物理驱动要求

实验中沙盒应优先使用：

```text
--control-mode physical
```

原因：

```text
physical-actuator 更接近真实执行约束，可以暴露 actuator 跟踪、关节限位、接触和滑移问题。
direct-position 只保留为显式调试/对照模式，不作为当前实验主线。
```

当前限制：

```text
physical sandbox 不是完整真机等价模型。
不能声称仿真能准确预测真机成功率。
```

安全表述：

```text
系统支持在真机执行前，对候选异常处理计划进行物理驱动 MuJoCo 沙盒推演和规则 critic 风险过滤。
```

## 最小可落地实验版本

如果现在要做现场实验，建议先做最小闭环：

```text
1. 手工或半自动记录真机场景对象位置
2. 生成 runtime_sandbox_scene_v1
3. 生成 MuJoCo runtime XML
4. 真机执行一个预设错误技能序列并记录异常
5. 把异常 episode 导入经验库
6. LLM 基于经验生成恢复计划
7. 沙盒验证并输出 validated_robot_plan
8. 人工确认后在真机侧手动执行对应技能
9. 保存执行结果并写回经验库
```

这版不要求自动下发真机，但可以验证经验库主线：

```text
经验检索
LLM 计划生成
沙盒推演
critic 风险过滤
validated_robot_plan 输出
真机经验写回
```

## 后续需要补齐的模块

优先级如下：

```text
P0: real_scene_to_runtime_sandbox_scene.py
P1: validated_robot_plan_real_executor.py
P2: real_execution_episode_writer.py
P3: sandbox_and_real_writeback_pipeline.py
P4: physical sandbox 控制诊断增强
```

## 当前结论

现有机器人仿真和经验库设计可以支撑实验主线，但不能直接完成全自动真机闭环。

当前能可靠完成的是：

```text
经验库检索
LLM 异常处理计划生成
语义校验
MuJoCo 沙盒验证
critic 评分
validated_robot_plan 输出
real episode 导入
```

当前还需要现场接口支持的是：

```text
真机感知自动建场景
validated_robot_plan 自动下发真机
真机执行日志自动写回
```
