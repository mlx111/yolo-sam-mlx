# RISE 对 Sim-Real 双源经验库的启发

## 1. 阅读对象

本文档记录对以下材料的分析：

1. `Self-Improving Robot Policy with Compositional World Model.pdf`
2. `RISE/` 代码目录

重点关注 RISE 对当前研究方向 **Sim-Real 双源经验库与异常恢复沙盒推演** 的启发，尤其是真机经验采集、真实 episode 数据结构、候选动作推演和价值评分设计。

## 2. RISE 的核心机制

RISE 的核心不是传统物理仿真，而是使用真实机器人数据训练一个 **Compositional World Model**，再在该世界模型中进行想象式策略改进。

其整体流程可以概括为：

```text
真实机器人数据
→ 训练 action-conditioned dynamics model
→ 训练 progress value model
→ 在 imagination/world model 中 rollout 候选动作
→ value model 给 imagined trajectory 打分
→ 用 advantage 更新策略
```

RISE 的 Compositional World Model 由两个主要模块构成：

1. **Dynamics Model**
   - 输入历史多视角图像和动作 chunk；
   - 预测未来多视角观测；
   - 用于在不真实执行的情况下模拟候选动作后果。

2. **Value Model**
   - 对真实或 imagined observation 进行进度评分；
   - 区分成功、失败、中间进展和细微失败；
   - 将预测结果转化为 advantage，用于后续策略优化。

因此，RISE 的关键思想是：

```text
机器人不必每次都在真实世界试错，
可以先在由真实数据训练出的想象空间中预测动作后果，
再用 value model 判断哪些动作更值得执行。
```

这与当前课题中的“真机异常恢复前先进行沙盒推演”在思想上高度相关。

## 3. RISE 代码结构

`RISE/` 代码目录主要包括三部分：

```text
RISE/
  dynamics/dynamics_model/
  policy_and_value/policy_offline_and_value/
  policy_and_value/policy_online/
  deploy/
```

各部分作用如下：

### 3.1 dynamics/dynamics_model

该部分实现 action-conditioned dynamics model。

文档 `RISE/docs/dynamics_model.md` 中说明：

1. 数据使用 LeRobot 格式。
2. 推荐三视角输入：
   - head view
   - left wrist view
   - right wrist view
3. 图像建议预处理到 `256 × 192`。
4. 输入包括历史观测和动作 token。
5. 输出是未来视频/多视角观测。

该模块的目标是预测候选动作执行后的未来视觉状态。

### 3.2 policy_and_value/policy_offline_and_value

该部分包括 offline policy training 和 value model training。

文档 `RISE/docs/offline_learning.md` 中说明：

1. 训练数据为 LeRobot-style dataset。
2. value model 可以给每帧或每段轨迹打 value/advantage 标签。
3. `label_value.sh` 用于给数据集标注 value 和 advantage。
4. `vis_value.sh` 用于可视化 value 预测。

这对当前课题的启发是：异常恢复结果不一定只能用 success/failure 二值表示，也可以设计连续的 progress value。

### 3.3 policy_and_value/policy_online

该部分实现在线 RL/self-improvement。

文档 `RISE/docs/online_training.md` 中说明：

1. 训练中包含 env、rollout、actor 等组件。
2. dynamics model 和 reward/value model 作为在线训练环境的一部分。
3. 通过 imagined rollout 产生训练样本。
4. 使用 advantage-conditioning 更新策略。

该部分工程复杂度较高，依赖多 GPU、world model、value model 和 VLA policy，不适合当前阶段直接完整复现。

### 3.4 deploy/data_collection

该部分对当前课题最直接有用。

其中：

```text
RISE/deploy/data_collection/collect_data.py
RISE/deploy/data_collection/collect_inference_data.py
```

提供了真机数据采集范式。

`collect_data.py` 支持采集：

1. 多相机图像：
   - `cam_high`
   - `cam_left_wrist`
   - `cam_right_wrist`
2. 可选深度图。
3. 双臂关节位置 `qpos`。
4. 双臂关节速度 `qvel`。
5. effort。
6. action。
7. base_action。
8. 视频导出。

数据保存为 HDF5，主要字段包括：

```text
/observations/qpos
/observations/qvel
/observations/effort
/action
/base_action
```

同时导出各相机视频。

`collect_inference_data.py` 支持自主执行 episode 采集，并让操作者在结束后标注：

```text
1 → success
0 → failure
```

保存目录区分为：

```text
aloha_mobile_success
aloha_mobile_fail
```

这对当前课题的真实经验采集非常有参考价值。

## 4. RISE 对当前课题的直接启发

### 4.1 真机经验应分成两层

借鉴 RISE，当前课题中的真机经验不应只保存一个 JSON，而应分成两层：

#### 第一层：原始真实 episode 数据

用于保存完整执行数据，类似 RISE 的 HDF5。

```text
real_episode_raw {
  multi_view_rgb,
  optional_depth,
  qpos,
  qvel,
  effort,
  action,
  base_action,
  video,
  success_or_failure_label
}
```

这一层负责保留真实执行证据，便于后续重新分析、训练 value model 或进行视觉检索。

#### 第二层：异常恢复经验元数据

用于保存当前研究需要的经验库字段。

```text
real_anomaly_experience {
  anomaly_type,
  task_stage,
  recovery_action_signature,
  recovery_success,
  failure_reason,
  linked_sim_experience_ids,
  sim_real_gap,
  keyframes
}
```

这一层负责把原始真机数据转化为可检索、可配对、可回写的经验。

### 4.2 真机经验采集不只是记录日志

RISE 的思想说明，真实数据的价值不只是“记录发生了什么”，还可以用于：

1. 训练 dynamics model。
2. 训练 value model。
3. 给候选动作打分。
4. 判断当前动作是否比历史动作更好。

当前课题短期内不一定训练完整 world model，但应从一开始让真机经验具备未来训练 value/world model 的条件。

因此，真机经验采集时应尽量保存：

1. 多视角图像。
2. 关节状态。
3. action。
4. 成功/失败标签。
5. 关键帧。
6. 时间同步信息。

### 4.3 引入 value-style 评分思想

当前经验库中很多结果是二值的：

```text
recovery_success = true / false
task_success = true / false
```

RISE 提醒我们，异常恢复可以引入连续进度评分。

例如：

```text
value_score:
  0.0 = 明显失败
  0.3 = 有进展但未恢复
  0.6 = 恢复部分状态
  1.0 = 完成恢复
```

对于异常恢复任务，这比二值成功率更细致。

示例：

```text
抓取失败恢复：
  0.0 = 未接近目标
  0.3 = 接近目标但未闭合夹爪
  0.6 = 闭合夹爪但未稳定抬升
  1.0 = 成功重新抓取并抬升

放置偏移恢复：
  0.0 = 未重新定位目标
  0.3 = 重新抓取但未移动到目标区域
  0.6 = 移动到目标区域但放置误差较大
  1.0 = 成功放置到目标区域
```

该评分可以用于：

1. 经验库排序。
2. 真机经验质量评估。
3. 沙盒推演候选动作评分。
4. 后续训练 value model。

### 4.4 用真实失败经验校准沙盒评分

RISE 的 value model 能区分 imagined rollout 中的成功和失败。当前课题可以先采用简化方式：

```text
real failure memory
→ lower score for similar simulated / imagined recovery actions
```

例如：

```text
如果某类动作在仿真中经常成功，
但在真机经验中经常导致滑落，
则该动作在沙盒推演中的评分应下降。
```

这可以写入候选动作评分：

```text
candidate_score =
  sim_success_score
  + real_success_prior
  - real_failure_risk
  - sim_real_gap_uncertainty
  - safety_risk
```

这实际上是把 RISE 中的 value model 思想转化为经验库中的真实先验和风险惩罚。

## 5. RISE 不能直接替代当前方案的原因

RISE 对当前课题很有启发，但不适合短期完整照搬。

原因如下：

### 5.1 工程复杂度高

RISE 完整系统包含：

1. 多视角真实机器人数据。
2. LeRobot 数据格式。
3. dynamics model。
4. value model。
5. VLA policy。
6. online RL。
7. 多 GPU 训练。
8. deployment pipeline。

当前课题的重点是 Sim-Real 双源经验库和异常恢复沙盒推演，直接复现 RISE 会导致研究重心偏移。

### 5.2 研究目标不同

RISE 的目标是：

```text
通过 world model 进行 policy self-improvement
```

当前课题的目标是：

```text
通过 Sim-Real 双源异常经验库进行 recovery-time sandbox reasoning
```

两者都关注真实试错成本和执行前推演，但核心贡献不同。

### 5.3 RISE 经验不是异常恢复专用经验

RISE 采集的是任务执行 episode，标签主要是 success/failure。

当前课题需要的是异常恢复经验，必须额外标注：

1. 异常类型。
2. 异常阶段。
3. 恢复动作。
4. 失败根因。
5. 对应仿真经验。
6. Sim-Real gap。

因此，不能直接把 RISE 的 episode 数据当作最终经验库条目，需要在其上增加异常恢复元数据层。

## 6. 推荐的真机经验数据结构

结合 RISE 和当前经验库设计，建议第一版真机经验采用如下目录结构：

```text
real_experience/
  raw/
    episode_000.hdf5
    episode_001.hdf5
    videos/
      cam_high/
      cam_left_wrist/
      cam_right_wrist/
  metadata/
    real_exp_000.json
    real_exp_001.json
  keyframes/
    real_exp_000_after_anomaly.jpg
    real_exp_000_before_recovery.jpg
    real_exp_000_after_recovery.jpg
```

### 6.1 原始 HDF5 数据

借鉴 RISE，`episode_000.hdf5` 可保存：

```text
/observations/qpos
/observations/qvel
/observations/effort
/observations/images/cam_high
/observations/images/cam_left_wrist
/observations/images/cam_right_wrist
/action
/base_action
```

如果第一版实现困难，也可以先保存：

```text
/observations/qpos
/observations/qvel
/action
```

并将多视角图像保存为视频或关键帧路径。

### 6.2 异常恢复经验 JSON

每条真实经验对应一个 JSON：

```json
{
  "experience_id": "real_u3_001",
  "source": "real",
  "task_type": "pick_and_place",
  "task_stage": "grasp_recovery",
  "anomaly_type": "object_slip",
  "raw_episode_path": "raw/episode_000.hdf5",
  "recovery_action_signature": "detect->pregrasp->grasp->close->lift",
  "recovery_success": true,
  "task_success": false,
  "failure_reason": "",
  "value_score": 1.0,
  "linked_sim_experience_ids": ["sim_u3_014"],
  "sim_real_gap": {
    "pose_gap": 0.012,
    "contact_gap": 0.0,
    "outcome_gap": 0.0,
    "execution_delay_gap": 0.2
  },
  "keyframes": {
    "after_anomaly": "keyframes/real_exp_000_after_anomaly.jpg",
    "before_recovery": "keyframes/real_exp_000_before_recovery.jpg",
    "after_recovery": "keyframes/real_exp_000_after_recovery.jpg"
  }
}
```

## 7. 推荐落地路线

### 7.1 短期路线

短期不建议完整复现 RISE 的 dynamics model 和 online RL。

更合理的是：

```text
RISE-style 数据采集格式
+ 当前经验库元数据
+ MuJoCo 沙盒评分
+ Sim-Real gap 统计
```

目标是先形成真实经验接入能力。

### 7.2 中期路线

中期可以利用真实 episode 统计：

1. `real_success_prior`
2. `real_failure_risk`
3. `sim_real_gap`
4. `value_score`

这些统计量直接进入候选恢复动作评分。

例如：

```text
candidate_score =
  sim_success_score
  + real_success_prior
  - real_failure_risk
  - sim_real_gap_uncertainty
  - safety_risk
```

### 7.3 长期路线

长期可以考虑训练轻量化 value model 或 dynamics model。

但这应作为后续扩展，而不是第一阶段主目标。

可选方向：

1. 训练一个轻量 value model 预测恢复进度。
2. 训练一个视觉-动作 outcome predictor。
3. 用真实失败 episode 校准沙盒风险评分。
4. 用真实成功/失败 episode 生成 action-conditioned recovery prior。

## 8. 当前课题与 RISE 的关系表述

论文中可以这样表述与 RISE 的关系：

```text
RISE shows that real robot experience can be used to construct an
imagination space through a compositional world model, where candidate
actions are evaluated before costly physical execution. Our work shares
the motivation of reducing real-world trial-and-error, but focuses on a
different problem setting: recovery-time anomaly handling through a
Sim-Real dual-source experience library. Instead of training a full
world model for policy self-improvement, we organize simulated and real
failure/recovery episodes into a structured memory, estimate Sim-Real
gap signatures, and use them to calibrate a pre-execution sandbox for
candidate recovery selection.
```

中文表述：

```text
RISE 证明了真实机器人经验可以用于构建想象空间，
并在真实执行前评估候选动作后果。
本文与其共享“减少真机试错”的动机，
但研究对象不同：本文关注异常发生时的恢复决策，
通过 Sim-Real 双源经验库组织仿真和真机异常经验，
显式估计二者差异，并用这些差异校准执行前沙盒，
从而选择更可靠的恢复动作。
```

## 9. 一句话结论

RISE 对当前研究最重要的启发不是“直接训练一个完整世界模型”，而是：

```text
真机经验应以可训练、可回放、可评分的 episode 形式采集；
异常恢复不应只记录 success/failure，
还应记录连续进度价值、失败风险和与仿真预测之间的差异。
```

因此，当前最值得做的是：

```text
把 RISE 风格的真机 episode 数据格式接入 real_memory，
再在其上增加异常类型、恢复动作、仿真配对和 Sim-Real gap 元数据。
```
