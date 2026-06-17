> 状态说明：本文是旧缺口分析与升级路线，包含 UR5e/experiment-sim-wrapper
> 历史背景。当前 R1Pro 经验库实现已经迁移到 `experience_system`。
> 已实现与剩余事项请以 `experience_system/docs/remaining_experience_system_roadmap.md`
> 和 `experience_system/docs/sandbox_fidelity_optimization_roadmap.md` 为准。

# Sim-Real 双源经验库缺口分析与升级路线

## 1. 当前定位

如果最终研究目标是 **Sim-Real 双源经验库**，那么当前已有的结构化正负经验库可以作为一个重要基础模块，但还不足以单独支撑最终创新点。

当前系统已经具备：

1. MuJoCo UR5e 仿真异常恢复实验框架。
2. U1-U5 异常 benchmark。
3. 成功经验与失败经验分区。
4. 结构化 episode memory。
5. rolling memory。
6. sim wrapper 执行前验证。
7. failed memory blocker / rewrite。
8. keyframe、text summary、retrieval key、failure taxonomy 等经验字段。

这些内容可以作为最终系统中的 **仿真侧结构化经验库模块**。

但如果论文主线是 Sim-Real 双源经验库，当前还缺少：

```text
真实经验接入
Sim-Real 差异签名
仿真经验与真机经验配对
真机经验校准沙盒
真机执行后回写闭环
```

因此，当前内容更适合放在最终研究中的一个子部分，而不是作为最终主创新本身。

## 2. 当前已有内容适合作为什么

当前已有系统可以作为：

**仿真侧异常经验生成与管理模块。**

它的作用包括：

1. 批量生成异常场景。
2. 记录成功恢复经验。
3. 记录失败恢复经验。
4. 提供结构化检索。
5. 提供失败经验约束。
6. 为后续真机实验提供候选恢复策略。
7. 为 Sim-Real 双源经验库提供仿真经验源。

也就是说，现有系统不是最终目标的全部，而是最终目标中的 **Simulation Memory Branch**。

最终系统可以表达为：

```text
Simulation Memory Branch:
  当前已有结构化正负经验库

Real Memory Branch:
  待新增真机异常经验库

Sim-Real Gap Branch:
  待新增仿真-真机差异签名与校准机制

Sandbox Reasoning Branch:
  当前已有 sim wrapper，需要升级为被真机经验校准的沙盒推演模块
```

## 3. 缺口一：真机经验数据

这是当前最大缺口。

当前代码和文档中已经预留了：

```text
real_memory
real_validated
real_executed
```

但这些目前主要是接口预留，还没有真实机器人执行数据。

如果要支撑 Sim-Real 双源经验库，至少需要采集少量真机经验。每条真机经验应包含：

1. 真机异常状态。
2. 真机恢复动作。
3. 真机执行结果。
4. 真机失败原因。
5. 真机关键帧。
6. 真机传感器或控制反馈。
7. 与对应仿真经验的关联关系。

最小版本不需要大量真机数据。可以先采集：

```text
2 类异常 × 每类 5-10 条真机经验
```

优先选择：

1. 抓取失败 / 抓取不稳。
2. 放置偏移 / 运输掉落。

只要有少量真实经验，就可以把系统从“仿真经验库”推进到“Sim-Real 双源经验库”的初始版本。

## 4. 缺口二：Sim-Real 差异签名

双源经验库的关键不是简单地同时保存仿真经验和真机经验，而是显式保存二者之间的差异。

建议新增字段：

```text
sim_real_gap {
  pose_gap,
  contact_gap,
  outcome_gap,
  execution_delay_gap,
  perception_gap,
  recovery_success_gap
}
```

各字段含义：

1. `pose_gap`：仿真预测物体/末端位姿与真机观测位姿之间的差异。
2. `contact_gap`：仿真接触状态与真机接触状态之间的差异。
3. `outcome_gap`：仿真成功但真机失败，或仿真失败但真机成功。
4. `execution_delay_gap`：控制执行延迟、夹爪响应延迟等差异。
5. `perception_gap`：真机感知位置与仿真/标定位置之间的差异。
6. `recovery_success_gap`：同类恢复策略在仿真和真机中的成功率差异。

示例：

```text
仿真中抓取稳定，但真机中发生滑落 → contact_gap 高
仿真中无碰撞，但真机中碰撞 → outcome_gap 高
仿真中位姿准确，但真机视觉偏差 3cm → perception_gap 高
仿真恢复成功率 90%，真机成功率 50% → recovery_success_gap 高
```

这部分是区别于普通经验库的关键创新点。

## 5. 缺口三：仿真经验与真机经验配对机制

如果只分别保存 `simulation_memory` 和 `real_memory`，它们仍然只是两个分区，不是真正的双源弥合。

系统需要建立仿真经验与真机经验之间的配对关系。

建议新增关系字段：

```text
sim_real_pair {
  sim_experience_id,
  real_experience_id,
  paired_by,
  pair_score,
  gap_score,
  validation_status
}
```

其中：

1. `sim_experience_id`：对应的仿真经验。
2. `real_experience_id`：对应的真机经验。
3. `paired_by`：配对依据，例如 condition、object、action_signature、state_similarity。
4. `pair_score`：仿真经验与真机经验的相似度。
5. `gap_score`：二者差异大小。
6. `validation_status`：该仿真经验是否被真机验证、修正或否定。

系统需要回答：

1. 当前真机异常，对应历史上哪些仿真异常？
2. 哪些仿真经验被真机验证过？
3. 哪些仿真恢复策略在真机上失效？
4. 哪些仿真经验需要降低可信度？
5. 哪些真机失败可以反向修正仿真沙盒？

## 6. 缺口四：沙盒校准机制

当前 `sim_wrapper` 更像执行前验证器，还不是被真机经验校准的沙盒。

最终目标需要加入：

```text
real experience → estimate gap → adjust sandbox
```

第一版可以采用工程化校准，而不必一开始做复杂残差动力学。

可实现的简单校准包括：

1. 真机抓取经常滑落 → 提高滑落风险评分。
2. 真机位姿偏差平均 2cm → 沙盒初始化加入 pose bias。
3. 真机夹爪闭合延迟 → 沙盒执行加入 delay。
4. 真机摩擦更低 → 降低 friction 或增加 slip probability。
5. 真机放置偏移稳定朝某方向 → 对目标放置位姿加入修正偏置。

建议第一阶段只做三类校准：

```text
object_pose_bias
gripper_delay_bias
slip_risk_bias
```

这样可以形成最小闭环：

```text
真机经验告诉沙盒哪里不可信
沙盒根据真机经验修正推演
修正后的沙盒再用于恢复动作选择
```

## 7. 缺口五：候选恢复策略排序需要使用真机经验

当前 hierarchical memory 主要通过 prompt 影响 LLM 规划，并通过 failed memory blocker 避免重复失败。

最终目标需要让真机经验直接影响候选恢复策略排序。

建议评分函数升级为：

```text
candidate_score =
  sim_success_score
  + real_success_prior
  - failure_memory_risk
  - sim_real_gap_uncertainty
  - safety_risk
```

其中：

1. `sim_success_score`：候选策略在仿真沙盒中的成功率。
2. `real_success_prior`：相似真机经验中该策略的成功先验。
3. `failure_memory_risk`：相似失败经验带来的风险。
4. `sim_real_gap_uncertainty`：该类动作在仿真和真机之间是否经常不一致。
5. `safety_risk`：碰撞、关节越界、夹爪风险等。

核心原则：

```text
仿真中成功但真机历史上经常失败的策略，不能排第一。
```

这会让系统从“仿真验证有效”升级为“真机经验校正后的仿真验证有效”。

## 8. 缺口六：真机执行后的回写闭环

每次真机执行恢复动作后，系统需要回写经验库。

回写内容包括：

1. 新增一条 `real_memory`。
2. 更新对应 `simulation_memory` 或 `validated_memory` 的可信度。
3. 更新 `sim_real_gap`。
4. 更新恢复动作在真机中的成功率。
5. 如果真机失败，写入 `failed_memory`。
6. 如果真机成功，写入或提升对应成功经验。

闭环应表达为：

```text
sim experience
→ sandbox test
→ real execution
→ gap update
→ better sandbox
→ better future recovery
```

这部分是最终系统区别于普通仿真验证和普通经验库的关键。

## 9. 最小真实验证实验设计

如果希望把现有内容作为小点，最终实验可以这样设计。

### 9.1 实验任务

优先选择长时程 pick-and-place 中的两类异常：

1. 抓取失败 / 抓取不稳。
2. 放置偏移 / 运输掉落。

原因：

1. 这两类异常能体现接触、感知和执行误差。
2. 与当前 U3/U4 仿真异常体系衔接较好。
3. 不需要一开始覆盖所有 U1-U5。

### 9.2 对比方法

建议设置五组：

| 方法 | 说明 |
| --- | --- |
| Direct | 真机异常后直接恢复 |
| Sim memory only | 只用仿真经验推演 |
| Real memory only | 只用真机历史经验 |
| Sim + Real memory | 双源经验，但不显式建模 gap |
| Sim + Real + Gap | 双源经验 + 差异校正 + 沙盒推演 |

真正要证明的是：

```text
Sim + Real + Gap > Sim memory only
Sim + Real + Gap > Sim + Real memory
```

也就是说，不只是“双源经验更多”，而是“差异建模与沙盒校准确实有用”。

### 9.3 指标

建议使用：

1. 真机异常恢复成功率。
2. 真机试错次数。
3. 平均恢复时间。
4. 碰撞或滑落次数。
5. 沙盒预测与真机结果一致性。
6. 仿真经验迁移到真机的有效率。
7. Sim-Real gap 更新前后的预测误差变化。

其中最关键的是：

```text
沙盒预测与真机结果一致性
```

因为这是证明 Sim-Real 差异经验有效的直接指标。

## 10. 论文贡献重新分层

建议把最终论文贡献分成主贡献和子贡献。

### 10.1 主贡献

#### 贡献一：Sim-Real 双源异常经验表示

提出一种面向机器人异常恢复的 Sim-Real 双源异常经验表示方法，统一存储仿真经验、真机经验和 Sim-Real 差异签名。

#### 贡献二：基于差异校正的沙盒推演机制

利用历史真机经验修正仿真沙盒，使真机在执行恢复动作前能够进行更接近真实物理结果的仿真推演。

#### 贡献三：双源经验闭环更新机制

真机执行结果回写经验库，持续更新仿真经验可信度、真机成功先验和 Sim-Real 差异签名。

### 10.2 子贡献

#### 子贡献：结构化正负经验库

当前已有的成功经验、失败经验、检索、blocker、rewrite、U1-U5 benchmark 可以作为仿真侧基础模块。

其作用是：

1. 提供仿真异常经验源。
2. 提供成功/失败经验组织方式。
3. 提供候选恢复策略。
4. 提供失败风险约束。
5. 为双源经验库提供 simulation branch。

这样现有工作不会浪费，但不会承担最终主创新的全部压力。

## 11. 推荐最终系统表述

可以将最终系统描述为：

```text
本文提出一种面向机器人异常恢复的 Sim-Real 双源经验库。
该系统首先在仿真环境中构建结构化正负异常经验库，
用于积累成功恢复模式和失败风险模式；
随后接入少量真机异常执行经验，
将真机恢复结果与相似仿真经验配对，
显式估计 Sim-Real 差异签名；
在真机异常发生时，系统检索仿真经验、真机经验和差异经验，
校准执行前仿真沙盒，
并在沙盒中推演候选恢复策略，
最终选择更可靠的恢复动作执行到真机。
真机执行结果再回写经验库，
形成持续缩小 Sim-to-Real Gap 的闭环。
```

## 12. 一句话结论

如果最终目标是 Sim-Real 双源经验库，那么当前系统已经完成了重要的仿真侧基础，但还缺少最终创新所必需的四个核心组件：

```text
真实经验接入
Sim-Real 差异签名
真机经验校准沙盒
真机执行回写闭环
```

只要补上这四件事，当前结构化正负经验库就可以自然升级为 Sim-Real 双源经验库，而不是停留在“仿真异常恢复经验库”的层面。
