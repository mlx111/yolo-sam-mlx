# 仿真经验写回规则说明

这份说明只回答一个问题：**什么样的仿真经验会进入经验库**。

## 结论

不是所有 `candidate sandbox rollout` 都会写回。

当前经验库采用的是**上层触发写回 + write policy 过滤**的方式：

```text
MuJoCo 仿真执行
-> R1ProMujocoAdapter.normalize_episode
-> ExperienceEntry
-> write_policy
-> ExperienceLibrary.add_with_policy
-> universal_experience_library.json
```

其中，`candidate sandbox rollout` 默认只负责产出 rollout 结果、critic 结果和评分，不自动写库。

## 会写入的仿真经验

### 1. 失败经验

只要仿真里出现明确失败，通常都会保留，例如：

- `success = false`
- `recovery_success = false`
- 有明确失败原因，如抓取失败、放置失败、滑落、碰撞等

这些经验会被视为风险记忆，而不是噪声。

### 2. critic 判定为 block 的经验

如果 critic 认为该经验有明显风险，或者技能序列不安全，会优先保留。

这类经验通常用于：

- 风险先验
- 失败检索
- 后续候选重写

### 3. 带有可行动失败类型的经验

例如已经归类的失败类型：

- `grasp_miss`
- `grasp_slip`
- `object_not_lifted`
- `place_occupied`
- `transport_collision`

这类经验可用于后续规避，因此通常会写入。

### 4. 带有 sim-real gap 的经验

如果仿真和真实/伪真实经验之间存在差异，且该差异被编码为 gap 证据，也会保留。

这类经验的价值在于：

- 校准 sandbox
- 修正初始状态
- 提供 sim-real 偏差先验

### 5. 有价值的成功经验

不是所有成功都会写，但满足以下条件的成功经验通常可写：

- 异常条件下仍成功恢复
- 选择了替代放置位
- 通过了仿真验证
- 具有较高 `write_score`

这类经验用于增强后续检索和排序。

## 可能不新增条目的仿真经验

### 1. 重复的低风险成功经验

如果一条经验和已有经验高度重复，系统可能不新增条目，而是合并支持次数。

### 2. 低价值普通成功经验

如果经验可合并且 `write_score` 太低，会被跳过。

### 3. 字段不完整的经验

缺少关键字段时会被拒绝，例如：

- `scenario_id`
- `condition_id`
- `robot_type`
- `backend`
- `skill_sequence`
- `result`
- `object_class`

## 为什么这样设计

因为 sandbox 的任务是**筛选候选**，不是把每次 rollout 都当成长期记忆。

如果每个候选都自动写回，经验库会很快被：

- 临时候选污染
- 重复成功淹没
- 低质量失败占满

所以当前分成两层：

- `candidate sandbox rollout`：评估层
- `write_policy`：记忆层

## 目前最重要的边界

当前可以明确说的是：

- 仿真经验可以写回经验库
- 但不是“跑过 sandbox 就自动写回”
- 真正写回依赖 `write_policy` 和上层脚本显式调用

## 以后更合理的三档策略

后面如果继续收紧规则，建议分成三档：

```text
1. selected_plan_execution -> 默认写回
2. critic_block / high_risk_failure -> 可选写回为风险记忆
3. ordinary_unselected_candidate -> 默认不写
```

这会更符合论文和实验流程。
