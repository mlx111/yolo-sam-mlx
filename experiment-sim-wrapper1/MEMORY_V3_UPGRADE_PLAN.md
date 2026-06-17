# memory_v3 升级到旧版经验库功能的计划

## 1. 背景与目标

当前实验主路径使用 `memory/v3.py` 中的轻量经验库。它的核心优点是按 `condition_id` 隔离检索，适合 U1-U5 共 25 类细分异常，能够避免旧版经验库中相似但不等价异常之间的误复用。

旧版 `memory_v2` 的核心优点是 episode 信息完整：它不仅保存技能序列和结果，还保存异常状态、感知快照、虚拟重建、执行反馈、关键帧、检索键、失败归因和验证状态。它更容易体现经验库的作用，也更适合后续论文中说明“经验为何可复用”。

升级目标不是简单回退到旧版，而是形成 `memory_v3_plus`：

- 保留新版 `condition_id` 隔离，作为第一层硬约束。
- 迁回旧版的 episode 结构、验证状态、关键帧、成功证据、失败归因和检索解释。
- 失败经验继续只作为反例，不恢复旧版 `avoidance_hint` 直接给策略的做法。
- 让成功经验能向 LLM 提供“技能序列为什么成功”的物理证据，而不仅是动作列表。

## 2. 两版经验库的功能差异

| 功能 | 旧版 `memory_v2` | 当前 `memory_v3` | 升级方向 |
| --- | --- | --- | --- |
| 检索边界 | 按 `anomaly_type`、任务状态、文本和结构相似度检索 | 严格按 `condition_id` 检索 | 保留 `condition_id` 硬过滤，再在同条件内做结构相似排序 |
| 顶层 schema | episode 型，字段完整 | 轻量型，字段少 | 增加 episode 子结构，但保持 v3 兼容 |
| 成功经验 | 有感知、重建、计划、结果、关键帧、验证状态 | 主要是技能序列、结果、summary、metadata | 把成功物理证据和验证状态提升为稳定字段 |
| 失败经验 | deterministic critic，含 `critic_flags` 和 `avoidance_hint` | LLM critic，只保留根因和证据 | 保留 LLM 归因，增加规则 critic 作为可解释补充，但不生成规避建议 |
| 关键帧 | 顶层 `keyframes` 原生支持 | 放在 `metadata`，利用不足 | 迁为顶层字段，检索和 prompt 都能使用 |
| 验证信息 | `validation_status/evidence` 区分仿真、真实、失败 | 只有预留的 `validation_evidence` | 增加 `validation_status`、`validation_source`、`promotion_history` |
| 检索解释 | 有多维 similarity explanation | 只有 condition/action/success 简单解释 | 增加结构化评分明细，便于分析实验结果 |
| 文本检索 | 支持 text summary / vector index | 当前忽略旧文本向量索引 | 作为同条件内的可选二级排序，不跨 condition |
| 经验分区 | simulation/validated/real/failed | success -> simulation_memory, failure -> failed_memory | 恢复分区优先级，但不突破 condition 过滤 |

## 3. 目标 schema 设计

建议不要直接把 `memory_v2` dataclass 全量替换 `MemoryV3Entry`，否则会影响当前 U1-U5 实验路径。更稳妥的做法是在 `MemoryV3Entry` 上逐步增加旧版能力。

### 3.1 保留现有顶层字段

继续保留：

```text
schema_version
experience_id
condition_id
scenario_id
available_actions
skill_sequence
result
status
source
created_at
updated_at
summary
metadata
validation_evidence
```

这些字段已经被当前 runner、batch、analysis 脚本依赖，应保持兼容。

### 3.2 新增稳定顶层字段

建议新增：

```text
episode_type
domain
source_run_id
source_trial_id
confidence_score
memory_partition
task
perception
reconstruction_artifacts
recovery_plan
execution_feedback
key_slices
keyframes
anomaly_state
retrieval_key
failure_taxonomy
validation_status
validation_source
promotion_history
text_summary
text_embedding_id
embeddings
```

其中最优先的是：

1. `keyframes`
2. `validation_status`
3. `anomaly_state`
4. `retrieval_key`
5. `execution_feedback`
6. `failure_taxonomy`
7. `perception`
8. `reconstruction_artifacts`

这 8 个字段直接影响经验能否解释、检索和复用。

### 3.3 成功经验最小证据包

成功经验进入 prompt 前，应至少能提供：

```text
condition_id
task_stage
available_actions
skill_sequence
recovery_success
task_success
lift_from_table / apple_z_after_recovery
contact_after_close
contact_after_lift
observed_pos / perceived_position
virtual_validation_success
validation_status
keyframes
summary
```

成功经验的 prompt 文本建议改成：

```text
成功案例:
- 条件编号
- 可复用技能序列
- 成功证据: 抬升高度、夹爪接触、目标位置、恢复判据
- 验证状态: simulation_only / simulation_validated / real_validated
- 适用边界: condition_id 和 available_actions
```

这样可以避免 LLM 只机械复制动作序列，而是根据物理证据判断经验是否可信。

### 3.4 失败经验最小证据包

失败经验应包含：

```text
condition_id
failure_stage
failure_type
root_cause
failed_predicates
failure_evidence
plan_signature
failed_skill_sequence
validation_status=failed
```

明确不恢复旧字段：

```text
avoidance_hint
suggested_recovery_constraints
memory_text 中的直接恢复策略
```

如果需要 deterministic critic 的规则诊断，可以保存为：

```text
failure_taxonomy.rule_critic.flags
```

但每条 flag 只描述事实和证据，不写“下次应该怎么做”。

## 4. 检索升级方案

### 4.1 第一层硬过滤

继续使用：

```text
entry.condition_id == query.condition_id
```

这是新版经验库最重要的改进，不应退回到旧版跨 `anomaly_type` 的宽松检索。

### 4.2 第二层兼容过滤

保留当前可用技能检查：

```text
entry.skill_sequence.actions subset of current available_actions
```

并扩展为：

```text
entry.available_actions compatible with current available_actions
task_stage compatible
validation_status not invalid
```

### 4.3 第三层结构化排序

在同一 `condition_id` 内引入旧版 similarity 的核心维度：

| 维度 | 建议权重 | 说明 |
| --- | ---: | --- |
| validation_status | 0.25 | real_validated > simulation_validated > simulation_only > failed |
| result_success | 0.20 | 成功经验优先，但 hierarchical 模式保留失败反例 |
| retrieval_key | 0.20 | 位移桶、接触模式、动作签名等 |
| anomaly_state | 0.15 | 异常状态相似度 |
| task_stage | 0.10 | 同阶段优先 |
| text_summary | 0.05 | 只作为同条件内辅助排序 |
| recency/diversity | 0.05 | 防止重复经验刷屏 |

注意：文本相似度只允许在同 `condition_id` 内参与排序，不允许跨条件召回。

### 4.4 检索解释

每次 query 保存：

```text
final_score
condition_id_match
available_action_compatible
validation_score
result_success
retrieval_key_similarity
anomaly_state_similarity
task_stage_match
text_summary_similarity
diversity_penalty
```

这些字段应写入 `metrics["retrieved_memories"]`，方便后续分析 memory 是否真的被用到。

## 5. 写入与迁移方案

### 5.1 写入路径改造

主要修改点：

- `memory/v3.py`
  - 扩展 `MemoryV3Entry` dataclass。
  - 增加 `entry_to_dict` / `entry_from_dict` 的向后兼容处理。
  - 增加 `validation_status` 到 `memory_partition` 的推断。
  - 增加结构化 query scoring。

- `run_experiment_v4.py`
  - `_build_experience_entry()` 不再把关键数据只塞进 `metadata`。
  - 将 `keyframes`、`perception`、`reconstruction_artifacts`、`recovery_plan`、`execution_feedback`、`anomaly_state`、`retrieval_key`、`validation_status` 写到顶层。
  - 根据 `recovery_success`、`task_success`、`virtual_validation_success` 推断 `validation_status`。

- `experiment_method_runner.py`
  - 保存失败经验时，将 LLM critic 结果写入顶层 `failure_taxonomy`。
  - 检索指标增加新的 score explanation 字段。

- `llm_handler.py`
  - 成功经验 prompt 增加成功证据、验证状态、关键帧说明。
  - 失败经验 prompt 保持反例形式，不加入规避建议。

### 5.2 旧数据迁移

增加工具：

```text
tools/upgrade_memory_v3_plus.py
```

迁移逻辑：

1. 读取旧 `memory_v3` JSON。
2. 对每条 entry 保留原字段。
3. 从 `metadata` 中提升字段：
   - `metadata.keyframes` -> `keyframes`
   - `metadata.key_slices` -> `key_slices`
   - `metadata.perception_before/after` -> `perception`
   - `metadata.reconstruction_artifacts` -> `reconstruction_artifacts`
   - `metadata.recovery_plan` -> `recovery_plan`
   - `metadata.contact_after_close/lift` -> `execution_feedback`
   - `metadata.failure_taxonomy` -> `failure_taxonomy`
4. 根据已有 metrics 推断：
   - `validation_status`
   - `memory_partition`
   - `anomaly_state`
   - `retrieval_key`
   - `text_summary`
5. 输出新文件，不覆盖原始文件：
   - `memory_v3_plus.json`
   - `memory_v3_plus_migration_report.md`

### 5.3 schema 版本

推荐使用：

```text
schema_version = "memory_v3_plus"
```

加载逻辑兼容：

- `memory_v3`：按旧字段加载，缺失字段填默认值。
- `memory_v3_plus`：完整加载。
- `memory_v2`：可选支持转换，但不作为当前主路径。

## 6. 实施阶段

### 阶段 1：schema 扩展，不改变行为

目标：先让新字段能读写，但 query 和 prompt 行为保持不变。

任务：

- 扩展 `MemoryV3Entry`。
- 更新序列化/反序列化。
- `_build_experience_entry()` 写入顶层 episode 字段。
- 增加单元/脚本检查：旧 `memory_v3` 文件仍能加载。

验收：

- 原 U3 batch 能继续跑。
- 旧 rolling memory 文件能 load/save。
- 新文件中出现顶层 `keyframes`、`validation_status`、`retrieval_key`。

### 阶段 2：成功经验增强

目标：让成功经验真正携带可解释证据。

任务：

- 从 metrics 中生成 `execution_feedback`。
- 构造 `success_evidence` 或统一放入 `validation_evidence`。
- 修改 `llm_handler.py`，成功案例 prompt 增加物理证据和验证状态。
- 在分析脚本中输出成功证据覆盖率。

验收：

- 每条成功经验至少有一个物理证据字段。
- prompt 中能看到成功证据，而不仅是技能序列。
- `sim_memory_weak` 的冷启动后表现可解释。

### 阶段 3：检索排序升级

目标：保留 condition 隔离，同时恢复旧版结构相似排序能力。

任务：

- 在 `MemoryV3Library.query()` 中增加结构化评分。
- 增加 `get_last_score_explanation()` 的完整解释。
- 支持同条件内 text summary 可选排序。
- 增加 diversity，避免 top_k 全是相同动作签名。

验收：

- 不同 `condition_id` 之间不会互相召回。
- 同一 `condition_id` 内，validated/success/结构相似经验排序靠前。
- `metrics["retrieved_memories"]` 能解释每条经验为什么被选中。

### 阶段 4：失败经验增强

目标：恢复旧版失败诊断的可解释性，但不恢复“教答案”的字段。

任务：

- 保留 LLM critic 作为主失败归因。
- 增加 deterministic rule critic，只输出事实 flags。
- 去除或禁止生成 `avoidance_hint`、`suggested_recovery_constraints`。
- prompt 中继续显示“不要复现的技能序列”和失败证据。

验收：

- 失败经验包含 `failure_stage/type/root_cause/evidence/plan_signature`。
- prompt 中没有“下一次应该如何做”的策略文本。
- failed memory blocker 可以识别重复失败动作签名。

### 阶段 5：验证状态与经验晋升

目标：恢复旧版 simulation/validated/real/failed 分区能力。

任务：

- 定义 `validation_status` 状态机：
  - `simulation_only`
  - `simulation_validated`
  - `real_executed`
  - `real_validated`
  - `failed`
- 将虚拟验证结果写入 `validation_evidence`。
- 更新 `promote_experience_library.py` / `merge_memory_growth_experiences.py`，兼容 `memory_v3_plus`。
- query 排序按验证等级加权，但仍受 `condition_id` 约束。

验收：

- 经验能从 simulation_only 晋升为 simulation_validated。
- 失败经验明确进入 failed partition。
- 分析报告能统计各验证状态下的成功率。

### 阶段 6：实验验证

目标：证明升级不是只增加字段，而是改善经验库可用性和可解释性。

建议实验：

1. U3 原配置复跑：
   - `direct_llm_weak`
   - `sim_only_weak`
   - `sim_memory_weak`
   - `hierarchical_memory_weak`

2. 对比三种 memory：
   - 空 `memory_v3`
   - 原 rolling `memory_v3`
   - 种子 + 结构化 `memory_v3_plus`

3. 指标：
   - recovery_success
   - task_success
   - retrieved_positive_count
   - retrieved_failed_count
   - useful_memory_ratio
   - memory_action_overlap_mean
   - failed_plan_blocker_matches
   - prompt_success_evidence_coverage
   - validation_status_distribution

4. 关键分析：
   - 成功经验是否真的被召回。
   - 召回经验是否来自同一 `condition_id`。
   - LLM 是否利用了成功物理证据。
   - failed memory 是否减少重复失败动作签名。

## 7. 推荐优先级

最高优先级：

1. 顶层 `keyframes`、`validation_status`、`retrieval_key`、`failure_taxonomy`。
2. 成功经验 prompt 增加物理证据。
3. `condition_id` 内结构化检索排序。
4. 迁移工具，保证已有 rolling memory 可升级。

中等优先级：

1. text summary 向量索引。
2. validated/real memory 分区晋升。
3. diversity rerank。
4. deterministic rule critic 补充。

低优先级：

1. 完整复刻旧版 `ExperienceLibrary` 的所有字段。
2. 跨 `condition_id` 检索。
3. 让失败经验输出直接策略建议。

跨 `condition_id` 检索和失败经验策略建议都不建议恢复，因为它们会削弱当前 25 类异常设计的实验控制性。

## 8. 风险与控制

| 风险 | 影响 | 控制方式 |
| --- | --- | --- |
| schema 扩展破坏旧 JSON 加载 | rolling memory 无法复用 | 所有新增字段必须有默认值，迁移工具不覆盖原文件 |
| 检索过宽导致跨异常混用 | 实验结论不干净 | `condition_id` 永远作为第一层硬过滤 |
| 失败经验再次教答案 | hierarchical 方法不公平 | 禁止 `avoidance_hint` 和恢复约束进入 prompt |
| prompt 过长 | LLM 计划质量下降 | 成功证据只取 top_k，关键帧只传路径和阶段摘要 |
| 字段很多但不被使用 | 复杂度上升无收益 | 每个新增字段都要进入 query、prompt、analysis 至少一个环节 |
| 新旧工具不兼容 | 分析脚本断裂 | 先兼容 `memory_v3`，再引入 `memory_v3_plus` |

## 9. 最小可行版本

如果只做一版最小升级，建议范围如下：

1. `MemoryV3Entry` 新增：
   - `keyframes`
   - `validation_status`
   - `execution_feedback`
   - `anomaly_state`
   - `retrieval_key`
   - `failure_taxonomy`

2. `_build_experience_entry()` 顶层写入：
   - keyframes
   - contact_after_close
   - contact_after_lift
   - observed_pos
   - recovery_success_criteria
   - task_success_criteria
   - virtual_validation_success

3. `MemoryV3Library.query()` 改为：
   - hard filter: `condition_id`
   - filter: available action compatible
   - score: success + validation + retrieval_key + action coverage

4. `llm_handler.py` prompt 改为：
   - 成功经验显示技能序列和成功证据。
   - 失败经验显示失败证据和不要复现的动作签名。

5. 增加迁移工具：
   - 从 `metadata` 提升旧字段。
   - 生成迁移报告。

这个版本能最快补上当前 `memory_v3` 的主要短板，同时不会破坏 U1-U5 条件隔离实验设计。
