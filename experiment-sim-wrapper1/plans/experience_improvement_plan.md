# 经验库改进计划

基于文献调研与现有代码实际数据分析，归纳改进方向。

---

## 当前架构快照

| 维度 | 当前状态 |
|---|---|
| 经验存储 | `MemoryV3Library`，JSON 文件，condition-isolated |
| 检索方式 | 纯文本匹配（`dict_similarity` + `token_jaccard`） |
| 经验分级 | 单层 `rolling_memory`，按 scenario/cell 范围共享 |
| 失败分类 | LLM critic + deterministic rule critic，5 种粗粒度 stage |
| 检索多样性 | `diversity_lambda=0.0`，MMR 禁用 |
| 视觉利用 | keyframe 图像已采集并存入经验，检索时可选传给 LLM 看（`--use-memory-keyframes`） |
| 经验抽象 | 原始 plan 直接入库，无泛化 |

---

## 先决问题: 基础质量修复

在进入任何高级改进前，现有数据有两个明显问题。

### Q1: 失败经验描述几乎无用

**实际数据:** 5 条 U1-2 失败经验，`failure_type` 全是 `incomplete_recovery_steps`，`root_cause` 一模一样。本质是废话——LLM 看到 5 条相同的废话，没有获得任何新信息。

**原因:**
1. critic 的输入 metrics 缺失 injection 参数（不知道 dx/dy 偏移），所以无法识别真正根因
2. `recovery_steps` 为空或极短时 → critic 只能抱怨"步骤太少"
3. 所有 trial 的异常现象相同（z_change≈0.00027），critic 无法区分

**改动:**
- `experiment_method_runner.py::save_experience()`：把 `condition_spec.params`（含 injection 参数）传给 critic 输入
- `llm_handler.py::critique_failure_experience()`：prompt 中增加 injection 参数上下文，让 critic 能分析"因 dx/dy 偏移导致目标位置偏移，当前 attach_max_distance 下无法抓取"这类真实根因
- 强制 critic 输出差异化描述：如果输入 metrics 跟上一条完全相同，至少标注为 `duplicate_of=<id>`

### Q2: Keyframe 数量不足

**当前 `used_for_retrieval=True` 的 keyframe：**
1. `after_anomaly` — 感知管线检测图（YOLO+SAM，对 U1-2 可能是模糊/遮挡的）
2. `before_recovery` — 渲染的规划前场景图

两张基本是同一场景（异常后 apple 在桌上），LLM 看到的信息量有限。

**增加的 keyframe：**
| stage | 时机 | 内容 |
|---|---|---|
| `after_injection` | 异常注入后（grasp_miss 松开后） | apple 掉落后的位置 |
| `before_grasp` | 恢复方案执行前，夹爪移到 pregrasp 后 | 夹爪与 apple 的相对位姿 |
| `after_grasp_close` | 夹爪闭合后 | 是否抓住了 apple |
| `after_lift` | 提拉动作后 | apple 是否被提起 |

**此外:** 对于 U1-2/U1-5，当前有特殊逻辑（line 1851-1854）跳过发送图像给 LLM，原因是"VLM 看到的图像模糊→误判 apple 不可抓取"。这在启用 `--use-memory-keyframes` 时也需要处理——传给 LLM 的 keyframe 应该是**清晰渲染图**而不是感知图。

---

## 方向 A: 视觉经验

### A1: 启用经验配图（已有基础设施）

**描述:** `--use-memory-keyframes` 开关已实现。检索到相似经验后，将其 keyframe 图像作为 VLM 上下文传给 `plan_recovery()`。

**问题:** 当前 Q2 的 keyframe 数量不足，加上 U1-2/U1-5 的图像跳过逻辑，即使打开效果也有限。

**前置依赖:** Q2 修复。

### A2: 视觉特征检索（新增）

**描述:** 引入图像 embedding（CLIP/DINOv2），使检索时能按**当前场景图像与历史经验图像的视觉相似度**召回，而不只靠文本匹配。

**改动:**
- 新增 `memory/vision_index.py`：CLIP/DINOv2 embedding + FAISS 索引
- 修改 `memory/v3.py::MemoryV3Library`：`save/load` 同步读写 FAISS 索引；`query()` 增加 `vision_weight` 参数
- 修改 `experiment_method_runner.py::_query_experiences_for_policy()`：传入当前 keyframe 图像做视觉检索

**前置依赖:** GPU 或接受 CPU CLIP 推理（慢 5-10x）。

**预估工作量:** 3-4 天。

---

## 方向 B: 短/长期记忆分层

**描述:** 当前 rolling memory 单层平铺。ExpTeach：STM（短期，快适应） + LTM（长期，跨批次持久化）。

**改动:**
- `run_experiment_batch.py`：增加 `--ltm-path` 参数
- `experiment_method_runner.py`：
  - `_query_experiences_for_policy()`：STM top_k=5 + LTM top_k=3 合并
  - `save_experience()`：写入 STM + 异步写 LTM（`confidence_score >= 0.7`）
- 新增 `memory/ltm_store.py`：LTM 持久化，去重合并

| 维度 | STM（现有 rolling memory） | LTM（新增） |
|---|---|---|
| 作用域 | 单批次内 | 跨所有批次 |
| 容量 | 50-100 条 | 无上限（按置信度裁剪） |
| 写入时机 | 每次 trial 结束 | 批次结束后 |
| 过滤 | 全部保留 | `confidence_score >= 0.7` |

**预估工作量:** 2-3 天。

---

## 方向 C: 检索多样性 & 预过滤（最小改动，最高性价比）

**描述:** 现在 `diversity_lambda=0.0`，MMR 禁用。加上同类失败经验去重，防止检索结果被同一类失败淹没。

**改动:**
- `ur5e/experiment_config.py`：`DEFAULT_DIVERSITY_LAMBDA = 0.0` → `0.3`
- `memory/v3.py::MemoryV3Library.query()`：增加 `critic_prefilter` 参数
  - 去重：相同 `failure_type` 的失败经验只保留最新 2 条
  - 跨 condition 平衡：至少有 1 条不同 `condition_id` 的经验
- `memory/v3.py::_mmr_select()`：相似度从 action set Jaccard 改为带参数的细粒度签名

**预估工作量:** 0.5-1 天。

---

## 方向 D: 失败分类细化

**描述:** 当前 `failure_stage` 仅 5 种粗粒度值。MEMO 用聚类把零散修正泛化。

**改动:**
- `llm_handler.py::critique_failure_experience()`：扩展输出 `recovery_gap`、`suggested_fix`；输入中增加 `condition_spec.params`（injection 参数），让 critic 能关联感知偏移量
- 新增 `memory/failure_cluster.py`：text embedding 聚类，写回 `failure_taxonomy.cluster_id`
- `experiment_method_runner.py::_mark_failed_plan_blocker()`：按 cluster 去重

**预估工作量:** 2-3 天。

---

## 方向 E: 经验抽象为技能

**描述:** 把原始 `skill_sequence` 蒸馏成参数化通用技能模板。LRLL：wake 积累 → sleep 蒸馏。

**改动:**
- 新增 `memory/skill_abstraction.py`：LLM 驱动蒸馏，同类 > 5 条时触发
- `memory/v3.py`：`MemoryV3Entry` 增加 `abstracted_skill` 字段
- `experiment_method_runner.py`：优先使用抽象技能，回退原始经验

**预估工作量:** 4-5 天（高风险）。

---

## 优先级总表

| 优先级 | 方向 | 收益预期 | 工作量 | 依赖 |
|---|---|---|---|---|
| **P0** | **Q1: 修复失败描述** | 高（critic 输出目前基本无用） | 1 天 | 无 |
| **P0** | **Q2: 增加 keyframe** | 中-高（目前仅 2 张冗余图） | 1 天 | 无 |
| **P0** | **C: 检索多样性** | 中（防同类失败淹没） | 0.5-1 天 | Q1 |
| P1 | D: 失败分类细化 | 中（需 Q1 先完成） | 2-3 天 | Q1 |
| P1 | A1: 启用 keyframe | 视场景 | 0（已有） | Q2 |
| P2 | B: 短/长期分层 | 高（防跨 batch 遗忘） | 2-3 天 | Q1+Q2 |
| P2 | A2: 视觉特征检索 | 高（论文收益大） | 3-4 天 | GPU，Q2 |
| P3 | E: 经验抽象技能 | 待验证 | 4-5 天 | D |

## 推荐实施路径

```
第 1 步: Q1 + Q2 (1-2 天)
   → 基础质量修复，让经验描述有用、关键帧够多

第 2 步: C + A1 (1 天)
   → 检索多样性的同时跑一轮 A1 看效果

第 3 步: D + B (4-5 天)
   → 分类细化 + 分层架构。如果 U1-2/U1-5 的结果还不理想再考虑 A2

第 4 步: A2 (3-4 天) 或 E (评估)
   → 视实际实验效果决定
```
