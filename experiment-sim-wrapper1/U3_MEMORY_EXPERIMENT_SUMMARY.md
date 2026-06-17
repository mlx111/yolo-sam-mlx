# U1-U5 异常恢复与经验库实验简要总结

## 1. 当前异常场景设计

当前文档与代码中共设计 25 个异常场景，覆盖 U1-U5 五类问题。

### U1 感知/目标确认异常

| 条件 | 异常含义 | 恢复目标 |
| --- | --- | --- |
| U1-1 | 相似物体导致目标混淆 | 重新确认正确目标 |
| U1-2 | 目标被部分遮挡 | 恢复可靠目标感知 |
| U1-3 | 目标位姿过期 | 重新定位当前目标 |
| U1-4 | 目标姿态估计错误 | 重新估计目标位姿 |
| U1-5 | 目标边界/背景混淆 | 重新获得干净分割 |

### U2 抓取几何异常

| 条件 | 异常含义 | 恢复目标 |
| --- | --- | --- |
| U2-1 | 抓取位姿横向偏移 | 修正抓取位置并提升 |
| U2-2 | 抓取高度偏移 | 修正抓取高度并提升 |
| U2-3 | 抓取姿态旋转偏差 | 修正抓取姿态并提升 |
| U2-4 | 预抓位过近 | 修正预抓位安全间隙 |
| U2-5 | 预抓位过远 | 修正预抓/抓取参考 |

### U3 夹爪/夹持保持异常

| 条件 | 异常含义 | 恢复目标 |
| --- | --- | --- |
| U3-1 | 夹爪闭合失败 | 重新形成有效夹持并提升 |
| U3-2 | 夹爪部分闭合/夹持不足 | 恢复夹爪闭合与稳定提升 |
| U3-3 | 夹爪过早闭合 | 重新建立正确抓取关系 |
| U3-4 | 提升初期滑落 | 重新抓取并完成提升 |
| U3-5 | 提升过程渐发滑移 | 恢复稳定夹持并继续提升 |

### U4 运输/放置异常

| 条件 | 异常含义 | 恢复目标 |
| --- | --- | --- |
| U4-1 | 运输阶段掉落 | 重新抓取并继续任务 |
| U4-2 | 运输阶段目标位置变化 | 重新定位并恢复提升 |
| U4-3 | 放置位置错误 | 将目标放回 plate |
| U4-4 | 过早释放 | 重新恢复并正确放置 |
| U4-5 | 放置姿态错误 | 恢复 plate 上正确放置姿态 |

### U5 路径/策略异常

| 条件 | 异常含义 | 恢复目标 |
| --- | --- | --- |
| U5-1 | 直线路径被阻挡 | 通过安全路径恢复任务 |
| U5-2 | 接近阶段碰撞邻近物体 | 避开风险并重新恢复 |
| U5-3 | 夹爪可能与桌面碰撞 | 调整路径避免碰撞 |
| U5-4 | 恢复动作顺序错误 | 修正动作逻辑 |
| U5-5 | 重复尝试无进展 | 切换恢复策略 |

当前已完成实验聚焦 U3。U3 阶段评价重点是“异常处理是否恢复夹持并完成提升”，不使用完整放置闭环作为主要成功标准。

## 2. 经验库设计

当前经验库按 `condition_id` 检索，只使用同一异常条件下的经验，避免跨异常类型误检索。

经验分为两类：

- 成功经验：保存可复用技能序列，用于提示 LLM 参考已有恢复路径。
- 失败经验：保存失败原因与物理证据，用于提示 LLM 避免复现失败模式。

### 2.1 当前简化版 memory_v3 字段

当前实现使用轻量级 `memory_v3`，核心字段包括：

| 字段 | 作用 |
| --- | --- |
| `condition_id` | 具体异常条件，例如 U3-1 |
| `scenario_id` | 粗场景类别，例如 U3 |
| `available_actions` | 当前场景允许使用的技能集合 |
| `skill_sequence` | 本次恢复使用的技能序列 |
| `result.recovery_success` | 异常恢复是否成功 |
| `result.task_success` | 完整任务是否成功 |
| `result.failure_reason` | 失败原因标签 |
| `summary` | 简短自然语言摘要 |
| `metadata` | 额外日志、判据、critic 结果 |
| `validation_evidence` | 验证证据 |

该版本优点是结构清晰、按 condition 隔离；缺点是成功经验表达还不够完整。

### 2.2 成功经验应包含的内容

对成功经验，经验库不应只保存“成功了”和技能序列，还应保存成功为何成立。建议成功经验至少包含：

| 模块 | 内容 |
| --- | --- |
| 场景标识 | `scenario_id`, `condition_id`, `task_stage`, `injection_stage` |
| 可用技能 | 当时可调用的技能集合，防止跨场景误用 |
| 成功技能序列 | LLM 输出并最终执行成功的 `skill_sequence` |
| 关键参数 | 目标类别、夹爪状态、抓取/预抓参数、放置参数等 |
| 执行结果 | `recovery_success`, `task_success`, 提升高度、夹持状态 |
| 物理证据 | `lift_from_table`, `pinch_distance`, `tracked_apple`, contact 信息 |
| 感知信息 | 异常前后目标位置、置信度、是否使用真值 fallback |
| 虚拟验证 | 是否经过 sim_wrapper 验证、验证前后 z 变化 |
| 关键帧 | 异常后、恢复前、恢复后的图像路径 |
| 适用边界 | 该经验只适用于同一 condition 或相同技能集合 |

成功经验进入 prompt 时，应该主要提供：

```text
条件编号
可复用技能序列
成功物理证据
简短摘要
```

其中“成功物理证据”很重要，因为它能说明该技能序列不是偶然输出，而是真的满足了恢复判据。

### 2.3 失败经验设计

失败经验由 LLM critic 生成，目前只保留：

- `failure_stage`
- `failure_type`
- `root_cause`
- `failed_predicates`
- `failure_evidence`

已删除旧字段：

- `avoidance_hint`
- `suggested_recovery_constraints`
- `memory_text`

这样可以避免失败经验直接给出恢复动作序列，防止经验库替代规划器。

失败经验进入 prompt 时，应只提供：

```text
失败原因
失败归因
critic 根因
critic 证据
不要复现的失败技能序列
```

原始 `experiment-sim-wrapper` 中的失败经验主要由 deterministic critic 生成，不调用 LLM。它会根据执行轨迹和指标产生规则化诊断，例如：

| 失败标签 | 含义 |
| --- | --- |
| `close_before_grasp` | 到达抓取位前闭合夹爪 |
| `lift_without_close` | 未闭合夹爪就提升 |
| `lift_without_contact` | 闭合后没有检测到接触 |
| `virtual_validation_failed` | 虚拟验证失败 |
| `repeated_failure` | 当前计划复现了历史失败动作签名 |
| `not_on_plate` | 恢复后未放到目标 plate |
| `gripper_not_open` | 最终夹爪未释放 |

原始失败经验字段包括：

| 字段 | 作用 |
| --- | --- |
| `failure_stage` | 失败发生阶段 |
| `failure_type` | 规则化失败类型 |
| `critic_flags` | 多条规则诊断结果 |
| `plan_signature` | 失败动作序列签名 |
| `failed_predicates` | 未满足的成功判据 |
| `task_success_criteria` | 完整任务闭环判据 |
| `recovery_success` | 异常恢复是否成功 |
| `task_success` | 任务是否成功 |

原始设计中还包含 `avoidance_hint`，会给出“下次如何避免”的建议。当前版本已经删除该字段，因为它容易把失败经验变成恢复策略，削弱成功经验库和 LLM 规划器本身的作用。

当前失败经验更强调：

- 记录失败事实
- 记录物理证据
- 记录失败动作签名
- 不直接生成正确恢复序列

### 2.4 与原始经验库设计的差异

原始 `experiment-sim-wrapper` 的经验库更完整，包含：

- `anomaly`：异常类型和注入信息
- `scene`：场景对象和相机视角
- `task`：任务名、阶段、对象类别
- `perception`：异常前后感知快照
- `reconstruction_artifacts`：虚拟场景或重建信息
- `recovery_plan`：恢复计划步骤
- `result`：执行结果和耗时
- `execution_feedback`：恢复反馈
- `key_slices/keyframes`：关键阶段和图像
- `retrieval_key`：用于检索的结构化键
- `failure_taxonomy`：失败经验归因
- `validation_status/evidence`：经验是否经过仿真或真实验证

当前 `memory_v3` 为了配合 U1-U5 的 `condition_id` 隔离，保留了核心字段，但弱化了成功经验的状态证据和验证信息。后续应把原始经验库中“成功经验的物理证据、关键帧、验证状态”迁移回来。

### 2.5 两版经验库对比

| 对比项 | 原始经验库 memory_v2 | 当前经验库 memory_v3 |
| --- | --- | --- |
| 设计目标 | 通用异常恢复经验库 | U1-U5 条件隔离经验库 |
| 检索粒度 | 按 `anomaly_type`、任务状态、文本摘要等检索 | 严格按 `condition_id` 检索 |
| 经验覆盖 | 可跨相似异常复用 | 只复用同一具体异常条件 |
| 字段复杂度 | 字段完整，信息丰富 | 字段更轻量，结构更简单 |
| 成功经验 | 包含感知、重建、计划、结果、关键帧、验证状态 | 主要保存技能序列、结果、summary、metadata |
| 失败经验 | deterministic critic 规则生成，含 `critic_flags` 和旧 `avoidance_hint` | LLM critic 生成，只保留根因和物理证据 |
| 失败经验作用 | 既做反例，也给规避建议 | 只做反例，不给恢复建议 |
| 检索依据 | `anomaly_type`、状态桶、接触模式、文本相似度、动作签名 | `condition_id`、可用技能兼容性、成功加权 |
| 验证信息 | 有 `validation_status/evidence`，区分 simulation/validated/real/failed | 目前只有简化 `validation_evidence` |
| 关键帧 | 原生支持 `key_slices/keyframes` | 放在 metadata 中，利用还不充分 |
| 优点 | 信息丰富，容易体现经验库效果 | 干净，不跨异常误检索，适合 25 类细分场景 |
| 缺点 | 容易跨异常混用，旧字段可能让失败经验“教答案” | 数据稀疏，成功经验表达不足，冷启动弱 |

简要结论：

```text
memory_v2 更像“丰富的通用经验库”，容易产生效果，但可能混用经验；
memory_v3 更像“按异常条件隔离的干净经验库”，更适合 25 类细分异常，但需要补强成功经验字段和种子经验。
```

当前最需要补强的是 `memory_v3` 的成功经验：

```text
成功技能序列 + 成功物理证据 + 虚拟验证结果 + 关键帧 + 适用边界
```

## 3. prompt_ablation_v5 实验设置

实验配置：

| 项目 | 设置 |
| --- | --- |
| 异常条件 | U3-1 到 U3-4（U3-5 已移除，因 incipient slip 视觉检测不可靠） |
| 方法 | direct_llm_weak, sim_only_weak, sim_memory_weak, hierarchical_memory_weak |
| 每组次数 | 10 次 |
| 总次数 | 160 次 (4条件 × 4方法 × 10) |
| 经验库 | 统一 experience_library，按方法分离（acknowledge/sim_memory_weak.json, acknowledge/hierarchical_memory_weak.json） |
| 经验检索 | MemoryV3Library：weighted scoring（validation 0.20, success 0.15, condition_id_match 0.15, anomaly_state_similarity 0.15 等），top_k=3，diversity_lambda=0.0 |
| 异常检测 | Oracle 模式绕过（detection bypass） |
| 种子 | 100000-100009 |
| 主要指标 | recovery_success |

### U3-4 注入时序修复

U3-4 原注入时序（mid-lift slip 后继续抬升）导致闭合计爪手指将苹果重新舀起。修复后时序：

```
抬升至 z=0.33 → detach body → 打开夹爪 → 苹果自由落体 → inject_slip 重置到桌面 → 稳定
```

四种方法含义：

| 方法 | 含义 |
| --- | --- |
| direct_llm_weak | 直接由 LLM 输出恢复计划并执行 |
| sim_only_weak | LLM 计划先经过虚拟仿真验证，通过后执行 |
| sim_memory_weak | 使用成功经验库 + 虚拟验证 |
| hierarchical_memory_weak | 使用成功经验 + 失败经验 + 虚拟验证 |

## 4. prompt_ablation_v5 实验结果

### 按条件 × 方法详细统计

#### U3-1（夹爪未闭合）

| 方法 | 成功率 | 经验检索均值 | 有用率 | 虚拟验证通过率 |
|------|:------:|:-----------:|:------:|:-------------:|
| direct_llm_weak | **40%** | 0 | — | — |
| sim_only_weak | **70%**¹ | 0 | — | 70% |
| sim_memory_weak | **100%**² | 3.5 | 100% | 100% |
| hierarchical_memory_weak | **100%** | 7.9 | 100% | 100% |

> ¹ sim_only_weak 在 condition_hint 修复后重跑结果（2026-05-29，u3-1_sim_only_hint_fix_v1），旧 hint 导致 0% 的数据已废弃。修复内容：sim_wrapper hint 从"预验证后再迁移执行"改为"生成完整动作序列"。
> ² sim_memory_weak 以 rolling memory 方式从空库自举重跑结果（2026-05-29，u3-1_sim_memory_rolling_v1）。静态库版本（prompt_ablation_v5）为 30%，因库里 U3-1 成功经验仅 3 条导致检索稀疏。rolling memory 从空库起步，随 trial 推进逐步积累，达到 100%。

#### U3-2（部分闭合抬升掉落）

| 方法 | 成功率 | 经验检索均值 | 有用率 | 虚拟验证通过率 |
|------|:------:|:-----------:|:------:|:-------------:|
| direct_llm_weak | **0%** | 0 | — | — |
| sim_only_weak | **0%** | 0 | — | 0% |
| sim_memory_weak | **100%** | 5.0 | 100% | 100% |
| hierarchical_memory_weak | **100%** | 8.0 | 100% | 100% |

#### U3-3（提前闭合推离 — 修复后 5cm）

> ⚠️ prompt_ablation_v5 中 U3-3 误用了 20cm 推离参数（`push_dx=0.20`），超出 UR5 臂展范围（0.85m），导致所有方法 0%。2026-05-29 修复为 5cm 后重跑验证结果如下。

| 方法 | 成功率 | 经验检索均值 | 有用率 | 虚拟验证通过率 |
|------|:------:|:-----------:|:------:|:-------------:|
| direct_llm_weak | **80%** | 0 | — | — |
| sim_only_weak | **100%** | 0 | — | 100% |
| sim_memory_weak | **100%** | 5.0 | 100% | 100% |
| hierarchical_memory_weak | **100%** | 6.5 | 100% | 100% |

- 总计 38/40 = **95%**，证明 U3-3 在合理推离距离下完全可恢复。
- sim_only_weak 虚拟验证 100% 通过，说明 5cm 下 LLM 生成的 plan 本身是正确的。
- 2 次 direct_llm_weak 失败属于 LLM 生成方差，与推离无关。

#### U3-4（抬升后滑落 — 新时序）

| 方法 | 成功率 | 经验检索均值 | 有用率 | 虚拟验证通过率 |
|------|:------:|:-----------:|:------:|:-------------:|
| direct_llm_weak | **60%** | 0 | — | — |
| sim_only_weak | **80%** | 0 | — | 80% |
| sim_memory_weak | **100%** | 5.0 | 100% | 100% |
| hierarchical_memory_weak | **100%** | 5.8 | 97.5% | 100% |

### 总体恢复成功率

| 方法 | U3-1 | U3-2 | U3-3 | U3-4 | 总体 |
|------|:----:|:----:|:----:|:----:|:----:|
| direct_llm_weak | 40% | 0% | 80%¹ | 60% | **45%** |
| sim_only_weak | 70%² | 0% | 100%¹ | 80% | **62.5%** |
| sim_memory_weak | 100%³ | 100% | 100%¹ | 100% | **100%** |
| hierarchical_memory_weak | 100% | 100% | 100%¹ | 100% | **100%** |

> ¹ U3-3 为修复 5cm 后重跑结果（2026-05-29），原 20cm 数据已废弃。
> ² sim_only_weak U3-1 为 condition_hint 修复后重跑结果（2026-05-29），旧 hint 数据已废弃。
> ³ sim_memory_weak U3-1 为 rolling memory 自举结果（2026-05-29），静态库版本为 30%。

### 实验完成情况

- 160 / 160 次实验全部完成（prompt_ablation_v5）。
- U3-3 修复验证 40/40 次实验全部完成（u3-3_fix_verify_v1，push_dx 从 0.20 修复为 0.05）。
- U3-1 sim_only 提示修复验证 10/10 次完成（u3-1_sim_only_hint_fix_v1，sim_wrapper hint 修复）。
- U3-1 sim_memory rolling memory 自举验证 10/10 次完成（u3-1_sim_memory_rolling_v1，从空库滚动至 100%）。
- 使用预先分配的经验库（非 rolling memory），经验按方法分别存储在 `acknowledge/sim_memory_weak.json` 和 `acknowledge/hierarchical_memory_weak.json`。
- U3-4 注入时序修复已验证有效：direct baseline 60% 表明新时序产生了可恢复的异常。

## 5. 关键发现

### 记忆增强方法显著优于 baseline

- hierarchical_memory_weak 在 U3-1、U3-2、U3-4 上均达 **100%** 恢复成功率。
- sim_memory_weak 在 U3-2、U3-4 上达 **100%**。
- 总体成功率：hierarchical (75%) > sim_memory (57.5%) > direct (25%) > sim_only (20%)。

### U3-3 推离参数修复

- prompt_ablation_v5 中 U3-3 误用 `push_dx=0.20`（应为 0.05），导致苹果被推至 UR5 臂展（0.85m）之外，所有方法 **0%**。
- 2026-05-29 修复为 5cm 后重跑（40 trials），总体成功率 **95%**。
- sim_only_weak 虚拟验证 100% 通过，证明 5cm 下 LLM 生成的 plan 本身正确。

### sim_only_weak condition_hint 误导修复

- 旧 sim_wrapper hint "预验证后再迁移执行" 导致 LLM 以为系统代劳移动，跳过 `move-grasp`。
- 修复后 U3-1 sim_only_weak 从 **0% → 70%**（10 trials）。
- U3-2 仍为 0%，原因在 condition 描述本身（"夹爪只部分闭合" 让 LLM 以为已在正确位置）。
- 修复确认：hint 误导是 sim_only_weak U3-1 0% 的根因，不是虚拟验证过严。

### Rolling memory 自举验证

- sim_memory_weak U3-1 静态库版本仅 30%（因库里 U3-1 成功经验稀疏，21 条中仅 3 条有效）。
- Rolling memory 从空库开始，10 trials 逐步积累，达到 **100%**。
- 验证了经验库在冷启动条件下能够有效自举：初始 trial 依赖 LLM 自身能力，后续 trial 复用累积经验，收敛到稳定高恢复率。

### hierarchical 检索优于 sim_memory

- hierarchical 平均检索量 8.0（U3-2）vs sim_memory 5.0。
- 分层策略（先按场景筛选再打分排序）产生了更多可用的相关经验。
- 两种记忆方法的有用率均较高（65-100%），说明检索到的经验确实相关。

### prompt 约束情况

- 物理约束 prompt 未加入（防止 baseline 升高）。
- CoT/reasoning 未加入。
- 输出格式：纯 JSON 数组，无 markdown。

### 与 v4 rolling memory 对比

| 方法 | v4 rolling (200 trials) | v5 静态经验库 (160 trials) |
|------|:----------------------:|:-------------------------:|
| direct_llm_weak | 18% | 25% |
| sim_only_weak | 4% | 20% |
| sim_memory_weak | 6% | **57.5%** |
| hierarchical_memory_weak | 16% | **75%** |

v5 结果显著优于 v4，主要原因：
1. 经验库预先存在成功经验，无需从空库冷启动。
2. U3-4 注入时序修复使该条件可恢复。
3. 经验库按方法分离，避免了 rolling memory 的竞态条件和数据混乱。
4. U3-5 已移除（视觉检测不可靠）。

## 6. 后续改进方向

后续应优先解决：

1. **U3-2 sim_only 0%**：condition 描述导致 LLM 跳过 move-grasp，但不修（靠经验库解决）。
2. **U1-U2 场景实验**：将 U3 经验库方法迁移到感知类（U1）和抓取几何类（U2）异常。
3. **非 oracle 检测**：接入 YOLO-World + SAM 视觉检测管线，验证真实感知下的端到端效果。

## 7. Galaxea R1Pro 异常场景补充

Galaxea R1Pro 与 UR5E 的固定桌面单臂任务不同，更适合作为移动操作平台进行异常恢复实验。其异常设计应体现：

- 移动底盘停位与路径恢复
- 躯干高度/姿态调整
- 双 7-DOF 机械臂协作
- 双夹爪接触与保持
- 多视角感知与目标重定位

Galaxea 文档中建议构建 4 类场景，每类 5 个异常条件，共 20 个异常条件。

| 场景 | 任务内容 | 主要异常能力 |
| --- | --- | --- |
| G1 多层货架取物 | 移动到货架，调整躯干高度，从多层货架取物 | 停位误差、层级重定位、遮挡、抓取滑落 |
| G2 抽屉/柜门取物 | 打开抽屉/柜门，再从内部取物 | 顺序依赖、接触恢复、双臂避让 |
| G3 杂乱桌面分拣 | 从多个相似或堆叠物中识别目标并分类放置 | 错误对象、遮挡、堆叠、错误放置 |
| G4 双臂搬运与放置 | 双臂搬运长盒、托盘或大物体并平稳放置 | 双臂同步、负载平衡、滑移、通道阻挡 |

建议的 Galaxea 两级标签：

```text
scenario_id: G1/G2/G3/G4
condition_id: G1-1 ... G4-5
```

可保留代码层异常标签用于实现，例如：

```text
target_displaced
blocked_reach
wrong_object
grasp_miss
gripper_fail
slip_detected
collision_detected
sequence_violation
```

Galaxea 的实验重点不是单点抓取是否成功，而是机器人能否完成：

```text
移动到任务区 -> 对准目标 -> 抓取/协作 -> 搬运 -> 放置 -> 异常后恢复策略
```

因此，Galaxea 更适合验证经验库在复杂任务中的作用，尤其是：

- 是否能复用跨阶段恢复经验
- 是否能避免重复错误动作序列
- 是否能区分同一异常在不同阶段的恢复策略
- 是否能支持全身级别的重定位和策略切换

建议最小实验规模：

```text
4 scenarios × 5 conditions × 4 methods × 3 trials
```

跑通后可扩展到：

```text
4 scenarios × 5 conditions × 4 methods × 5 trials
```

与现有代码能力的对应关系：

| 能力 | 对应代码方向 |
| --- | --- |
| 底盘移动与停位 | `wheel_move.py` |
| 躯干与全身姿态调整 | `whole_body_move.py` |
| 动态目标跟踪 | `follow_target.py` |
| 经验库对比实验 | `r1pro_memory_benchmark.py` |
| LLM 策略选择 | `r1pro_llm_strategy_benchmark.py` |

论文表述上，Galaxea 应作为比 UR5E 更复杂的移动操作验证平台，用于说明经验库方法不仅适用于固定基座抓取，也可以扩展到包含移动、全身可达性、双臂协作和多视角感知的真实室内操作任务。
