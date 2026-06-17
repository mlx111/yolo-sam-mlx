# Baseline 提升方案

## 背景

当前 U3 实验 baseline（direct_llm_weak）成功率仅 18%，unsafe gripper action 是 dominant failure（145/200 trials 出现 open-after-lift）。经验库难以发挥作用的原因：80%+ 条目是失败案例，检索返回 mostly 负样本，LLM 看到大量 "不要这样做" 但缺少 "应该这样做" 的正面指导。

需要 baseline 提升至 ~40%，经验库才有足够的成功基础来展示增量价值。

---

## Plan A: Prompt Engineering 修复（最小改动）

### 思路

不改动任何架构、不加外部约束，仅通过改进 skill 描述让 LLM 自然产生安全动作序列。skill description 是 prompt engineering，属于工程质量优化，不构成方法论贡献，不影响 ablation 纯度。

### 需要修改的文件

**`prompts/skills.yaml`** — 修改 `vertical-grasp` 和 `gripper-action` 的描述

#### 关键问题定位

当前 unsafe 的 root cause：LLM 学到的 plan 模板是：
```
move-grasp → gripper-action(state=1) → vertical-grasp → ... → gripper-action(state=0)
```
即：抓住 → 提起 → 松开。模型不理解 "提起后松开 = 摔落物体"。

#### 具体修改

1. **`gripper-action` skill 描述**：明确强调 state=0 只能在 move-grasp 之后、垂直移动之前执行。提起后绝对不能松开。

2. **`vertical-grasp` skill 描述**：明确说明此动作之后不要再 gripper-action(state=0)，否则物体会掉落。

3. 可选的辅助措施：
   - 在 system prompt 中加入一条自然语言的安全提醒（不是硬约束，只是提醒）
   - 在 few-shot example 中展示正确的终止模式（gripper-action(state=1) 保持到结束）

### 预期效果

| 指标 | 当前值 | 目标值 |
|---|---|---|
| open-after-lift 比例 | 72.5% | <30% |
| unsafe_gripper_action_count | 0.88/trial | <0.3/trial |
| 整体成功率 | 18% | ~35-45% |

### 验证方法

```
python run_experiment_v4.py --mode single_batch \
  --method direct_llm --condition all \
  --trials 10 --output results/prompt_ablation/
```

先用少量 trials 验证 open-after-lift 是否减少，确认后再跑完整 batch。

### 局限与风险

- 不改变模型本身的推理能力，高风险是 prompt 改完后模型还是学不会
- 可能过拟合到当前 U3 条件的特定模式，迁移到新条件时仍需调整
- **不是方法论贡献，论文中只能放在 Implementation Details 一节**

---

## Plan B: Inner Monologue-style Stepwise Execution（方法级 baseline）

### 思路

借鉴 Inner Monologue 的思路：不要一次性生成完整的 plan 然后执行，而是逐步生成 → 执行 → 检测结果 → 反馈给 LLM → 继续下一步。这样：

1. 每一步都有成功检测（grasp 是否成功、lift 是否成功）
2. 检测到失败立即 replan，不会继续执行后续 unsafe 动作
3. 天然避免了 open-after-lift：检测到已 lift 后，下一步如果 LLM 说 "松开"，系统可以提前干预或 replan

### 当前架构 vs Inner Monologue 架构

```
当前: LLM ──(full plan)──→ Executor ──(execute all)──→ done
                                     └── 失败也无法干预

Inner Monologue: LLM ──(step)──→ Executor ──(result)──→ Success Detector
                                     ↑                       │
                                     └─── feedback ──────────┘
```

### 实现步骤

#### Step 1: 新增 Stepwise Recovery Strategy

**文件:** `experiment_method_runner.py`

添加一个新的 method 类型 `direct_llm_stepwise`：

```python
class DirectLLMStepwiseRecoveryStrategy(BaseRecoveryStrategy):
    """Inner Monologue-style step-by-step execution with feedback."""
    
    method_name = "direct_llm_stepwise"
    
    def recover(self, scene_codes, perf_measure, error_msg):
        max_steps = 15  # 最大步数
        executed = []
        
        for i in range(max_steps):
            # 1. 将已执行步骤 + 检测结果作为 context 发给 LLM
            context = self._build_stepwise_context(executed, scene_codes)
            llm_response = self.llm.generate(context)
            
            # 2. 只取下一个 action（不要多个）
            action = self._parse_single_action(llm_response)
            
            # 3. 执行
            result = self.sim.execute(action)
            executed.append({"action": action, "result": result})
            
            # 4. 成功检测
            detection = self._detect_success(action, result)
            if detection.get("grasp_failed"):
                # 重新尝试 grasp
                continue
            if detection.get("object_dropped"):
                # 记录失败，replan
                correction = self.llm.generate(
                    f"物体掉落，原因是：{detection['reason']}，请重新规划"
                )
                # 处理 correction
                continue
            if detection.get("task_complete"):
                break
        
        return executed, self._evaluate(scene_codes)
```

#### Step 2: 实现 Success Detection

**文件:** `sim_wrapper.py` 或新建 `stepwise_detector.py`

关键检测维度：
- **grasp success**: gripper 闭合后，接触点数量是否 > 阈值
- **lift success**: 垂直上升后，物体是否跟随 gripper 移动
- **object dropped**: gripper 张开后，物体位置是否显著变化（相对于底座）
- **task complete**: 所有目标物体是否达到目标区域

#### Step 3: 适配经验库集成

将 `sim_memory` / `hierarchical_memory` 方法同样适配到 stepwise 模式：

```python
class StepwiseMemoryRecoveryStrategy(DirectLLMStepwiseRecoveryStrategy):
    method_name = "hierarchical_stepwise"
    
    def _build_stepwise_context(self, executed, scene_codes):
        # 在 standard stepwise context 基础上，加入经验库检索结果
        memories = self.experience_library.query(scene_codes)
        return self._format_context_with_memories(executed, scene_codes, memories)
```

这样 ablation 框架变成：

| Method | Baseline | +Virtual | +Success Lib | +Failure Lib |
|---|---|---|---|---|
| direct_llm | Plan A | ✓ | ✓ | ✓ |
| **direct_llm_stepwise** | **Plan B** | ✓ | ✓ | ✓ |

两个 baseline 可以独立对比增量。

#### Step 4: 配置文件

**文件:** `configs/methods.yaml`

```yaml
direct_llm_stepwise:
  recovery: direct_llm_stepwise
  config:
    max_steps: 15
    detection:
      grasp_contact_threshold: 3
      lift_displacement_threshold: 0.02
    replan_on:
      - grasp_failed
      - object_dropped
      - plan_complete_with_gaps
```

### 预期效果

| 指标 | 当前 direct_llm | direct_llm_stepwise |
|---|---|---|
| 成功率 | 18% | ~40-50% |
| unsafe_gripper_action | 0.88/trial | ~0.1/trial（replan 前就被检测到） |
| 平均执行步数 | 6-8 | 8-12（含 replan） |
| 可解释性 | 低（黑盒 full plan） | 高（每步可见） |

### 文献依据

- **Inner Monologue** (Huang et al., 2022): Embodied Reasoning through Planning with Language Models. 逐步执行 + 成功检测 + 反馈闭环。
- **SayCan** (Ahn et al., 2022): Do as I can, not as I say. LLM 选动作 + 低层 policy 执行 + 成功检测。
- **ExpTeach** (Zhou et al., 2024): 用成功检测构建 teaching signal，指导 LLM 从经验中学习。

### 风险

- 需要实现 success detection 逻辑（sim 环境中可行，但检测精度可能影响效果）
- 逐步执行增加耗时（但 sim 环境下不明显）
- 检测器本身不构成贡献，但 stepwise + experience library 的增量是贡献

---

## 对比总结

| 维度 | Plan A (Prompt Engineering) | Plan B (Inner Monologue) |
|---|---|---|
| 改动量 | 小（1-2 个文件，几行描述） | 中（3-4 个文件，新 class） |
| 风险 | 低 | 中（检测器质量是关键） |
| 预期提升 | 18% → ~35-45% | 18% → ~40-50% |
| 论文可写性 | Implementation Details | 可作为 baseline method |
| 与经验库的兼容性 | 完全兼容 | 需适配 stepwise 检索 |
| 对贡献的影响 | 无（纯 prompt 优化） | baseline 来自文献，增量是贡献 |

### 建议路线

**短期**: 先做 Plan A 快速验证（1-2 天），如果 prompt 修复后 baseline 能达到 30%+，则不需要大改动。

**如果 Plan A 不够**: 实施 Plan B，这也是更有论文价值的方案——stepwise execution + experience library 的组合本身就是一个小贡献点（类似 ExpTeach 的思路）。

建议先从 Plan A 开始——改动最小，验证成本低——如果效果不够再切换到 Plan B。
