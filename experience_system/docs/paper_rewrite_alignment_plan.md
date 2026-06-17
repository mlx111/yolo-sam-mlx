# Paper Rewrite Alignment Plan

本文档用于把当前论文写作从旧的 UR5e `memory_v2/memory_v3`
异常恢复实验线，对齐到当前已经实现的 `experience_system` 主线。

## 1. 核心结论

当前代码主线已经不是单纯的 UR5e 正负经验库，而是：

```text
Sim-Real dual-source experience memory
for anomaly recovery with sandbox rollout, critic scoring,
gap calibration, stage-aware retrieval, visual evidence, lessons,
writeback, and real-format episode support.
```

现有 `论文/paper_icra.tex` 仍主要描述：

```text
UR5e tabletop apple grasping
memory_v3
U1-U5 benchmark
STM/LTM
positive/failed memory prompt
CLIP+FAISS visual retrieval
```

这条旧线可以作为历史实验背景或补充实验，但不能直接代表当前
`experience_system` 的实现。论文需要重写主线，否则会出现“论文声称的系统”
和“当前可复现代码证据”不一致的问题。

## 2. 建议论文定位

推荐新题目方向：

```text
Experience-Guided Robot Anomaly Recovery with Sim-Real Dual-Source Memory
and Sandbox-Critic Candidate Selection
```

推荐一句话主张：

```text
The system ranks anomaly-recovery candidates using memory support,
failure risk, stage-specific retrieval evidence, gap-calibrated sandbox
rollout, motion-level critic feedback, and LLM multi-candidate plan search.
```

这句话是当前最强且安全的主 claim。

## 3. 推荐贡献点

论文贡献建议从旧的“经验库提升成功率”改为以下五点：

1. 统一经验 schema：支持 simulation、pseudo-real、real-format episode，
   并统一保存 retrieval key、critic result、sim-real gap、sandbox calibration、
   sensor evidence 和 memory gate。
2. 双源 gap 与 calibration：支持 sim/pseudo-real pairing、gap signature，
   并将 gap 转换为 sandbox 风险惩罚和 object initial pose bias。
3. 候选策略沙盒推演：实现 LLM candidate plans -> semantic validation ->
   parallel sandbox rollout -> critic -> validated_robot_plan。
4. 风险记忆参与排序：Top-O failure memory、motion-level critic、stage-aware
   retrieval、visual keyframe retrieval 和 LLM lesson 都参与候选排序或解释。
5. 可复现实验报告链：ablation、safety stress、harder adversarial stress、
   writeback benchmark、calibration ablation、paper evidence summary 都可生成。

## 4. 旧论文需要重写的部分

### Abstract

旧 abstract 主要写 UR5e、920 trials、U1-U5 success rate。新 abstract 应转为：

```text
structured dual-source memory
candidate sandbox rollout
motion-level critic
gap-calibrated sandbox scoring
reportable ablations
real-format episode support
```

如果继续引用旧 U1-U5 成功率，必须明确它是 legacy UR5e benchmark，
不是当前 R1Pro `experience_system` 主实验。

### Introduction

需要从“LLM 异常恢复缺少历史经验”升级为：

```text
LLM anomaly recovery needs executable memory evidence,
failure risk priors, and pre-execution sandbox evaluation.
```

重点不是 RAG，而是：

```text
memory -> candidate generation/ranking -> sandbox rollout -> critic -> selection
```

### System

旧系统章节应替换为当前八个模块：

```text
adapter normalization
ExperienceEntry library
write gate and lifecycle
structured/stage/visual/sensor retrieval
sim-real pairing and gap extraction
gap-derived sandbox calibration
candidate sandbox rollout and critic
writeback and paper evidence reports
```

### Memory Schema

旧 `memory_v3` schema 应替换为 `ExperienceEntry`：

```text
robot, scenario, condition, task, anomaly
skill_sequence, action_trace, observation_trace
sensor_summary, sensor_evidence
object_state, spatial_state
memory_gate, critic_result, failure_taxonomy
sim_real_pair, sim_real_gap, sandbox_calibration
real_episode_ref, raw_refs
```

### Experiments

当前最安全的实验结构：

```text
E1 universal memory pipeline summary
E2 candidate sandbox rollout
E3 calibration ablation
E4 stage-aware retrieval/context report
E5 visual retrieval ablation
E6 lesson quality and policy adjustment
E7 safety stress
E8 harder adversarial safety stress
E9 repeated writeback benchmark
E10 real-format/pseudo-real evidence pack
```

旧 U1-U5 可以保留为 background/legacy baseline，但不要和当前 R1Pro
报告混成同一个主实验口径。

## 5. 可直接引用的证据文件

当前主证据目录：

```text
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/
```

关键文件：

```text
universal_experience_library.json
summary_report.json
real_format_evidence_pack.json
paper_evidence_summary.json
paper_evidence_summary.md
llm_plan_candidate_search_g3_clean_real_llm_report.json
llm_plan_search_ablation_g3_clean_real_llm.json
sandbox_rollout_g4_place_occupied.json
visual_retrieval_ablation_g3_clean.json
visual_retrieval_ablation_g4_place_occupied.json
sandbox_calibration_ablation_g4_place_occupied.json
lesson_quality_report_g4_place_occupied.json
safety_stress_g4_place_occupied.json
harder_safety_stress_g4_place_occupied.json
writeback_benchmark_g3_clean.json
```

主 evidence table 已由以下工具生成：

```text
experience_system/tools/build_paper_evidence_summary.py
```

当前状态：

```text
claim_count = 22
supported_claim_count = 21
missing_report_count = 0
sensor evidence row = partial
```

`sensor evidence row = partial` 的原因是主库尚无真实机器人 sensor evidence。
这不是代码缺失，而是数据缺失。

## 6. 安全 claim 边界

可以写：

```text
The implementation supports real-format episode import and pseudo-real
evidence for exercising sim-real pairing, gap extraction, and sandbox
calibration.
```

可以写：

```text
Candidate plans are shadow-executed in MuJoCo and fused with memory scores
before selection.
```

可以写：

```text
The rule-based motion critic identifies risky candidates in stress reports.
```

不能写：

```text
Real robot experiments prove improved success rate.
The sandbox is a full digital twin.
The critic is learned.
Sensor-derived calibration improves real execution.
The system statistically improves long-horizon real-world recovery.
```

## 7. 还可以继续优化的非真机事项

### P1. 论文正文重写

新建或重写一份：

```text
论文/paper_experience_system.tex
```

不要直接覆盖 `paper_icra.tex`，除非确认旧 UR5e 论文不再需要。

### P2. Evidence Appendix

从 `paper_evidence_summary.json` 自动生成：

```text
paper_evidence_appendix.md
paper_evidence_appendix.tex
```

内容包括：

```text
claim
status
primary report
key metrics
safe wording
avoid wording
```

当前已实现：

```text
experience_system/tools/build_paper_evidence_appendix.py
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_appendix.md
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_appendix.tex
```

用途：把 22 条 claim 的状态、主证据文件、关键指标、安全写法和禁写边界
整理成论文附录。当前 21 条为 `supported`，传感器/真机证据相关 1 条为
`partial`，不能写成真实机器人验证结论。

### P3. Memory-Type Coverage Report

参考 RoboMemory/RoboMME，把经验库覆盖度统计为：

```text
temporal_memory
spatial_memory
episodic_memory
semantic_memory
perceptual_memory
sim_real_gap_memory
```

这能增强论文中“多类型机器人记忆系统”的说服力。

当前已实现：

```text
experience_system/tools/build_memory_type_coverage_report.py
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/memory_type_coverage_report.json
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/memory_type_coverage_report.md
```

当前报告统计 12 条经验，平均每条覆盖 5.6667 类记忆，8 条覆盖全部六类。
该报告已接入 `paper_evidence_summary.json/md` 和 `paper_evidence_appendix.md/tex`，
作为第 12 条论文证据 claim。注意：这是字段级覆盖统计，不是 RoboMME 规模
benchmark，也不证明真实机器人传感器验证。

### P4. Universal Text-Semantic Retrieval Report

当前已实现：

```text
experience_system/tools/build_text_semantic_memory_report.py
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/text_semantic_memory_report.json
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/text_semantic_memory_report.md
```

该工具从 scenario、condition、task、anomaly、failure taxonomy、critic flags、
sim-real gap、memory tags、retrieval key 和 skill sequence 构建显式文本语义摘要，
并默认使用 TF-IDF 向量 + FAISS inner-product index 做辅助语义检索报告
（缺少依赖时可回退到 token-overlap）。当前 12 条经验都有非空 semantic
summary，9 个代表性 query 的 `semantic_signal_rate=1.0`，当前报告后端为
`faiss_tfidf`。

该报告已接入 `paper_evidence_summary.json/md` 和 `paper_evidence_appendix.md/tex`，
作为第 13 条论文证据 claim。注意：这不是 learned language encoder，也不是
neural embedding benchmark，只能写成轻量 TF-IDF + FAISS 文本语义摘要和辅助
检索信号。

### P5. 文档分流

将旧文档明确标记：

```text
legacy UR5e memory line
current R1Pro experience_system line
real-robot future work
```

避免后续写作时把三条线混在一起。

## 8. 等真机数据后再做

以下内容应等真实 episode 导入后再推进：

```text
real robot success-rate comparison
real sensor evidence supported claim
sensor-derived sim-real gap statistics from real runs
real execution writeback loop
real-format evidence pack supported by true real-source entries
```

真机数据导入后，必须重新生成：

```text
real_episode_import_report.json
real_format_evidence_pack.json
paper_evidence_summary.json/md
```

并检查 sensor 行是否从 `partial` 变为 `supported`。

## 9. 建议下一步

下一步最值得做的是二选一：

```text
1. 根据 experience_system/docs/paper_experience_system.md 写正式论文 tex
2. 彻底清理/分流旧文档：legacy UR5e / current R1Pro / real-robot future work
```

`tools/build_paper_evidence_appendix.py` 已实现并已生成 Markdown/LaTeX 附录。
后续如果新增报告，只需要重新生成 `paper_evidence_summary.json/md` 和
`paper_evidence_appendix.md/tex`。
