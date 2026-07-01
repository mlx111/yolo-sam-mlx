# Experience System Document Lineage Index

本文档是当前经验库论文写作和实现维护的文档分流索引。目标是把三条线彻底
分开：

```text
legacy UR5e / experiment-sim-wrapper memory_v2-v3 line
current R1Pro / experience_system line
real-robot collection / future evidence line
```

后续写论文、README 或实验报告时，以本文档为入口判断文档能否作为当前
`experience_system` 的实现证据。

## 0. 使用规则

| 文档类别 | 能否作为当前论文主证据 | 使用方式 |
|---|---:|---|
| `current` | 是 | 可直接支撑当前 R1Pro `experience_system` claim。 |
| `generated evidence` | 是 | 可作为指标、表格和 appendix 的主证据。 |
| `real-robot collection` | 否 | 只能说明真机数据格式和采集计划，不能证明真机效果。 |
| `auxiliary` | 部分 | 只能支撑实现说明、技能规划或图示，不作为实验结论。 |
| `legacy UR5e` | 否 | 只能作为历史背景或旧 baseline，不得混入当前主实验。 |
| `research background` | 否 | 只能作为相关工作/设计启发，不作为代码实现证据。 |

硬性边界：

```text
1. 当前主 claim 只引用 experience_system/docs 和 results/memory/universal_pipeline_calibration_v1。
2. ur5e_mujoco 文档全部视为 legacy UR5e 线。
3. 经验库/ 目录中的论文阅读材料全部视为 research background。
4. 真机采集文档不等于真机验证结果。
5. 旧 U1-U5 成功率不得和当前 R1Pro G3/G4 evidence 混成同一实验表。
```

## 1. 当前主线文档

这些文档是当前论文和实现对齐的主要依据。

| 文档 | 状态 | 用途 |
|---|---|---|
| `experience_system/docs/universal_experience_memory_current_implementation.md` | current | 当前实现总说明。 |
| `experience_system/docs/universal_sim_real_experience_memory_design.md` | current | 通用 Sim-Real 双源经验库设计。 |
| `experience_system/docs/paper_implementation_alignment.md` | current | 论文 claim 与实现证据边界。 |
| `experience_system/docs/paper_rewrite_alignment_plan.md` | current | 论文重写路线和已完成增强项。 |
| `experience_system/docs/paper_experience_system.md` | current | 当前 R1Pro `experience_system` 论文正文草稿。 |
| `experience_system/docs/remaining_experience_system_roadmap.md` | current | 剩余优化路线。 |
| `experience_system/README.md` | current | 工具入口、运行命令和 LLM `.env` 配置说明。 |
| `galaxea_mujoco/docs/no_real_robot_remaining_work_zh.md` | current | 无真机阶段剩余工作和已完成状态。 |
| `galaxea_mujoco/docs/real_robot_experience_sandbox_experiment_plan.md` | current | 现场真机实验闭环设计。 |

这些文档可以互相引用，但如果内容冲突，优先级为：

```text
paper_implementation_alignment.md
> paper_rewrite_alignment_plan.md
> remaining_experience_system_roadmap.md
> galaxea_mujoco/docs/no_real_robot_remaining_work_zh.md
> universal_experience_memory_current_implementation.md
> universal_sim_real_experience_memory_design.md
```

原因：`paper_implementation_alignment.md` 记录的是当前 claim 与证据边界，
最接近论文最终表述。

## 2. 当前生成证据文件

这些文件是当前论文可直接引用的主要 evidence。它们不是手写结论，而是由
工具从实验或报告中生成。

| 文件 | 状态 | 用途 |
|---|---|---|
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_summary.json` | generated evidence | claim 级 JSON evidence summary。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_summary.md` | generated evidence | claim 级 Markdown evidence table。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_appendix.md` | generated evidence | 论文附录 Markdown。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_appendix.tex` | generated evidence | 论文附录 LaTeX。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/memory_type_coverage_report.json` | generated evidence | 多类型记忆覆盖报告。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/memory_type_coverage_report.md` | generated evidence | 多类型记忆覆盖 Markdown。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/text_semantic_memory_report.json` | generated evidence | TF-IDF + FAISS 文本语义检索报告。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/text_semantic_memory_report.md` | generated evidence | 文本语义检索 Markdown。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/sandbox_rollout_g4_place_occupied.json` | generated evidence | G4 候选 sandbox rollout 主报告。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/sandbox_calibration_ablation_g4_place_occupied.json` | generated evidence | sandbox calibration ablation。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/safety_stress_g4_place_occupied.json` | generated evidence | 标准 safety stress。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/harder_safety_stress_g4_place_occupied.json` | generated evidence | adversarial safety stress。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/writeback_benchmark_g3_clean.json` | generated evidence | repeated writeback benchmark。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/physical_default_audit_report.json` | generated evidence | 核心技能默认 physical-first / `direct_qpos=false` 审计。 |
| `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/write_policy_pressure_report.json` | generated evidence | write_policy write/merge/skip/reject 压力测试。 |

这些文件的安全写法和禁写边界以 `paper_evidence_summary.json/md` 和
`paper_evidence_appendix.md/tex` 为准。

## 3. 真机采集线文档

这些文档用于后续真实机器人 episode 采集和导入，不代表当前已有真机验证结果。

| 文档 | 状态 | 用途 |
|---|---|---|
| `experience_system/docs/r1pro_real_episode_collection_guide.md` | real-robot collection | R1Pro 真机 episode 采集说明。 |
| `experience_system/templates/r1pro_real_episode_template.json` | real-format import | episode JSON 模板。 |

使用限制：

```text
可以写：system supports real-format episode import.
不能写：real robot experiments validate recovery success.
```

## 4. 当前辅助文档

这些文档仍可用于实现说明或技能规划，但不是论文主证据。

| 文档 | 状态 | 用途 |
|---|---|---|
| `galaxea_mujoco/docs/galaxea_skill_gap_plan.md` | auxiliary | R1Pro/Galaxea 技能缺口清单。 |
| `galaxea_mujoco/docs/pseudo_real_episode_matrix.md` | auxiliary | pseudo-real episode 生成矩阵。 |
| `galaxea_mujoco/docs/visual_keyframe_retrieval.md` | auxiliary | visual keyframe retrieval 使用说明。 |
| `论文/experience_memory_diagram_philosophy.md` | auxiliary | 经验库图示设计语言说明。 |

## 5. Legacy UR5e / experiment-sim-wrapper 文档

这些文档仍可能存在于仓库中，但它们描述的是 UR5e、U1-U5、
`memory_v2/memory_v3/memory_v3_plus` 或旧 `experiment-sim-wrapper` 线。
它们不能作为当前 R1Pro `experience_system` 的主证据。

| 文档 | 状态 | 用途 |
|---|---|---|
| `ur5e_mujoco/Sim-Real双源经验库开发基线设计文档.md` | legacy UR5e | 旧 UR5e 双源经验库设计基线，可作为历史启发。 |
| `ur5e_mujoco/Sim-Real双源经验库系统说明.md` | legacy UR5e | 旧系统说明。 |
| `ur5e_mujoco/Sim-Real双源经验库实验报告.md` | legacy UR5e | 旧实验报告。 |
| `ur5e_mujoco/docs/双源经验库设计与使用说明.md` | legacy UR5e | 旧使用说明。 |
| `ur5e_mujoco/U3_MEMORY_EXPERIMENT_SUMMARY.md` | legacy UR5e | U1-U5 / memory_v3 总结。 |
| `ur5e_mujoco/MEMORY_V3_UPGRADE_PLAN.md` | legacy UR5e | memory_v3 升级计划。 |
| `ur5e_mujoco/PLAN_BASELINE_IMPROVEMENT.md` | legacy UR5e | 旧 baseline prompt/skill 改进。 |
| `ur5e_mujoco/EXPERIMENT_NOTES.md` | legacy UR5e | 旧实验记录。 |
| `ur5e_mujoco/论文方法章节草稿.md` | legacy UR5e | 旧方法章节草稿。 |
| `ur5e_mujoco/plans/experience_improvement_plan.md` | legacy UR5e | 旧优化计划。 |

这些文档可以用于 Related Work / Historical Baseline 说明，但必须加限定：

```text
legacy UR5e benchmark / previous implementation line
```

不得写成：

```text
current R1Pro experience_system result
```

## 6. 已删除或排除的旧论文线文档

以下旧文档如果仍在历史记录或备份中出现，均视为排除出当前论文主线。
原因是它们描述的是 UR5e / Isaac / U1-U5 / `memory_v2/memory_v3` 线，
不再作为当前 `experience_system` 论文主线。

```text
论文/EXPERIMENT_PROTOCOL_CURRENT.md
论文/Isaac_R1Pro_经验库Benchmark计划.md
论文/Isaac_R1Pro异常处理与经验库实验计划.md
论文/Isaac_R1Pro迁移到MuJoCo计划.md
论文/MuJoCo_4异常4方法_10trial结果_v11.md
论文/U1_U2_U3_U4_实验结果与经验库设计整理.md
论文/U1_U2_U3_U4_实验结果文档.md
论文/U1_U2_U3_实验结果文档.md
论文/U1_U3_实验结果文档.md
论文/icra_style_paper_draft.md
论文/paper_icra.tex
论文/paper_icra*.docx/html
论文/25S103379-马兰萱-论文*.docx
论文/实验结果汇总_PPT版.md
论文/经验库优化文档.md
论文/经验库当前功能说明.md
论文/经验库整体介绍.md
galaxea_mujoco/docs/r1pro_sim_real_dual_source_memory_plan.md
galaxea_mujoco/docs/universal_experience_memory_current_implementation.md
galaxea_mujoco/docs/universal_sim_real_experience_memory_design.md
```

## 7. Research Background 文档

`经验库/` 目录下的论文阅读材料、PPT 材料和研究启发文档属于 background：

```text
经验库/Plan in Sandbox对Sim-Real双源经验库的启发.md
经验库/RoboCritics对Sim-Real双源经验库的启发.md
经验库/RAP对Sim-Real双源经验库的启发.md
经验库/Learning From Failure对Sim-Real双源经验库的启发.md
经验库/Worth Remembering对Sim-Real双源经验库的启发.md
经验库/机器人记忆与评测论文对Sim-Real双源经验库的综合启发.md
...
```

这些材料可以用于 Related Work 和方法动机，但不能写成实现结果。若要引用
其中的观点，必须回到原论文或当前实现证据，而不是引用读书笔记本身。

## 8. 论文写作边界

当前论文主线只能引用：

```text
experience_system/docs/*
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/*
```

旧 UR5e 结果只能作为历史背景，不能与当前 R1Pro `experience_system`
证据混合成同一个主实验口径。

推荐论文 evidence 引用顺序：

```text
1. paper_evidence_summary.json/md
2. paper_evidence_appendix.md/tex
3. paper_implementation_alignment.md
4. 具体 report JSON
```

## 9. 后续维护动作

新增任何经验库相关文档时，必须在本文档中分类：

```text
current
generated evidence
real-robot collection
auxiliary
legacy UR5e
research background
```

新增任何报告后，必须重新生成：

```text
paper_evidence_summary.json/md
paper_evidence_appendix.md/tex
```
