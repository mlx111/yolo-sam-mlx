# 经验库旧文档状态说明

本目录主要保存论文阅读笔记、早期研究方案和旧 UR5e/experiment-sim-wrapper 线设计文档。

当前可复现实现主线已经迁移到：

```text
experience_system/
galaxea_mujoco/
```

当前论文证据链以以下文件为准：

```text
experience_system/docs/paper_experience_system.md
experience_system/docs/paper_rewrite_alignment_plan.md
experience_system/docs/sandbox_fidelity_optimization_roadmap.md
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_summary.json
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_appendix.md
```

## 文档分流

### 当前实现证据

不要从本目录直接引用“已实现结果”。当前实现证据应引用 `experience_system/docs` 和
`results/memory/universal_pipeline_calibration_v1/` 下的报告。

### 论文启发材料

以下文档可作为 related work / design rationale 的阅读笔记使用：

```text
Learning From Failure对Sim-Real双源经验库的启发.md
Plan in Sandbox对Sim-Real双源经验库的启发.md
RAP对Sim-Real双源经验库的启发.md
RISE对Sim-Real双源经验库的启发.md
Reconciling Reality through Simulation对Sim-Real双源经验库的启发.md
RoboCritics对Sim-Real双源经验库的启发.md
Robotic Sim-to-Real Transfer对Sim-Real双源经验库的启发.md
Worth Remembering对Sim-Real双源经验库的启发.md
机器人记忆与评测论文对Sim-Real双源经验库的综合启发.md
```

### 旧设计/研究方案

以下文档包含早期目标、旧 UR5e 基线或真机愿景，不能直接当作当前实现证据：

```text
双源经验库与沙盒推演研究方案.md
Sim-Real双源经验库开发基线设计文档.md
Sim-Real双源经验库沙盒推演实现方案.md
Sim-Real双源经验库研究方向创新性分析.md
Sim-Real双源经验库缺口分析与升级路线.md
```

这些文档中关于“提升真实环境异常恢复成功率”“真机执行前下发”“真机闭环回写”的表述，
在没有真实机器人 episode 报告前，只能作为研究目标或未来工作，不能写成已验证结论。

## 当前安全主线

当前可以安全写：

```text
The system supports Sim-Real dual-source experience memory, real-format episode
fields, LLM multi-candidate recovery-plan generation, skill semantic validation,
parallel MuJoCo sandbox rollout, critic-based selection, and validated robot-plan
export for dry-run dispatch.
```

当前不能写：

```text
real-robot success-rate improvement
full digital twin
learned critic
sensor-derived calibration validated on real robot
arbitrary unseen-skill planning without metadata
```
