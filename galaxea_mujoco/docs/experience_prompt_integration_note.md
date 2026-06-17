# 经验如何影响 LLM 计划

## 结论

经验库不是直接替 LLM 决策，而是先整理成结构化 `planner_input`，再进入 prompt 和后续打分流程。

## 作用链路

```text
经验库检索
-> 结构化 planner_input
-> 写入 LLM prompt
-> 生成候选异常处理计划
-> 语义校验
-> 沙盒推演
-> critic 打分
-> 输出 validated_robot_plan
```

## 第一层：构造 planner_input

经验库先把相关经验整理成可读、可控的结构，而不是把整条历史直接塞给 LLM。

常见字段：

```json
{
  "positive_memory_ids": ["exp_success_001"],
  "risk_memory_ids": ["exp_fail_003"],
  "gap_memory_ids": ["exp_gap_002"],
  "recommended_steps": ["verify_grasp", "choose_alternate_place"],
  "risk_notes": ["grasp slip occurred when contact was weak"],
  "sim_real_gap": {
    "object_pose_bias": [0.02, 0.00, 0.00]
  }
}
```

这一层的目标是把经验变成“可供 LLM 使用的摘要”。

## 第二层：写入 prompt

`planner_input` 会进入 LLM 的恢复计划 prompt。LLM 根据这些经验生成异常处理技能序列。

LLM 看到的不是原始日志，而是：

- 成功经验
- 失败经验
- 风险提示
- 推荐动作
- 真机/仿真差异

LLM 的任务是输出一个新的结构化计划，而不是复述历史。

## 第三层：参与打分和约束

经验不只写进 prompt，还会直接影响后续判断：

- 失败经验会变成风险惩罚
- 真机/仿真差异会变成 sandbox calibration
- 视觉经验会影响候选排序
- stage 经验会影响分阶段 prompt/context
- critic 会参考经验里的风险模式做 reject/review

也就是说，经验库是“提示 + 约束 + 校准 + 评分”四种作用一起发挥。

## 为什么不能只靠 prompt

如果只把经验拼进 prompt，问题会很多：

- 计划不可控
- 风险约束不稳定
- 真机/仿真差异无法显式建模
- LLM 容易忽略少数关键失败经验

所以当前设计不是“经验直接驱动动作”，而是：

```text
结构化经验 + prompt + 语义校验 + 沙盒验证 + critic
```

## 真机和仿真经验的区别

两类经验都会进入同一个经验库，但作用略有不同：

- 真机经验更接近真实执行，优先级通常更高
- 仿真经验覆盖更多异常和组合情况，适合补充策略空间
- sim-real gap 会把仿真经验修正成更接近真实机的版本

## 最终输出

经验库影响 LLM 的最终结果，不是输出一段自然语言，而是输出：

```text
validated_robot_plan
```

也就是经过语义校验、沙盒验证、critic 过滤后的可执行计划。

## 一句话总结

经验库的作用不是“直接控制机器人”，而是先把过去经验提炼成结构化提示和约束，再帮助 LLM 生成更可靠的异常处理计划。
