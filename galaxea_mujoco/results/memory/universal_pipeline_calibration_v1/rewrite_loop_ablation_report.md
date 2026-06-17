# Rewrite Loop Ablation Report

This report consolidates existing evidence for rewrite-loop components and explicitly marks missing sequential rewrite evidence.

## Summary

- Variant count: 5
- Status counts: `{"supported": 5}`
- Sequential rewrite reports: 1
- True sequential rewrite evidence: True

## Variants

| Variant | Status | Source | Key metrics | Boundary |
|---|---|---|---|---|
| single_plan_no_feedback | supported | `results/memory/universal_pipeline_calibration_v1/single_plan_real_llm_g3_clean_report.json` | `{"attempt_count": 1, "final_sandbox_status": "accept", "rewrite_rounds": 0}` | This variant does not test critic-driven rewriting because rewrite_rounds is zero. |
| sequential_critic_feedback_rewrite | supported | `results/memory/universal_pipeline_calibration_v1/rewrite_loop_g3_clean_real_report.json` | `{"attempt_count": 2, "final_sandbox_status": "reject", "rewrite_rounds": 1}` | If final_sandbox_status is not accept, this proves rewrite-loop mechanics, not successful recovery. |
| multi_candidate_critic_ranking | supported | `results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g3_clean_real_llm_report.json` | `{"final_sandbox_status": "accept", "sandboxed_plan_count": 2}` | This is not a sequential rewrite loop unless a report contains multiple rewrite attempts. |
| critic_plus_failure_memory | supported | `results/memory/universal_pipeline_calibration_v1/policy_baseline_vs_memory_mitigation_report.json` | `{"changed_rate": 0.375, "failure_evidence_count": 17}` | This report proves memory-aware ranking evidence, not a full LLM rewrite response to that evidence. |
| critic_failure_memory_parameter_priors | supported | `/tmp/field_atomic_ablation.json; results/memory/universal_pipeline_calibration_v1/field_atomic_trace_summary.json` | `{"parameter_changed_count": 0, "trace_count": 10, "warm_memory_count": 3, "warm_prior_action_count": 3}` | Dry-run field_atomic ablation does not prove parameter improvement unless a real LLM changes parameters. |

## Paper Wording

- Safe claim: Existing reports support single-plan validation, multi-candidate critic ranking, memory-backed risk evidence, field-atomic parameter-prior exposure, and a dry-run sequential critic-feedback rewrite attempt.
- Avoid claim: Do not claim critic-feedback rewriting improves recovery unless the underlying report has rewrite_rounds > 0 and final_sandbox_status=accept under a real LLM or clearly stated dry-run setting.
