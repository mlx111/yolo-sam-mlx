# LLM Plan Search Ablation

- Single-plan report: `results/memory/universal_pipeline_calibration_v1/single_plan_real_llm_g3_clean_report.json`
- Multi-plan report: `results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g3_clean_real_llm_report.json`

| Variant | Plans | Sandboxed | Accepted | Review | Rejected | Best score | Unique step sequences | Failed workers | Final status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| single_plan_real_llm | 1 | 1 | 1 | 0 | 0 | 1.0 | 1 | 0 | accept |
| multi_plan_real_llm_parallel_sandbox | 2 | 2 | 2 | 0 | 0 | 1.0 | 2 | 0 | accept |

## Comparison

```json
{
  "plan_count_delta": 1,
  "sandboxed_plan_count_delta": 1,
  "accepted_plan_count_delta": 1,
  "best_sandbox_score_delta": 0.0,
  "unique_step_sequence_delta": 1,
  "single_final_status": "accept",
  "multi_final_status": "accept"
}
```

Safe claim: The multi-candidate search evaluates more LLM-generated recovery alternatives than a single-plan baseline, runs valid candidates through parallel sandbox rollout, and selects a critic-approved validated plan.

Avoid claim: Do not claim a success-rate improvement from this small G3 clean ablation alone; both variants succeeded in the current smoke.
