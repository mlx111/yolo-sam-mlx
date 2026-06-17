# Current Experience-System Flow

```text
Unified Experience Memory
  simulation / pseudo-real / real-format
  failure risk + Top-O memory
  sim-real gap + calibration
  visual / text-semantic evidence
        |
        v
Stage Planner Context
  generation evidence
  ranking risk priors
  sandbox rewrite feedback
  writeback history
        |
        v
LLM Candidate Planner
  single-plan loop
  multi-candidate plans[] search
  allowed skill names only
  structured JSON recovery_plan
        |
        v
Semantic Validator
  schema checks
  evidence-id grounding
  skill requires/effects facts
  not fixed to G3/G4 order rules
        |
        v
Parallel MuJoCo Sandbox
  subprocess rollout workers
  candidate x perturbation jobs
  gap-calibrated initial state
        |
        v
Motion / Contact Critic
  task success and sandbox score
  collision / joint / pose risk
  contact loss and slip proxies
        |
        v
Validated Robot Plan
  validated_robot_plan_v1
  dry-run executor boundary
  writeback after execution
```

Evidence reports:

```text
paper_evidence_summary: 15 claims, 14 supported, 1 partial sensor-evidence row
llm_plan_candidate_search_g3_clean_real_llm_report.json
llm_plan_search_ablation_g3_clean_real_llm.json
sandbox_sweep_g4_place_occupied_parallel_report.json
```
