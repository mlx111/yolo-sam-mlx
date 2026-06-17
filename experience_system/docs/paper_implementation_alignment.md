# Paper-Implementation Alignment for the Experience System

This document maps the current paper/design claims to the actual
`experience_system` implementation. Use it as the source of truth when deciding
what can be claimed as implemented, what needs more experiments, and what should
remain future work.

## Current Paper Position

The root implementation is no longer only a UR5e `memory_v2` positive/negative
memory system. It is now closer to:

```text
Sim-Real dual-source experience memory for anomaly recovery
with candidate sandbox rollout, critic scoring, and reportable ablations.
```

The older paper draft under `论文/icra_style_paper_draft.md` still describes a
UR5e-focused `memory_v2` system. If that paper is retained, the new
`experience_system` should be framed as an extended implementation/evaluation
section, not silently mixed into the older claims.

## Claim Matrix

| Paper/design claim | Implementation status | Code evidence | Experiment/report evidence | Safe paper wording |
| --- | --- | --- | --- | --- |
| Unified experience schema for simulation, pseudo-real, and real episodes | Implemented | `experience_core/schema.py`, `experience_adapters/r1pro_mujoco.py`, `experience_adapters/real_episode.py`, `experience_adapters/wrapper1_ur5e.py` | `run_universal_memory_pipeline.py`, `summarize_universal_experience.py` | "We implement a unified schema and adapters for simulation, pseudo-real, and real-format episodes." |
| Experience write gating inspired by Worth Remembering | Implemented | `experience_core/gating.py`, `experience_core/write_policy.py`, `ExperienceLibrary.add_with_policy()` | Pipeline import reports include write-policy decisions | "We use an engineering write gate based on anomaly, failure, gap, utility, and surprise proxies." Do not claim learned Bayesian surprise. |
| Sim-Real pair and gap representation | Implemented | `experience_core/dual_source.py`, `ExperienceEntry.sim_real_pair`, `ExperienceEntry.sim_real_gap` | `build_universal_sim_real_pairs.py`, pipeline calibration reports | "The system computes pair-level sim-real/pseudo-real gap signatures." |
| Sandbox calibration from gap memory | Implemented as risk calibration | `experience_core/calibration.py`, `build_universal_calibration.py`, `SandboxCalibration` fields, `candidate_sandbox.py::calibration_risk` | Calibrated rollout reports include `calibration_applied`, `calibration_id`, and `calibration_risk_penalty` | "Gap memories are converted into sandbox calibration parameters that penalize risky rollouts." Avoid claiming full dynamics or scene-twin calibration. |
| Calibration is applied inside candidate sandbox rollout | Implemented for object initial pose and risk score | `candidate_sandbox.py::evaluate_candidate_in_sandbox`, `run_r1pro_task_chain.py::run_task_chain(..., sandbox_calibration=...)`, `run_candidate_sandbox_rollout.py --use-sandbox-calibration`, `run_memory_ablation_report.py --use-sandbox-calibration` | G4/place_occupied calibrated rollout applies `object_pose_bias=[0.04,0.04,-0.04]`, moving object start from `[0.16,0.0,0.805]` to `[0.2,0.04,0.765]`, and reports `calibration_risk_penalty=0.12` | "Gap-derived calibration is consumed both as a sandbox risk penalty and as an object initial-state shift in sandbox rollout." Avoid claiming full dynamics calibration. |
| Calibration ablation | Implemented for no/score-only/pose+score variants | `tools/run_sandbox_calibration_ablation.py`, `source/run_sandbox_calibration_ablation.py` | G4/place_occupied `g4_avoid_occupied_primary`: score-only keeps object start `[0.16,0.0,0.805]` and applies raw score delta `-0.12`; pose+score shifts object start to `[0.2,0.04,0.765]` with `object_start_delta=0.12` and raw score delta `-0.12`. Normalized sandbox score stays `1.0` because it saturates. | "Gap-derived calibration changes sandbox initialization and contributes an explicit risk penalty." Do not claim dynamics/friction/contact calibration. |
| Candidate plan -> sandbox rollout -> critic -> select | Implemented | `tools/candidate_sandbox.py`, `tools/run_candidate_sandbox_rollout.py`, `tools/run_r1pro_memory_policy_smoke.py --use-sandbox-rollout` | `/tmp/lesson_sandbox_rollout_g4_check3.json`, ablation full reports | "Candidate plans are shadow-executed in MuJoCo and fused with memory scores before selection." |
| Motion-level critic similar to RoboCritics | Implemented as rule critic | `experience_core/critic.py`, `run_r1pro_task_chain.py` motion probes, `g4_fast_transport` risky candidate | Full ablation with `critic_warn_rate=0.2` when risky candidate is included | "We implement rule-based motion-level critics for speed, step, workspace, and pose risks." Do not claim learned critic. |
| Safety stress benchmark for risky candidates | Implemented | `tools/run_memory_safety_stress_report.py`, `g4_fast_transport`, sandbox critic reports | G4/place_occupied stress: `risky_candidate_selected_rate=0.0`, `safe_candidate_selected_rate=1.0`; sandbox variants detect the risky candidate with `risky_warn_or_block_rate=1.0`, `critic_warn_rate_avg=0.2`, and increase selected-vs-risky score margin from 0.1785 to 0.3771 | "The stress report shows that sandbox critic identifies risky candidates and increases the safety margin." Do not claim it changed selection in this run because memory-only already selected a safe candidate. |
| Harder adversarial safety stress | Implemented | `tools/run_harder_safety_stress_cases.py`, `source/run_harder_safety_stress_cases.py` | G4/place_occupied adversarial stress: `case_count=4`, `memory_only_risky_selected_count=4`, `sandbox_prevented_risky_selection_count=4`, `sandbox_prevented_risky_selection_rate=1.0`, `selection_changed_by_sandbox_count=4`, `sandbox_risky_selected_count=0`; all cases redirect `g4_fast_transport -> g4_avoid_occupied_primary` | "Under adversarial ranking stress with artificially boosted risky candidates, sandbox critic can redirect selection away from the risky candidate." Must state artificial/adversarial perturbation. |
| Top-O failure memory as risk prior | Implemented | `experience_core/scoring.py::top_failure_risk_memory`, `score_candidate_plan()` | Candidate reports include `top_failure_risks`, `terminal_risk_score`, `failure_risk_penalty` | "Top critical failures are retrieved and converted into candidate risk penalties." |
| Failure memories are not blockers only; they affect ranking | Implemented | `failure_risk_penalty`, `risk_score`, `decision` in `score_candidate_plan()` | Policy smoke and sandbox reports expose candidate risk deltas | Safe to claim. |
| LLM-generated compact lessons/rules from evidence | Implemented | `tools/generate_experience_lessons_llm.py`, `experience_core/lessons.py`, `tools/merge_experience_lessons.py` | `llm_experience_lessons_g4_place_occupied.json`, merged lesson smoke | "An LLM can synthesize compact lessons from sandbox/policy evidence, and these lessons adjust candidate scores." |
| Lessons are non-template and directly usable by policy | Implemented with quality audit | Generator prompt forbids deterministic templates; `adjust_candidate_with_lessons()` parses action-order evidence; `tools/analyze_experience_lessons.py` audits generated lesson quality | Merged lesson policy smoke selects `g4_avoid_occupied_primary` with score increase. G4/place_occupied quality report: `quality_pass=true`, `lesson_count=1`, `evidence_id_valid_rate=1.0`, `candidate_id_valid_rate=1.0`, `skill_reference_valid_rate=1.0`, `conflict_pair_count=0`, `template_like_phrase_count=0`, `actionable_lesson_rate=1.0`, `confidence_avg=0.8` | "Generated lessons are checked for evidence grounding, candidate/skill validity, concision, and internal conflicts before policy adjustment." Safe for generated lessons, not arbitrary free-form rules. |
| Stage-aware retrieval from RAP | Implemented as ranking evidence, staged report, and planner context | `RetrievalQuery.task_stage`, `experience_core/stage_retrieval.py`, `experience_core/stage_prompt.py`, `tools/analyze_stage_aware_retrieval.py`, `tools/render_stage_planner_context.py`, `run_r1pro_memory_policy_smoke.py --use-stage-retrieval --render-stage-context`, `run_candidate_sandbox_rollout.py --use-stage-retrieval` | G4/place_occupied policy/sandbox report: 4 stage reports per candidate, `stage_specificity_score=0.7056`, `mean_stage_overlap=0.2944`, stage support/risk adjusts candidate score before sandbox fusion. Standalone planner context: `stage_context_distinct_memory_count_avg=6.0`, `risk_evidence_count_total=2`, `critic_warning_count_total=4`. Policy context over 4 candidates: `context_count=4`, `risk_evidence_count_total=8`, `critic_warning_count_total=16`. | "The retrieval system applies stage-specific query policies and renders planner context separating positive examples, risk priors, critic evidence, and writeback histories." Do not claim a learned stage planner. |
| Visual keyframe memory | Implemented with candidate-ranking ablation | `experience_core/visual_retrieval.py`, `load_visual_scores()`, `tools/run_visual_retrieval_ablation.py` | Visual index covers 12 entries / 60 keyframes. G3/clean ablation: `visual_score_count=6`, selected candidate changes `g3_default -> g3_place_first`, `retrieval_changed_rate=0.3333`, `candidate_score_delta_max=0.0409`. G4/place_occupied ablation: `visual_score_count=8`, selection unchanged, `candidate_score_delta_avg=0.0882`, `candidate_score_delta_max=0.1396`. | "Visual keyframes are indexed and used as an auxiliary retrieval signal during candidate ranking." Avoid claiming broad multimodal reasoning. |
| Closed-loop writeback after execution | Repeated benchmark implemented | `run_r1pro_memory_policy_smoke.py --write-experience`, `ExperienceLibrary.add_with_policy()`, `tools/run_memory_writeback_demo.py`, `tools/run_memory_writeback_benchmark.py` | G3/clean repeated benchmark with 3 rounds: `write_count=3`, `entry_count_delta=2`, `new_memory_retrieval_rate=1.0`, selected candidate changed in 1/3 rounds, score delta avg +0.0273, risk delta avg -0.0471. G4/place_occupied sandbox demo confirms selected candidate and retrieves new memory. | "Newly executed experiences can be repeatedly written back and retrieved in later ranking passes." Avoid claiming statistically proven long-horizon success-rate improvement. |
| Real robot experience branch | Real-format and pseudo-real evidence pack implemented; true real data still limited | `experience_adapters/real_episode.py`, `real_episode_ref`, `source=real`, `tools/build_real_format_evidence_pack.py` | `real_format_evidence_pack.json`: `real_format_entry_count=8`, `pseudo_real_entry_count=8`, `sim_entry_count=4`, `paired_gap_count=4`, `sim_real_gap_count=4`, `calibration_id_count=4`, `calibration_with_object_pose_bias_count=4`, `sandbox_calibration_application_count=4` | Say "real-format import and pseudo-real branches are supported, and pseudo-real evidence exercises pairing, gap extraction, and sandbox calibration." Avoid claiming real-robot validation unless data/report exists. |
| Paper-oriented ablation protocol | Implemented | `tools/run_memory_ablation_report.py`, `configs/ablation_r1pro_paper_v1.json` | JSON/CSV/Markdown/LaTeX via `summarize_memory_ablation_report.py` | Safe to claim as evaluation tooling. |
| Paper evidence summary | Implemented | `tools/build_paper_evidence_summary.py`, `source/build_paper_evidence_summary.py` | Latest regenerated summary reports `claim_count=22`, `supported_claim_count=21`, `missing_report_count=0`; rows cover schema, real-format/pseudo-real evidence, sensor evidence, sandbox rollout, stage context, visual/text retrieval, calibration, lesson audit, safety stress, writeback, field_atomic memory, physical perturbation, physical-default audit, runtime scene schema, write-policy audit, and write-policy pressure cases. The sensor row remains partial until true real-format sensor coverage is available. | Safe to use as a paper claim/evidence/boundary table. Do not treat it as new experimental evidence beyond the underlying reports. |
| Write-policy pressure cases | Implemented | `tools/run_write_policy_pressure_test.py`, `experience_core/write_policy.py` | Deterministic pressure test: `decision_counts={"write": 3, "merge": 1, "skip": 1, "reject": 1}` with reasons for preserved failure, field_atomic preservation, duplicate low-risk success, low-value success, and missing required fields. | "The write policy supports auditable write, merge, skip, and reject lifecycle decisions." Do not claim the handcrafted cases prove an optimal memory policy. |
| Success-rate improvement over baselines | Not yet established | Tooling exists | Current smoke examples show success but not statistically meaningful improvement | Do not claim success-rate gains unless full repeated ablations show them. Emphasize risk reduction, candidate change, critic warning/block, and evidence chains. |

## Recommended Claim Boundaries

### Strong Claims

These are implemented and have direct reportable evidence:

- A unified `ExperienceEntry` schema covers simulation, pseudo-real, real-format
  episodes, critic output, sim-real gap, calibration, retrieval keys, and memory
  tags.
- Candidate recovery plans can be ranked by memory support, Top-O failure risk,
  LLM lesson adjustment, and sandbox rollout score.
- Sandbox rollout produces per-candidate critic results and can expose risky
  candidates through motion-level warning flags.
- LLM-generated lessons can be stored as a reusable lesson library and used to
  adjust candidate scores.
- Newly executed experiences can be written through the memory gate and
  retrieved by the next ranking pass in a closed-loop demo.
- Ablation reports can be generated as JSON, CSV, Markdown, and LaTeX tables.

### Moderate Claims

These are implemented as interfaces or first-pass rules, but need careful
wording:

- Sim-real gap calibration is consumed by sandbox scoring and object initial
  pose initialization, but it should not be described as full dynamics or
  contact-parameter calibration.
- Stage-aware retrieval is integrated into candidate ranking as a rule-based
  evidence layer, but not implemented as a learned multi-stage planner.
- Closed-loop writeback has smoke-demo evidence, but not yet repeated ablation
  evidence for statistically meaningful success-rate improvement.

### Claims to Avoid for Now

- "Real robot experiments prove improved success rate."
- "The sandbox is a full digital twin."
- "The critic is learned or equivalent to RoboCritics."
- "Stage-aware RAP-style retrieval is fully implemented end-to-end."
- "The system reduces sim-real gap through closed-loop real execution" unless a
  writeback/gap-update experiment is added.

## Evidence Commands

Use these commands to regenerate the strongest current evidence.

### Candidate Sandbox Rollout

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_candidate_sandbox_rollout.py \
  --scenario G4 \
  --condition place_occupied \
  --include-risky-candidates \
  --universal-experience-lib results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --use-sandbox-calibration \
  --save /tmp/lesson_sandbox_rollout_g4_check.json
```

Expected evidence:

```text
selected_after_sandbox = g4_avoid_occupied_primary
critic_status_counts includes warn when risky candidate is enabled
g4_fast_transport receives sandbox warn/review
calibration_applied = true
calibration_risk_penalty is reported per candidate
sandbox_calibration_effect.applied = true
object_start reflects calibrated_object_start
```

### Ablation With Selected-Candidate Execution

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_memory_ablation_report.py \
  --config ../experience_system/configs/ablation_r1pro_paper_v1.json
```

Expected evidence:

```text
success_rate
candidate_changed_rate
risk_score_delta_avg
critic_block_rate
critic_warn_rate
repeated_failure_rate_avg
retrieval_count_delta_avg
```

### Paper Tables

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system python -B \
  ../experience_system/tools/summarize_memory_ablation_report.py \
  --input results/memory/universal_pipeline_calibration_v1/ablation_r1pro_paper_v1.json \
  --save-md results/memory/universal_pipeline_calibration_v1/ablation_r1pro_paper_v1.md \
  --save-tex results/memory/universal_pipeline_calibration_v1/ablation_r1pro_paper_v1.tex
```

### Lesson Library Maintenance

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system python -B \
  ../experience_system/tools/merge_experience_lessons.py \
  --input-glob 'results/memory/**/*lessons*.json' \
  --min-confidence 0.7 \
  --save results/memory/universal_pipeline_calibration_v1/experience_lessons_library.json
```

### Stage-Aware Retrieval Analysis

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system python -B \
  ../experience_system/tools/analyze_stage_aware_retrieval.py \
  --input results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --scenario G4 \
  --condition place_occupied \
  --candidate-id g4_avoid_occupied_primary \
  --save /tmp/stage_aware_g4_place_occupied.json \
  --save-csv /tmp/stage_aware_g4_place_occupied.csv
```

Expected evidence:

```text
stage_specificity_score is high when stages retrieve distinct memories
candidate_generation emphasizes validated memories
candidate_ranking emphasizes gap/risk memories
sandbox_rewrite emphasizes critic block cases
```

### Stage-Aware Ranking and Sandbox Rollout

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_candidate_sandbox_rollout.py \
  --scenario G4 \
  --condition place_occupied \
  --include-risky-candidates \
  --use-stage-retrieval \
  --use-sandbox-calibration \
  --universal-experience-lib results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save /tmp/stage_sandbox_g4.json
```

Expected evidence:

```text
stage_retrieval_enabled = true
stage_count = 4
stage_specificity_score = 0.7056
mean_stage_overlap = 0.2944
candidate scores include stage_support_score and stage_risk_score
sandbox critic still reports warn for risky candidate when included
```

### Safety Stress Report

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_memory_safety_stress_report.py \
  --scenario G4 \
  --condition place_occupied \
  --use-sandbox-calibration \
  --universal-experience-lib results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save /tmp/safety_stress_g4_place_occupied.json \
  --save-csv /tmp/safety_stress_g4_place_occupied.csv
```

Expected evidence:

```text
memory_only risky_candidate_selected_rate = 0.0
memory_sandbox_critic risky_warn_or_block_rate = 1.0
memory_sandbox_critic critic_warn_rate_avg = 0.2
memory_sandbox_critic score_margin_selected_vs_best_risky_avg = 0.3771
full_stage_lesson_sandbox stage_specificity_score_avg = 0.7056
```

### Closed-Loop Writeback Demo

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_memory_writeback_demo.py \
  --scenario G3 \
  --condition clean \
  --universal-experience-lib results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --save /tmp/writeback_demo_g3_clean.json
```

Expected evidence:

```text
entry_count_delta = 1
new_memory_retrieved = true
selected_candidate_changed = true
score_delta_after_writeback = 0.0818
risk_delta_after_writeback = -0.1415
```

Sandbox-calibrated G4 confirmation run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_memory_writeback_demo.py \
  --scenario G4 \
  --condition place_occupied \
  --include-risky-candidates \
  --use-sandbox-rollout \
  --use-sandbox-calibration \
  --universal-experience-lib results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save /tmp/writeback_demo_g4_place_occupied_sandbox.json
```

Expected evidence:

```text
entry_count_delta = 1
new_memory_retrieved = true
selected_candidate_changed = false
retrieval_overlap_before_after = 0.8571
critic_warn_rate = 0.2 when risky candidate is included
```

Repeated writeback benchmark:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../experience_system conda run -n mujoco1 python -B \
  ../experience_system/tools/run_memory_writeback_benchmark.py \
  --scenario G3 \
  --condition clean \
  --rounds 3 \
  --universal-experience-lib results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --save /tmp/writeback_benchmark_g3_clean.json
```

Expected evidence:

```text
round_count = 3
write_count = 3
entry_count_delta = 2
new_memory_retrieval_rate = 1.0
selected_candidate_change_rate = 0.3333
selected_candidate_confirmation_rate = 0.6667
score_delta_after_writeback_avg = 0.0273
risk_delta_after_writeback_avg = -0.0471
```

## Highest-Value Remaining Gaps

For the complete remaining roadmap, see
`docs/remaining_experience_system_roadmap.md`.

### 1. Full G4 Repeated Writeback Long Run

Current state:

- G3 repeated writeback benchmark is implemented and verified.
- G4 full stage+sandbox+calibration repeated benchmark is slower because each
  round runs before/after sandbox rollout for all candidates.

Minimal next implementation:

- Run the long G4 benchmark when enough time is available and report:

```text
new_memory_retrieval_rate
selected_candidate_change_rate
risk_delta_after_writeback
score_delta_after_writeback
memory_growth_count
```

## Suggested Paper Structure Update

If the paper is updated to the new system, structure the method section as:

1. Universal experience representation.
2. Dual-source pair/gap memory.
3. Candidate scoring with Top-O failure risk.
4. Sandbox rollout and motion-level critic.
5. LLM lesson abstraction and reuse.
6. Closed-loop writeback support.
7. Ablation protocol and metrics.

If the paper remains UR5e `memory_v2`, keep the `experience_system` as an
extended R1Pro/Sim-Real prototype and do not mix its claims into the UR5e
results section without separate experiments.
