# Experience-Guided Robot Anomaly Recovery with Sim-Real Dual-Source Memory

## Abstract

Large language models can propose robot recovery actions, but a plausible
plan is not necessarily executable, safe, or supported by prior experience.
We present `experience_system`, a Sim-Real dual-source experience memory
system for robot anomaly recovery. The system stores simulation, pseudo-real,
and real-format episodes in a unified schema and ranks recovery candidates
using memory support, Top-O failure risk, stage-specific retrieval evidence,
visual keyframes, text-semantic summaries, gap-calibrated sandbox rollout,
motion-level critic feedback, and LLM-generated recovery plans. Candidate
plans are semantically validated, shadow-executed in MuJoCo before selection,
critic-ranked, and exported only after structured validation. Current evidence covers 22
claim-level reports, with 21 supported claims and no missing reports. One
sensor-evidence claim remains partial until true real-robot sensor data is collected.
The system should be described as an implementation and evaluation
framework for evidence-guided anomaly recovery, not as proof of real-robot
success-rate improvement.

## 1. Introduction

Robot anomaly recovery is difficult because failures are often local,
physical, and history-dependent. A recovery plan that looks correct in text
may still repeat a known failure, ignore sim-real uncertainty, or trigger a
motion-level safety risk. This motivates a pipeline where memory and sandbox
evidence constrain planning before execution.

The current implementation treats recovery as a candidate selection problem:

```text
retrieve experience memory
-> construct planner input
-> generate single or multi-candidate LLM recovery plans
-> validate schema, allowed skills, and skill precondition/effect facts
-> shadow-roll out valid candidates in parallel MuJoCo sandbox workers
-> apply motion/contact/task critic checks
-> rank candidates by sandbox status, risk, and score
-> export a validated robot plan
```

The strongest safe claim is:

```text
The system ranks anomaly-recovery candidates using memory support, failure
risk, stage-specific retrieval evidence, gap-calibrated sandbox rollout, and
motion-level critic feedback.
```

## 2. Contributions

This implementation supports five paper-facing contributions:

1. A unified experience schema for simulation, pseudo-real, and real-format
   robot episodes.
2. Sim-real pair and gap representations that can calibrate sandbox rollout
   through object initial-state shifts and explicit risk penalties.
3. Candidate recovery selection through memory scoring, LLM generation,
   semantic validation, parallel sandbox rollout, critic evaluation, and ranking.
4. Multi-type robot memory evidence, including temporal, spatial, episodic,
   semantic, perceptual, and sim-real gap fields.
5. Reproducible evidence reports for ablation, safety stress, writeback,
   semantic retrieval, memory coverage, and paper appendices.

## 3. Related Work Positioning

This work is positioned between robot memory systems, LLM-based planning, and
simulation-based safety checking. Unlike a plain episode log, the experience
library stores retrieval keys, failure taxonomy, critic outputs, sim-real gap
signatures, sandbox calibration, and memory lifecycle fields. Unlike direct
LLM recovery, the generated plan is treated as a candidate that must pass
schema checks and can be shadow-executed before selection. Unlike a full
digital-twin approach, the current sandbox calibration is deliberately
limited: it shifts initial object state and contributes a risk penalty, but it
does not claim learned dynamics, friction, or contact calibration.

The legacy UR5e `memory_v2/memory_v3` line under `ur5e_mujoco`
should be treated as historical background. The current paper line is the
R1Pro `experience_system` implementation.

## 4. Method

### 4.1 Experience Schema

Each `ExperienceEntry` stores task and execution context, including:

- robot, scenario, condition, task, and anomaly descriptors
- skill sequence and action/observation traces
- sensor summary and optional raw references
- memory gate and lifecycle metadata
- critic result and failure taxonomy
- sim-real pair, sim-real gap, and sandbox calibration
- real-format episode references

This schema supports simulation and pseudo-real entries today, and it already
contains real-format fields for future robot episodes.

### 4.2 Memory Retrieval and Risk Scoring

Candidate plans are retrieved and scored through structured memory matching.
The scoring function combines support evidence with Top-O failure memory.
Failure memories are not used only as blockers; they contribute terminal risk
scores and risk penalties that affect candidate ranking.

### 4.3 Stage-Aware Planner Input

The system can build stage-aware planner context for:

- candidate generation
- candidate ranking
- sandbox rewrite
- execution writeback

For G4/place_occupied, the generated stage context reports
`stage_specificity_score_avg=0.7056`, `risk_evidence_count_total=2`, and
`critic_warning_count_total=4`. This should be described as deterministic
stage-specific evidence construction, not as a learned stage planner.

### 4.4 LLM Recovery Plans and Candidate Search

The LLM is configured through `experience_system/.env` and called by the
experience system itself. It receives allowed skill names, candidate steps,
and planner input. The output must be a structured JSON recovery plan. The
system validates:

- schema shape
- allowed skill names
- evidence id grounding
- confidence range

The plan is then checked by a skill precondition/effect graph before sandbox
rollout. The system supports both a single-plan validation loop and a
multi-candidate search loop. In the multi-candidate loop, the LLM returns a
`plans[]` array, valid plans are sandboxed in parallel subprocess workers, and
the selected plan is exported as `validated_robot_plan_v1`.

In a real Doubao G3/clean run, the LLM generated two distinct recovery plans:
one added `verify_grasp` before lift, and one added a second
`detect_place_occupancy` before place. Both passed semantic validation and
MuJoCo critic checks.

### 4.5 Sandbox Rollout and Critic

Candidate plans can be shadow-executed in MuJoCo. The sandbox report records
per-candidate success, critic status, sandbox score, and fused score. A
motion-level critic checks rule-based safety and task signals. Candidate
sweeps and LLM candidate search can run rollouts in subprocess workers so
different candidate/perturbation jobs do not share MuJoCo state.

In the current G4/place_occupied parallel sweep, four candidates and eight
rollouts were evaluated with four workers, `failed_worker_count=0`, and
`determinism_check_pass=true`.

### 4.6 Gap-Derived Calibration

Sim-real gap memories can produce sandbox calibration. The current supported
calibration changes object initialization and adds risk penalty. In the G4
calibration ablation, full calibration moves object start from
`[0.16, 0.0, 0.805]` to `[0.2, 0.04, 0.765]`, with
`object_start_delta_full_vs_no_calibration=0.12` and
`calibration_risk_penalty_full=0.12`.

### 4.7 Visual and Text-Semantic Retrieval

Visual keyframes are indexed and used as an auxiliary retrieval signal.
Text-semantic summaries are generated from explicit fields and indexed using
TF-IDF + FAISS. The text-semantic report covers 12 entries with
`semantic_summary_nonempty_rate=1.0`, `avg_token_count=148.75`, and
`semantic_signal_rate=1.0`. This is not a learned language encoder.

## 5. Experiments and Evidence

The evidence directory is:

```text
galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/
```

The current paper evidence summary reports:

- `claim_count=22`
- `supported_claim_count=21`
- `missing_report_count=0`
- one partial sensor-evidence claim

### 5.1 Memory Type Coverage

The memory type coverage report evaluates field-level evidence for six
memory types. The library contains 12 entries, with an average of 5.6667
covered memory types per entry. Eight entries cover all six memory types,
giving `entries_covering_all_memory_types_rate=0.6667`.

### 5.2 Candidate Sandbox Rollout

For G4/place_occupied, candidate sandbox rollout evaluates four candidates.
The selected candidate remains `g4_avoid_occupied_primary` before and after
sandbox. This supports the claim that candidates can be shadow-executed and
scored, but not that sandbox always changes selection.

### 5.3 Safety Stress

The standard safety stress report shows that sandbox critic identifies risky
candidates under stress variants. In the adversarial safety stress report,
artificially boosted risky candidates are selected by memory-only ranking in
four cases, and sandbox prevents risky selection in all four:

- `memory_only_risky_selected_count=4`
- `sandbox_prevented_risky_selection_count=4`
- `sandbox_prevented_risky_selection_rate=1.0`
- `selection_changed_by_sandbox_count=4`
- `sandbox_risky_selected_count=0`

This must be described as an adversarial ranking stress test.

### 5.4 Writeback

The repeated writeback benchmark runs three rounds on G3/clean:

- `round_count=3`
- `write_count=3`
- `entry_count_delta=2`
- `new_memory_retrieval_rate=1.0`
- `selected_candidate_change_rate=0.3333`
- `score_delta_after_writeback_avg=0.0273`
- `risk_delta_after_writeback_avg=-0.0471`

This supports the claim that new executions can be written back and retrieved
later. It does not prove statistically significant long-horizon improvement.

### 5.5 LLM + Sandbox End-to-End Smoke

Using `conda` environment `mujoco1`, the current system has been tested with:

```text
LLM recovery plan -> semantic validation -> MuJoCo sandbox rollout ->
critic pass -> validated robot plan
```

The G3/clean smoke produced:

- `recovery_plan_enabled=true`
- `sandbox_enabled=true`
- `sandbox_status=pass`
- `task_success=true`
- `sandbox_score=1.0`
- `decision=accept`

This verifies that the LLM integration is inside `experience_system`, not
through `ur5e_mujoco`.

### 5.6 LLM Multi-Candidate Search Ablation

The real LLM multi-candidate search report compares against a single-plan
real LLM baseline on G3/clean. Both variants succeed in this easy smoke, so it
should not be used as a success-rate improvement claim. Its value is showing
that the multi-candidate path evaluates more alternatives before selection:

- single-plan baseline: `plan_count=1`, `accepted_plan_count=1`
- multi-plan search: `plan_count=2`, `accepted_plan_count=2`
- `plan_count_delta=1`
- `accepted_plan_count_delta=1`
- `best_sandbox_score_delta=0.0`

Safe wording: multi-candidate search evaluates more LLM-generated recovery
alternatives than a single-plan baseline, runs valid candidates through
parallel sandbox rollout, and selects a critic-approved validated plan.

## 6. Claim Boundaries

Safe claims:

- Candidate plans are shadow-executed in MuJoCo and fused with memory scores
  before selection.
- A configured external LLM can generate multiple recovery-plan candidates
  that are semantically validated, sandboxed in parallel, critic-ranked, and
  exported as a dry-run dispatchable validated robot plan.
- Gap-derived calibration changes sandbox initialization and contributes an
  explicit risk penalty.
- Generated lessons are checked for evidence grounding, candidate/skill
  validity, concision, and internal conflicts.
- The library stores multiple memory types rather than only flat episode logs.

Claims to avoid:

- real-robot success-rate improvement
- full digital-twin calibration
- learned critic
- learned language embedding benchmark
- statistically proven long-horizon improvement
- arbitrary unseen-skill planning without skill metadata

## 7. Limitations

The current real branch is real-format and pseudo-real evidence, not a true
real-robot validation result. Sensor fields for RGB-D, lidar, and wrist force
are supported, but the current evidence summary reports zero true sensor
evidence entries. Real robot episodes must be imported before claiming
real-world success-rate improvement or sensor-derived calibration benefits.

The critic is rule-based. The sandbox is a pre-execution evaluation layer, not
a full digital twin. Text-semantic retrieval uses explicit TF-IDF + FAISS
summaries, not learned neural embeddings.

## 8. Reproducibility

Primary generated evidence:

- `paper_evidence_summary.json`
- `paper_evidence_appendix.md`
- `paper_evidence_appendix.tex`
- `llm_plan_candidate_search_g3_clean_real_llm_report.json`
- `llm_plan_search_ablation_g3_clean_real_llm.json`
- `memory_type_coverage_report.json`
- `text_semantic_memory_report.json`
- `sandbox_rollout_g4_place_occupied.json`
- `sandbox_calibration_ablation_g4_place_occupied.json`
- `safety_stress_g4_place_occupied.json`
- `harder_safety_stress_g4_place_occupied.json`
- `writeback_benchmark_g3_clean.json`

The document lineage and claim boundaries are maintained in:

- `experience_system/docs/document_lineage_index.md`
- `experience_system/docs/paper_implementation_alignment.md`
- `experience_system/docs/paper_rewrite_alignment_plan.md`
