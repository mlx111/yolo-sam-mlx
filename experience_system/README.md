# Experience System

Project-level experience-memory system migrated from `galaxea_mujoco`.

## Layout

- `experience_core/`: canonical universal experience-memory core package.
- `experience_adapters/`: adapters for R1Pro MuJoCo, real episodes, and wrapper1 UR5e.
- `tools/`: CLI scripts for calibration, retrieval, sandbox rollout, policy smoke, LLM lesson generation, and ablation reporting.
- `source/`: compatibility namespace so migrated tools can keep historical `source.*` imports.
- `docs/` and `configs/`: experience-memory design docs and pipeline configs.

For paper writing, use
`docs/paper_implementation_alignment.md` to separate implemented claims,
experimental evidence, and future-work items.
Use `docs/remaining_experience_system_roadmap.md` for the remaining
implementation roadmap and next evidence-producing experiments.
Use `docs/paper_experience_system.md` as the current paper-body draft for the
R1Pro `experience_system` line.

Legacy transitional packages (`core/`, `adapters/`, and `memory/`) have been
removed from `experience_system`; use the canonical package names above.

## Usage

Run tools from the repository root with `experience_system` on `PYTHONPATH`:

```bash
PYTHONPATH=experience_system python experience_system/tools/run_candidate_sandbox_rollout.py --help
```

The current R1Pro memory-policy loop is:

```text
retrieve experience memory
-> score candidate plans with success and Top-O failure risk evidence
-> optionally apply compact LLM lessons
-> optionally roll out candidates in sandbox
-> critic scores sandbox trajectories
-> fuse memory and sandbox scores for final selection
```

## LLM Settings

Experience-system LLM calls are configured by `experience_system/.env`, not via
`ur5e_mujoco`. Shell environment variables with the same names take
precedence over `.env` values.

Recommended shared variables:

```bash
EXPERIENCE_LLM_API_KEY=...
EXPERIENCE_LLM_BASE_URL=...
EXPERIENCE_LLM_MODEL=...
EXPERIENCE_LLM_TIMEOUT=120
```

Generate a paper-oriented ablation report:

```bash
PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_memory_ablation_report.py \
  --scenario G4 \
  --condition place_occupied \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --include-risky-candidates \
  --use-sandbox-calibration \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/ablation_report.json \
  --save-csv galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/ablation_report.csv
```

Run the reproducible paper config, including selected-candidate execution for
real `success_rate`:

```bash
PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_memory_ablation_report.py \
  --config experience_system/configs/ablation_r1pro_paper_v1.json
```

The ablation tool reports these variants by default: `baseline_no_memory`,
`memory_only`, `memory_lifecycle`, and `full_visual_sandbox_critic`. It writes
JSON plus optional CSV rows with `success_rate`, `candidate_changed_rate`,
`risk_score_delta`, `critic_block_rate`, `critic_warn_rate`,
`repeated_failure_rate`, `memory_write_count`, and retrieval-count statistics.
When `execute_selected` is enabled in the config, each final selected candidate
is executed once and the result is stored under `execution_report`.
When `use_sandbox_calibration` is enabled, matching gap-derived
`SandboxCalibration` is consumed by sandbox rollout initialization and scoring:
`object_pose_bias` shifts the target object's initial pose, while contact/slip
gap terms contribute `calibration_risk_penalty`.

Convert an ablation JSON report into Markdown and LaTeX tables:

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/summarize_memory_ablation_report.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/ablation_r1pro_paper_v1.json \
  --save-md galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/ablation_r1pro_paper_v1.md \
  --save-tex galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/ablation_r1pro_paper_v1.tex
```

Merge multiple LLM-generated lesson files into a reusable lesson library:

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/merge_experience_lessons.py \
  --input-glob 'galaxea_mujoco/results/memory/**/*lessons*.json' \
  --min-confidence 0.7 \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/experience_lessons_library.json
```

Audit generated lesson quality:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/analyze_experience_lessons.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/lesson_quality_report_g4_place_occupied.json
```

The audit checks evidence ids, candidate ids, skill references, concision,
duplicates, template-like phrases, and internal conflicts before lessons are
used for policy adjustment.

Build a paper evidence summary table:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/build_paper_evidence_summary.py \
  --report-dir galaxea_mujoco/results/memory/universal_pipeline_calibration_v1 \
  --save-json galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_summary.json \
  --save-md galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/paper_evidence_summary.md
```

The summary consolidates each paper claim with its primary report, key metrics,
safe wording, and wording to avoid. It is an evidence index over existing
reports, not a new experiment. The default summary uses persisted reports under
`galaxea_mujoco/results/memory/universal_pipeline_calibration_v1`; `/tmp`
reports are only fallback inputs.

Analyze stage-aware retrieval behavior without running simulation:

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/analyze_stage_aware_retrieval.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --scenario G4 \
  --condition place_occupied \
  --candidate-id g4_avoid_occupied_primary \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/stage_aware_retrieval_g4_place_occupied.json \
  --save-csv galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/stage_aware_retrieval_g4_place_occupied.csv
```

Run stage-aware retrieval inside candidate ranking and sandbox rollout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_candidate_sandbox_rollout.py \
  --scenario G4 \
  --condition place_occupied \
  --include-risky-candidates \
  --use-stage-retrieval \
  --use-sandbox-calibration \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save /tmp/stage_sandbox_g4.json
```

With `--use-stage-retrieval`, each candidate receives four stage-specific
retrieval reports: `candidate_generation`, `candidate_ranking`,
`sandbox_rewrite`, and `execution_writeback`. The stage support/risk summary is
used as a small score adjustment before sandbox fusion and is saved under
`stage_retrieval_summary`.

Render stage-aware planner context from the same stage retrieval evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/render_stage_planner_context.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --scenario G4 \
  --condition place_occupied \
  --candidate-id g4_avoid_occupied_primary \
  --save /tmp/stage_planner_context_g4.json \
  --save-text /tmp/stage_planner_context_g4.txt
```

Add `--render-stage-context` to `run_r1pro_memory_policy_smoke.py` together
with `--use-stage-retrieval` to attach `stage_planner_context` to every
candidate report. The context separates positive examples, risk priors, critic
warnings, and writeback evidence; it is deterministic planner context, not a
learned stage planner.

Build a real-format / pseudo-real evidence pack:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/build_real_format_evidence_pack.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --sandbox-report /tmp/sandbox_rollout_g4_calibrated_for_evidence_pack.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/real_format_evidence_pack.json
```

The pack reports real-format/pseudo-real coverage, sim-real pair/gap counts,
calibration ids, object-pose-bias calibration coverage, and whether sandbox
rollout reports actually consumed calibration. Current G4 calibrated evidence
shows `sandbox_calibration_application_count=4`; this supports real-format and
pseudo-real pipeline evidence, not real-robot validation.

Create a fillable real-robot episode template directory:

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/create_real_episode_template.py \
  --output-dir /tmp/r1pro_real_episode_demo \
  --episode-id r1pro_real_demo_001 \
  --scenario G1 \
  --condition clean \
  --task-name grasp_place_demo
```

This creates `episode.json` plus `rgb/`, `depth/`, `lidar/`, `force/`,
`keyframes/`, `video/`, and `logs/`. Fill the paths in `episode.json`, then
validate and import:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/validate_real_episode.py \
  --episode-dir /tmp/r1pro_real_episode_demo \
  --check-refs

PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/import_real_episode.py \
  --episode-dir /tmp/r1pro_real_episode_demo \
  --source real \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json
```

For the on-site data checklist and field definitions, see
`docs/r1pro_real_episode_collection_guide.md`.

Run a sandbox calibration ablation:

```bash
MUJOCO_GL=osmesa PYTHONDONTWRITEBYTECODE=1 python -B \
  galaxea_mujoco/source/run_sandbox_calibration_ablation.py \
  --scenario G4 \
  --condition place_occupied \
  --candidate-id g4_avoid_occupied_primary \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/sandbox_calibration_ablation_g4_place_occupied.json
```

The ablation compares `sandbox_no_calibration`,
`sandbox_score_calibration_only`, and `sandbox_pose_and_score_calibration`.
Current G4 evidence shows the full calibration shifts object start from
`[0.16,0.0,0.805]` to `[0.2,0.04,0.765]`, while score-only and full calibration
both apply `calibration_risk_penalty=0.12`. This is object initial-state and
risk-score calibration, not dynamics/friction/contact calibration.

Run a visual retrieval ablation without executing MuJoCo:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/run_visual_retrieval_ablation.py \
  --scenario G3 \
  --condition clean \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --visual-index-dir galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/visual_index \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/visual_retrieval_ablation_g3_clean.json \
  --save-csv galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/visual_retrieval_ablation_g3_clean.csv
```

The ablation compares candidate ranking with and without visual keyframe scores.
Current evidence: G3/clean changes selected candidate from `g3_default` to
`g3_place_first`; G4/place_occupied keeps the same selected candidate but raises
candidate scores. This supports an auxiliary visual retrieval claim, not broad
multimodal reasoning.

Run a safety-focused stress report with risky candidates:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_memory_safety_stress_report.py \
  --scenario G4 \
  --condition place_occupied \
  --use-sandbox-calibration \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save /tmp/safety_stress_g4_place_occupied.json \
  --save-csv /tmp/safety_stress_g4_place_occupied.csv
```

The stress report compares `memory_only`, `memory_stage`,
`memory_sandbox_critic`, and `full_stage_lesson_sandbox`. It reports
`risky_candidate_selected_rate`, `safe_candidate_selected_rate`,
`risky_warn_or_block_rate`, `critic_warn_rate_avg`, and the score margin between
the selected candidate and the best risky candidate.

Run harder adversarial safety stress:

```bash
MUJOCO_GL=osmesa PYTHONDONTWRITEBYTECODE=1 python -B \
  galaxea_mujoco/source/run_harder_safety_stress_cases.py \
  --scenario G4 \
  --condition place_occupied \
  --use-sandbox-calibration \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/harder_safety_stress_g4_place_occupied.json \
  --save-csv galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/harder_safety_stress_g4_place_occupied.csv
```

This stress test artificially boosts the risky candidate in memory-only ranking
and then applies sandbox critic fusion. Current evidence: 4/4 adversarial cases
select `g4_fast_transport` before sandbox and redirect to
`g4_avoid_occupied_primary` after sandbox. The claim must mention adversarial
or artificially boosted ranking.

Demonstrate closed-loop writeback without mutating the source library:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_memory_writeback_demo.py \
  --scenario G4 \
  --condition place_occupied \
  --include-risky-candidates \
  --use-sandbox-rollout \
  --use-sandbox-calibration \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --lesson-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/llm_experience_lessons_g4_place_occupied.json \
  --save /tmp/writeback_demo_g4_place_occupied_sandbox.json
```

The writeback demo copies the library in memory, executes the selected
candidate, writes the normalized episode through `ExperienceLibrary.add_with_policy()`,
then reruns retrieval and ranking. It reports `entry_count_delta`,
`new_memory_retrieved`, `retrieval_overlap_before_after`,
`selected_candidate_changed`, and score/risk deltas.

Run repeated closed-loop writeback rounds:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_memory_writeback_benchmark.py \
  --scenario G3 \
  --condition clean \
  --rounds 3 \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --policy-calibration galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/policy_risk_calibration.json \
  --save /tmp/writeback_benchmark_g3_clean.json
```

For the full G4 stage+sandbox+calibration path, use the same tool with
`--scenario G4 --condition place_occupied --include-risky-candidates
--use-stage-retrieval --use-sandbox-rollout --use-sandbox-calibration`. This is
substantially slower because every round performs before/after sandbox rollout
for all candidates.

Existing scripts under `galaxea_mujoco/source` are intentionally left in place
for compatibility while imports are migrated.
