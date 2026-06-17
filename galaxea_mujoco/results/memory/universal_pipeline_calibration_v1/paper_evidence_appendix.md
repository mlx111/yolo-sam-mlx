# Paper Evidence Appendix

This appendix maps paper claims to generated implementation reports and safe wording boundaries.
Use `supported` claims as main-text claims. Treat `partial` claims as implementation support only, not validation evidence.

## Summary

- Evidence summary: `results/memory/universal_pipeline_calibration_v1`
- Claim count: 24
- Supported claim count: 23
- Missing report count: 0
- Status distribution: `{"partial": 1, "supported": 23}`

## Compact Evidence Table

| ID | Status | Claim | Primary report | Key metrics | Safe wording | Avoid wording |
|---|---|---|---|---|---|---|
| C01 | supported | Unified experience memory stores simulation and pseudo-real episodes with gap/calibration fields. | `results/memory/universal_pipeline_calibration_v1/summary_report.json` | entry_count=12; source_distribution={"pseudo_real": 8, "simulation": 4}; gap_count=4; calibration_count=4 | The memory library stores simulation and pseudo-real experiences using a unified schema with gap and calibration fields. | Do not claim true real-robot validation from this report. |
| C02 | supported | Real-format import and pseudo-real evidence exercise sim-real pair, gap, and sandbox calibration paths. | `results/memory/universal_pipeline_calibration_v1/real_format_evidence_pack.json` | real_format_entry_count=8; pseudo_real_entry_count=8; sim_entry_count=4; paired_gap_count=4; sim_real_gap_count=4; sensor_gap_entry_count=0; +3 more | The implementation supports real-format episode import and uses pseudo-real evidence to exercise sim-real pairing, gap extraction, and sandbox calibration, and shows calibration consumed by sandbox rollout reports. | Do not claim real-robot validation or real-robot success-rate improvement unless true real-source execution reports are added. |
| C03 | partial | Shared real-format memory stores RGB-D, lidar, and wrist-force evidence and can derive conservative sensor-gap summaries. | `results/memory/universal_pipeline_calibration_v1/real_format_evidence_pack.json` | sensor_evidence_entry_count=0; rgbd_evidence_entry_count=0; lidar_evidence_entry_count=0; wrist_force_evidence_entry_count=0; sensor_modality_distribution={}; sensor_gap_entry_count=0 | The implementation supports real-format sensor evidence fields for RGB-D, lidar, and wrist-force observations, and can derive conservative sensor-gap summaries from those stored observations. | Do not claim real-robot success-rate improvement or validated sensor-derived calibration effects from this row alone. |
| C04 | supported | Candidate plans can be shadow-rolled out in sandbox and scored before selection. | `results/memory/universal_pipeline_calibration_v1/sandbox_rollout_g4_place_occupied.json` | candidate_count=4; selected_before_sandbox=g4_avoid_occupied_primary; selected_after_sandbox=g4_avoid_occupied_primary; candidate_changed_by_sandbox=False; critic_status_counts={"pass": 4} | Candidate plans are shadow-executed in MuJoCo and fused with memory scores before selection. | Do not claim a full digital twin or real-world execution success. |
| C05 | supported | Stage-aware retrieval renders planner context separating generation, ranking, rewrite, and writeback evidence. | `results/memory/universal_pipeline_calibration_v1/stage_planner_context_g4.json` | context_count=1; stage_context_token_count_avg=252; stage_context_distinct_memory_count_avg=6; stage_specificity_score_avg=0.7056; risk_evidence_count_total=2; critic_warning_count_total=4 | Planner context is assembled from stage-specific memory evidence separating positive examples, risk priors, critic evidence, and writeback histories. | Do not claim a learned multi-stage planner. |
| C06 | supported | Visual keyframes are used as an auxiliary retrieval signal during candidate ranking. | `results/memory/universal_pipeline_calibration_v1/visual_retrieval_ablation_g3_clean.json; results/memory/universal_pipeline_calibration_v1/visual_retrieval_ablation_g4_place_occupied.json` | visual_index.indexed_entry_count=12; visual_index.indexed_image_count=60; g3.selected_candidate_before_visual=g3_default; g3.selected_candidate_after_visual=g3_place_first; g3.selected_candidate_change=True; g3.retrieval_changed_rate=0.3333; +2 more | Visual keyframes are indexed and used as an auxiliary retrieval signal during candidate ranking. | Do not claim broad multimodal semantic reasoning. |
| C07 | supported | Gap-derived calibration changes sandbox initial state and contributes an explicit risk penalty. | `results/memory/universal_pipeline_calibration_v1/sandbox_calibration_ablation_g4_place_occupied.json` | candidate_id=g4_avoid_occupied_primary; object_start_delta_score_only_vs_no_calibration=0; object_start_delta_full_vs_no_calibration=0.12; sandbox_score_delta_score_only_vs_no_calibration=0; sandbox_score_delta_full_vs_no_calibration=0; raw_sandbox_score_delta_score_only_vs_no_calibration=-0.12; +13 more | Gap-derived calibration changes sandbox initialization and contributes an explicit risk penalty during candidate evaluation. | Do not claim dynamics, friction, or full contact-model calibration; this ablation covers object initial-state shift and score penalty. |
| C08 | supported | LLM-generated lessons are audited for grounding, validity, concision, and conflicts before policy use. | `results/memory/universal_pipeline_calibration_v1/lesson_quality_report_g4_place_occupied.json` | lesson_count=1; avg_lesson_length=7; max_lesson_length=7; avg_field_length=4.5; max_field_length=7; evidence_id_valid_rate=1; +8 more | Generated lessons are checked for evidence grounding, candidate/skill validity, concision, and internal conflicts before being used for policy adjustment. | This is a static quality audit for generated lessons, not proof of learned policy rules. |
| C09 | supported | Sandbox critic identifies risky candidates and increases safety evidence under stress variants. | `results/memory/universal_pipeline_calibration_v1/safety_stress_g4_place_occupied.json` | memory_sandbox_critic.risky_candidate_selected_rate=0; memory_sandbox_critic.safe_candidate_selected_rate=1; memory_sandbox_critic.risky_warn_or_block_rate=1; memory_sandbox_critic.critic_warn_rate_avg=0.2; full_stage_lesson_sandbox.risky_candidate_selected_rate=0; full_stage_lesson_sandbox.safe_candidate_selected_rate=1; +2 more | The sandbox critic identifies unsafe candidates and increases the safety margin in stress reports. | Do not claim selection changed if memory-only already selected a safe candidate. |
| C10 | supported | Under adversarial ranking stress, sandbox critic redirects selection away from artificially boosted risky candidates. | `results/memory/universal_pipeline_calibration_v1/harder_safety_stress_g4_place_occupied.json` | memory_only_risky_selected_count=4; sandbox_prevented_risky_selection_count=4; sandbox_prevented_risky_selection_rate=1; selection_changed_by_sandbox_count=4; sandbox_risky_selected_count=0 | Under adversarial ranking stress with artificially boosted risky candidates, sandbox critic can redirect selection away from the risky candidate. | This stress test uses artificial ranking perturbations; do not claim the same selection change occurs in the unperturbed policy. |
| C11 | supported | Closed-loop writeback stores executed experiences and retrieves them in later ranking passes. | `results/memory/universal_pipeline_calibration_v1/writeback_benchmark_g3_clean.json` | round_count=3; write_count=3; entry_count_delta=2; new_memory_retrieval_rate=1; selected_candidate_change_rate=0.3333; score_delta_after_writeback_avg=0.0273; +1 more | Newly executed experiences can be written back and retrieved by later candidate ranking passes. | Do not claim statistically proven long-horizon success-rate improvement. |
| C12 | supported | The experience library represents multiple robot memory types rather than only flat episode logs. | `results/memory/universal_pipeline_calibration_v1/memory_type_coverage_report.json` | entry_count=12; covered_memory_type_count_avg=5.667; entries_covering_all_memory_types=8; entries_covering_all_memory_types_rate=0.6667; temporal_memory.coverage_rate=1; spatial_memory.coverage_rate=1; +4 more | The experience library stores multi-type robot memories, with temporal, spatial, episodic, semantic, perceptual, and sim-real gap evidence represented by explicit fields. | Do not claim broad RoboMME-scale coverage or real-robot sensor validation from this report; the report measures field-level coverage in the current library. |
| C13 | supported | Text-semantic summaries provide an auxiliary retrieval signal over structured experience fields. | `results/memory/universal_pipeline_calibration_v1/text_semantic_memory_report.json` | entry_count=12; semantic_summary_nonempty_count=12; semantic_summary_nonempty_rate=1; avg_token_count=148.8; semantic_signal_rate=1; query_count=9; +3 more | The system constructs explicit text-semantic summaries from scenario, condition, task, anomaly, failure taxonomy, critic, gap, and retrieval-key fields, and uses TF-IDF vectors indexed by FAISS as an auxiliary semantic retrieval signal. | Do not claim this is a learned language encoder or neural embedding benchmark; the report measures lightweight TF-IDF + FAISS semantic retrieval only. |
| C14 | supported | Real LLM-generated recovery-plan candidates can be semantically validated, sandboxed in parallel, critic-ranked, and exported as a validated robot plan. | `results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g3_clean_real_llm_report.json` | dry_run_llm=False; llm_provider=doubao; num_plans_requested=2; num_plans_normalized=2; sandboxed_plan_count=2; failed_worker_count=0; +8 more | A configured external LLM generated multiple recovery-plan candidates; the system normalized them, validated skill preconditions/effects, ran parallel MuJoCo sandbox rollouts, ranked by critic/sandbox status, and exported a validated_robot_plan for dry-run dispatch. | Do not claim real-robot execution success, arbitrary unseen-skill planning, or statistically proven superiority over all single-plan baselines. |
| C15 | supported | Multi-candidate LLM plan search evaluates more recovery alternatives than a single-plan LLM baseline before selecting a validated plan. | `results/memory/universal_pipeline_calibration_v1/llm_plan_search_ablation_g3_clean_real_llm.json` | plan_count_delta=1; sandboxed_plan_count_delta=1; accepted_plan_count_delta=1; best_sandbox_score_delta=0; unique_step_sequence_delta=1; single_final_status=accept; +19 more | The multi-candidate search evaluates more LLM-generated recovery alternatives than a single-plan baseline, runs valid candidates through parallel sandbox rollout, and selects a critic-approved validated plan. | Do not claim a success-rate improvement from this small G3 clean ablation alone; both variants succeeded in the current smoke. |
| C16 | supported | Field-atomic execution experiences are converted into parameter priors for later LLM planning. | `/tmp/field_atomic_memory_report.json` | field_atomic_entry_count=3; field_atomic_success_count=3; field_atomic_failure_count=0; prior_action_count=0 | The system stores successful and failed field atomic experiences and uses them as explicit parameter priors for later planning. | Do not claim this proves improved real-robot success without a real-robot ablation. |
| C17 | supported | Cold-start and writeback rounds show whether field-atomic memory is exposed to the planner as explicit context. | `/tmp/field_atomic_ablation.json` | dry_run_llm=True; cold_memory_count=0; warm_memory_count=3; warm_prior_action_count=3; parameter_changed_count=0; action_set_changed=False; +2 more | Field atomic writeback can be reloaded as explicit planner_input priors in a subsequent planning round. | Do not claim parameter improvement from dry-run LLM if the mock plan is deterministic; use real LLM or varied goals for parameter-change evidence. |
| C18 | supported | Field-atomic execution traces expose action-level tracking error, gripper commands, and direct-qpos usage for debugging. | `results/memory/universal_pipeline_calibration_v1/field_atomic_trace_summary.json` | trace_count=10; action_count=5; success_count=10; failure_count=0; direct_qpos_true_count=0; direct_qpos_false_count=10; +2 more | Field-atomic executions expose action-level trace summaries, including final tracking error, gripper command, control mode, and direct-qpos usage for onsite debugging. | Do not claim these summaries prove real-robot tracking accuracy; they summarize MuJoCo or stored experience traces only. |
| C19 | supported | Rewrite-loop ablation reports separate supported component evidence from missing true sequential LLM rewrite evidence. | `results/memory/universal_pipeline_calibration_v1/rewrite_loop_ablation_report.json` | variant_count=5; status_counts={"supported": 5}; sequential_rewrite_report_count=1; has_single_plan_baseline=True; has_multi_candidate_critic_ranking=True; has_sequential_critic_feedback_rewrite=True; +17 more | Existing reports support single-plan validation, multi-candidate critic ranking, memory-backed risk evidence, field-atomic parameter-prior exposure, and a dry-run sequential critic-feedback rewrite attempt. | Do not claim critic-feedback rewriting improves recovery unless the underlying report has rewrite_rounds > 0 and final_sandbox_status=accept under a real LLM or clearly stated dry-run setting. |
| C20 | supported | Physical sandbox rollouts can sweep control and scene perturbations to expose sensitivity to pose, delay, gain, and gripper effects. | `/tmp/physical_sandbox_perturbation_report.json` | rollout_count=6; success_count=0; task_success_count=6; failure_label_counts={"object_not_lifted": 6, "skill_failed:verify_grasp": 6, "verify_grasp": 6}; nominal.success_rate=0; nominal.success_rate_delta_vs_nominal=0; +10 more | The physical MuJoCo sandbox can sweep pose, delay, gain, and gripper perturbations and report which failures are sensitive to those controls. | Do not claim real-driver calibration or real-robot robustness without measured hardware response data. |
| C21 | supported | Core skill defaults are audited to prefer physical actuator execution over direct-qpos shortcuts. | `results/memory/universal_pipeline_calibration_v1/physical_default_audit_report.json` | file_count=26; direct_qpos_true_total=0; direct_qpos_false_total=20 | Core base, torso, arm, gripper, and field atomic defaults have been audited to prefer physical control over direct-qpos shortcuts. | Do not claim that every explicit debug or test path is physical-only; the audit concerns defaults, not every possible parameter override. |
| C22 | supported | Memory writeback is auditable through explicit write, skip, merge, and reject decisions. | `/tmp/write_policy_audit_report.json` | entry_count=3; decision_counts={"write": 3}; reason_counts={"preserve_field_atomic_experience": 3}; preserved_reason_counts={"preserve_field_atomic_experience": 3}; memory_role_counts={"field_atomic_success": 3} | The write policy is auditable: each candidate entry receives an explicit write, skip, merge, or reject decision with a reason. | Do not claim the write policy is learned or globally optimal; it is an explicit engineering gate. |
| C23 | supported | Write-policy pressure tests exercise all lifecycle decisions: write, merge, skip, and reject. | `results/memory/universal_pipeline_calibration_v1/write_policy_pressure_report.json` | case_count=6; stored_library_entry_count=3; decision_counts={"merge": 1, "reject": 1, "skip": 1, "write": 3}; reason_counts={"accepted": 1, "duplicate_low_risk_success": 1, "low_value_success": 1, "missing_required_fields": 1, "preserve_failure_taxonomy": 1, "preserve_field_atomic_experience": 1}; expected_decisions_present={"merge": true, "reject": true, "skip": true, "write": true} | A deterministic pressure test constructs representative experiences that trigger write, merge, skip, and reject decisions, showing that writeback is not a plain append-only log. | Do not claim the handcrafted pressure cases prove optimal memory lifecycle policy. |
| C24 | supported | Runtime scene observations have a fixed schema for converting onsite RGB-D/LiDAR outputs into sandbox scene construction inputs. | `/home/mlx/mujoco/YOLO_World-SAM-GraspNet/experience_system/templates/field_runtime_scene_observation_template.json` | schema_exists=True | The implementation defines a field runtime observation template for converting onsite perception outputs into sandbox scene inputs. | Do not claim automatic high-fidelity scene reconstruction until real perception outputs are mapped and validated onsite. |

## Claim Cards

### C01-unified-experience-memory-stores-simulation-and-

- Status: `supported`
- Claim: Unified experience memory stores simulation and pseudo-real episodes with gap/calibration fields.
- Primary report: `results/memory/universal_pipeline_calibration_v1/summary_report.json`
- Safe wording: The memory library stores simulation and pseudo-real experiences using a unified schema with gap and calibration fields.
- Avoid wording: Do not claim true real-robot validation from this report.
- Key metrics: entry_count=12; source_distribution={"pseudo_real": 8, "simulation": 4}; gap_count=4; calibration_count=4

### C02-real-format-import-and-pseudo-real-evidence-exer

- Status: `supported`
- Claim: Real-format import and pseudo-real evidence exercise sim-real pair, gap, and sandbox calibration paths.
- Primary report: `results/memory/universal_pipeline_calibration_v1/real_format_evidence_pack.json`
- Safe wording: The implementation supports real-format episode import and uses pseudo-real evidence to exercise sim-real pairing, gap extraction, and sandbox calibration, and shows calibration consumed by sandbox rollout reports.
- Avoid wording: Do not claim real-robot validation or real-robot success-rate improvement unless true real-source execution reports are added.
- Key metrics: real_format_entry_count=8; pseudo_real_entry_count=8; sim_entry_count=4; paired_gap_count=4; sim_real_gap_count=4; sensor_gap_entry_count=0; calibration_id_count=4; calibration_with_object_pose_bias_count=4; sandbox_calibration_application_count=4

### C03-shared-real-format-memory-stores-rgb-d-lidar-and

- Status: `partial`
- Claim: Shared real-format memory stores RGB-D, lidar, and wrist-force evidence and can derive conservative sensor-gap summaries.
- Primary report: `results/memory/universal_pipeline_calibration_v1/real_format_evidence_pack.json`
- Safe wording: The implementation supports real-format sensor evidence fields for RGB-D, lidar, and wrist-force observations, and can derive conservative sensor-gap summaries from those stored observations.
- Avoid wording: Do not claim real-robot success-rate improvement or validated sensor-derived calibration effects from this row alone.
- Key metrics: sensor_evidence_entry_count=0; rgbd_evidence_entry_count=0; lidar_evidence_entry_count=0; wrist_force_evidence_entry_count=0; sensor_modality_distribution={}; sensor_gap_entry_count=0

### C04-candidate-plans-can-be-shadow-rolled-out-in-sand

- Status: `supported`
- Claim: Candidate plans can be shadow-rolled out in sandbox and scored before selection.
- Primary report: `results/memory/universal_pipeline_calibration_v1/sandbox_rollout_g4_place_occupied.json`
- Safe wording: Candidate plans are shadow-executed in MuJoCo and fused with memory scores before selection.
- Avoid wording: Do not claim a full digital twin or real-world execution success.
- Key metrics: candidate_count=4; selected_before_sandbox=g4_avoid_occupied_primary; selected_after_sandbox=g4_avoid_occupied_primary; candidate_changed_by_sandbox=False; critic_status_counts={"pass": 4}

### C05-stage-aware-retrieval-renders-planner-context-se

- Status: `supported`
- Claim: Stage-aware retrieval renders planner context separating generation, ranking, rewrite, and writeback evidence.
- Primary report: `results/memory/universal_pipeline_calibration_v1/stage_planner_context_g4.json`
- Safe wording: Planner context is assembled from stage-specific memory evidence separating positive examples, risk priors, critic evidence, and writeback histories.
- Avoid wording: Do not claim a learned multi-stage planner.
- Key metrics: context_count=1; stage_context_token_count_avg=252; stage_context_distinct_memory_count_avg=6; stage_specificity_score_avg=0.7056; risk_evidence_count_total=2; critic_warning_count_total=4

### C06-visual-keyframes-are-used-as-an-auxiliary-retrie

- Status: `supported`
- Claim: Visual keyframes are used as an auxiliary retrieval signal during candidate ranking.
- Primary report: `results/memory/universal_pipeline_calibration_v1/visual_retrieval_ablation_g3_clean.json; results/memory/universal_pipeline_calibration_v1/visual_retrieval_ablation_g4_place_occupied.json`
- Safe wording: Visual keyframes are indexed and used as an auxiliary retrieval signal during candidate ranking.
- Avoid wording: Do not claim broad multimodal semantic reasoning.
- Key metrics: visual_index.indexed_entry_count=12; visual_index.indexed_image_count=60; g3.selected_candidate_before_visual=g3_default; g3.selected_candidate_after_visual=g3_place_first; g3.selected_candidate_change=True; g3.retrieval_changed_rate=0.3333; g4.candidate_score_delta_avg=0.0882; g4.candidate_score_delta_max=0.1396

### C07-gap-derived-calibration-changes-sandbox-initial-

- Status: `supported`
- Claim: Gap-derived calibration changes sandbox initial state and contributes an explicit risk penalty.
- Primary report: `results/memory/universal_pipeline_calibration_v1/sandbox_calibration_ablation_g4_place_occupied.json`
- Safe wording: Gap-derived calibration changes sandbox initialization and contributes an explicit risk penalty during candidate evaluation.
- Avoid wording: Do not claim dynamics, friction, or full contact-model calibration; this ablation covers object initial-state shift and score penalty.
- Key metrics: candidate_id=g4_avoid_occupied_primary; object_start_delta_score_only_vs_no_calibration=0; object_start_delta_full_vs_no_calibration=0.12; sandbox_score_delta_score_only_vs_no_calibration=0; sandbox_score_delta_full_vs_no_calibration=0; raw_sandbox_score_delta_score_only_vs_no_calibration=-0.12; raw_sandbox_score_delta_full_vs_no_calibration=-0.12; critic_status_delta_score_only=pass->pass; critic_status_delta_full=pass->pass; critic_risk_delta_score_only=0; critic_risk_delta_full=0; calibration_risk_penalty_score_only=0.12; calibration_risk_penalty_full=0.12; selected_variant_without_calibration=sandbox_no_calibration; selected_variant_with_full_calibration=sandbox_pose_and_score_calibration; selected_candidate_delta=False; nominal_object_start=[0.16, 0.0, 0.805]; calibrated_object_start=[0.2, 0.04, 0.765]; object_pose_bias=[0.04, 0.04, -0.04]

### C08-llm-generated-lessons-are-audited-for-grounding-

- Status: `supported`
- Claim: LLM-generated lessons are audited for grounding, validity, concision, and conflicts before policy use.
- Primary report: `results/memory/universal_pipeline_calibration_v1/lesson_quality_report_g4_place_occupied.json`
- Safe wording: Generated lessons are checked for evidence grounding, candidate/skill validity, concision, and internal conflicts before being used for policy adjustment.
- Avoid wording: This is a static quality audit for generated lessons, not proof of learned policy rules.
- Key metrics: lesson_count=1; avg_lesson_length=7; max_lesson_length=7; avg_field_length=4.5; max_field_length=7; evidence_id_valid_rate=1; candidate_id_valid_rate=1; skill_reference_valid_rate=1; duplicate_lesson_count=0; conflict_pair_count=0; template_like_phrase_count=0; actionable_lesson_rate=1; concise_lesson_rate=1; confidence_avg=0.8

### C09-sandbox-critic-identifies-risky-candidates-and-i

- Status: `supported`
- Claim: Sandbox critic identifies risky candidates and increases safety evidence under stress variants.
- Primary report: `results/memory/universal_pipeline_calibration_v1/safety_stress_g4_place_occupied.json`
- Safe wording: The sandbox critic identifies unsafe candidates and increases the safety margin in stress reports.
- Avoid wording: Do not claim selection changed if memory-only already selected a safe candidate.
- Key metrics: memory_sandbox_critic.risky_candidate_selected_rate=0; memory_sandbox_critic.safe_candidate_selected_rate=1; memory_sandbox_critic.risky_warn_or_block_rate=1; memory_sandbox_critic.critic_warn_rate_avg=0.2; full_stage_lesson_sandbox.risky_candidate_selected_rate=0; full_stage_lesson_sandbox.safe_candidate_selected_rate=1; full_stage_lesson_sandbox.risky_warn_or_block_rate=1; full_stage_lesson_sandbox.critic_warn_rate_avg=0.2

### C10-under-adversarial-ranking-stress-sandbox-critic-

- Status: `supported`
- Claim: Under adversarial ranking stress, sandbox critic redirects selection away from artificially boosted risky candidates.
- Primary report: `results/memory/universal_pipeline_calibration_v1/harder_safety_stress_g4_place_occupied.json`
- Safe wording: Under adversarial ranking stress with artificially boosted risky candidates, sandbox critic can redirect selection away from the risky candidate.
- Avoid wording: This stress test uses artificial ranking perturbations; do not claim the same selection change occurs in the unperturbed policy.
- Key metrics: memory_only_risky_selected_count=4; sandbox_prevented_risky_selection_count=4; sandbox_prevented_risky_selection_rate=1; selection_changed_by_sandbox_count=4; sandbox_risky_selected_count=0

### C11-closed-loop-writeback-stores-executed-experience

- Status: `supported`
- Claim: Closed-loop writeback stores executed experiences and retrieves them in later ranking passes.
- Primary report: `results/memory/universal_pipeline_calibration_v1/writeback_benchmark_g3_clean.json`
- Safe wording: Newly executed experiences can be written back and retrieved by later candidate ranking passes.
- Avoid wording: Do not claim statistically proven long-horizon success-rate improvement.
- Key metrics: round_count=3; write_count=3; entry_count_delta=2; new_memory_retrieval_rate=1; selected_candidate_change_rate=0.3333; score_delta_after_writeback_avg=0.0273; risk_delta_after_writeback_avg=-0.0471

### C12-the-experience-library-represents-multiple-robot

- Status: `supported`
- Claim: The experience library represents multiple robot memory types rather than only flat episode logs.
- Primary report: `results/memory/universal_pipeline_calibration_v1/memory_type_coverage_report.json`
- Safe wording: The experience library stores multi-type robot memories, with temporal, spatial, episodic, semantic, perceptual, and sim-real gap evidence represented by explicit fields.
- Avoid wording: Do not claim broad RoboMME-scale coverage or real-robot sensor validation from this report; the report measures field-level coverage in the current library.
- Key metrics: entry_count=12; covered_memory_type_count_avg=5.667; entries_covering_all_memory_types=8; entries_covering_all_memory_types_rate=0.6667; temporal_memory.coverage_rate=1; spatial_memory.coverage_rate=1; episodic_memory.coverage_rate=1; semantic_memory.coverage_rate=1; perceptual_memory.coverage_rate=1; sim_real_gap_memory.coverage_rate=0.6667

### C13-text-semantic-summaries-provide-an-auxiliary-ret

- Status: `supported`
- Claim: Text-semantic summaries provide an auxiliary retrieval signal over structured experience fields.
- Primary report: `results/memory/universal_pipeline_calibration_v1/text_semantic_memory_report.json`
- Safe wording: The system constructs explicit text-semantic summaries from scenario, condition, task, anomaly, failure taxonomy, critic, gap, and retrieval-key fields, and uses TF-IDF vectors indexed by FAISS as an auxiliary semantic retrieval signal.
- Avoid wording: Do not claim this is a learned language encoder or neural embedding benchmark; the report measures lightweight TF-IDF + FAISS semantic retrieval only.
- Key metrics: entry_count=12; semantic_summary_nonempty_count=12; semantic_summary_nonempty_rate=1; avg_token_count=148.8; semantic_signal_rate=1; query_count=9; same_scenario_topk_match_count=31; cross_condition_topk_match_count=36; query_token_coverage={"anomaly": 2, "condition": 12, "critic": 12, "failure": 6, "gap": 8, "plan": 12, "scenario": 12, "task": 12}

### C14-real-llm-generated-recovery-plan-candidates-can-

- Status: `supported`
- Claim: Real LLM-generated recovery-plan candidates can be semantically validated, sandboxed in parallel, critic-ranked, and exported as a validated robot plan.
- Primary report: `results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g3_clean_real_llm_report.json`
- Safe wording: A configured external LLM generated multiple recovery-plan candidates; the system normalized them, validated skill preconditions/effects, ran parallel MuJoCo sandbox rollouts, ranked by critic/sandbox status, and exported a validated_robot_plan for dry-run dispatch.
- Avoid wording: Do not claim real-robot execution success, arbitrary unseen-skill planning, or statistically proven superiority over all single-plan baselines.
- Key metrics: dry_run_llm=False; llm_provider=doubao; num_plans_requested=2; num_plans_normalized=2; sandboxed_plan_count=2; failed_worker_count=0; final_sandbox_status=accept; selected_plan_index=0; rollouts_per_minute=1.207; search_status_counts={"accept": 2}; semantic_status_counts={"pass": 2}; critic_status_counts={"pass": 2}; dry_run_executor_status=executed; dry_run_executor_success=True

### C15-multi-candidate-llm-plan-search-evaluates-more-r

- Status: `supported`
- Claim: Multi-candidate LLM plan search evaluates more recovery alternatives than a single-plan LLM baseline before selecting a validated plan.
- Primary report: `results/memory/universal_pipeline_calibration_v1/llm_plan_search_ablation_g3_clean_real_llm.json`
- Safe wording: The multi-candidate search evaluates more LLM-generated recovery alternatives than a single-plan baseline, runs valid candidates through parallel sandbox rollout, and selects a critic-approved validated plan.
- Avoid wording: Do not claim a success-rate improvement from this small G3 clean ablation alone; both variants succeeded in the current smoke.
- Key metrics: plan_count_delta=1; sandboxed_plan_count_delta=1; accepted_plan_count_delta=1; best_sandbox_score_delta=0; unique_step_sequence_delta=1; single_final_status=accept; multi_final_status=accept; single_plan_real_llm.plan_count=1; single_plan_real_llm.sandboxed_plan_count=1; single_plan_real_llm.accepted_plan_count=1; single_plan_real_llm.review_plan_count=0; single_plan_real_llm.rejected_plan_count=0; single_plan_real_llm.best_sandbox_score=1; single_plan_real_llm.candidate_diversity_unique_step_sequences=1; single_plan_real_llm.failed_worker_count=0; single_plan_real_llm.final_sandbox_status=accept; multi_plan_real_llm_parallel_sandbox.plan_count=2; multi_plan_real_llm_parallel_sandbox.sandboxed_plan_count=2; multi_plan_real_llm_parallel_sandbox.accepted_plan_count=2; multi_plan_real_llm_parallel_sandbox.review_plan_count=0; multi_plan_real_llm_parallel_sandbox.rejected_plan_count=0; multi_plan_real_llm_parallel_sandbox.best_sandbox_score=1; multi_plan_real_llm_parallel_sandbox.candidate_diversity_unique_step_sequences=2; multi_plan_real_llm_parallel_sandbox.failed_worker_count=0; multi_plan_real_llm_parallel_sandbox.final_sandbox_status=accept

### C16-field-atomic-execution-experiences-are-converted

- Status: `supported`
- Claim: Field-atomic execution experiences are converted into parameter priors for later LLM planning.
- Primary report: `/tmp/field_atomic_memory_report.json`
- Safe wording: The system stores successful and failed field atomic experiences and uses them as explicit parameter priors for later planning.
- Avoid wording: Do not claim this proves improved real-robot success without a real-robot ablation.
- Key metrics: field_atomic_entry_count=3; field_atomic_success_count=3; field_atomic_failure_count=0; prior_action_count=0

### C17-cold-start-and-writeback-rounds-show-whether-fie

- Status: `supported`
- Claim: Cold-start and writeback rounds show whether field-atomic memory is exposed to the planner as explicit context.
- Primary report: `/tmp/field_atomic_ablation.json`
- Safe wording: Field atomic writeback can be reloaded as explicit planner_input priors in a subsequent planning round.
- Avoid wording: Do not claim parameter improvement from dry-run LLM if the mock plan is deterministic; use real LLM or varied goals for parameter-change evidence.
- Key metrics: dry_run_llm=True; cold_memory_count=0; warm_memory_count=3; warm_prior_action_count=3; parameter_changed_count=0; action_set_changed=False; cold_write_count=3; warm_write_count=3

### C18-field-atomic-execution-traces-expose-action-leve

- Status: `supported`
- Claim: Field-atomic execution traces expose action-level tracking error, gripper commands, and direct-qpos usage for debugging.
- Primary report: `results/memory/universal_pipeline_calibration_v1/field_atomic_trace_summary.json`
- Safe wording: Field-atomic executions expose action-level trace summaries, including final tracking error, gripper command, control mode, and direct-qpos usage for onsite debugging.
- Avoid wording: Do not claim these summaries prove real-robot tracking accuracy; they summarize MuJoCo or stored experience traces only.
- Key metrics: trace_count=10; action_count=5; success_count=10; failure_count=0; direct_qpos_true_count=0; direct_qpos_false_count=10; final_error={"count": 4, "max": 0.002026, "mean": 0.001998, "median": 0.001998, "min": 0.001971}; action_kind_counts={"base": 2, "gripper": 2, "sensor": 4, "torso": 2}

### C19-rewrite-loop-ablation-reports-separate-supported

- Status: `supported`
- Claim: Rewrite-loop ablation reports separate supported component evidence from missing true sequential LLM rewrite evidence.
- Primary report: `results/memory/universal_pipeline_calibration_v1/rewrite_loop_ablation_report.json`
- Safe wording: Existing reports support single-plan validation, multi-candidate critic ranking, memory-backed risk evidence, field-atomic parameter-prior exposure, and a dry-run sequential critic-feedback rewrite attempt.
- Avoid wording: Do not claim critic-feedback rewriting improves recovery unless the underlying report has rewrite_rounds > 0 and final_sandbox_status=accept under a real LLM or clearly stated dry-run setting.
- Key metrics: variant_count=5; status_counts={"supported": 5}; sequential_rewrite_report_count=1; has_single_plan_baseline=True; has_multi_candidate_critic_ranking=True; has_sequential_critic_feedback_rewrite=True; has_failure_memory_ranking_evidence=True; has_parameter_prior_evidence=True; has_true_sequential_rewrite_evidence=True; has_successful_sequential_rewrite_evidence=False; single_plan_no_feedback.rewrite_rounds=0; single_plan_no_feedback.final_sandbox_status=accept; sequential_critic_feedback_rewrite.rewrite_rounds=1; sequential_critic_feedback_rewrite.final_sandbox_status=reject; sequential_critic_feedback_rewrite.critic_feedback_history_count=2; sequential_critic_feedback_rewrite.sandbox_status_changed=False; multi_candidate_critic_ranking.final_sandbox_status=accept; multi_candidate_critic_ranking.sandboxed_plan_count=2; critic_plus_failure_memory.changed_rate=0.375; critic_plus_failure_memory.failure_evidence_count=17; critic_failure_memory_parameter_priors.warm_memory_count=3; critic_failure_memory_parameter_priors.warm_prior_action_count=3; critic_failure_memory_parameter_priors.parameter_changed_count=0

### C20-physical-sandbox-rollouts-can-sweep-control-and-

- Status: `supported`
- Claim: Physical sandbox rollouts can sweep control and scene perturbations to expose sensitivity to pose, delay, gain, and gripper effects.
- Primary report: `/tmp/physical_sandbox_perturbation_report.json`
- Safe wording: The physical MuJoCo sandbox can sweep pose, delay, gain, and gripper perturbations and report which failures are sensitive to those controls.
- Avoid wording: Do not claim real-driver calibration or real-robot robustness without measured hardware response data.
- Key metrics: rollout_count=6; success_count=0; task_success_count=6; failure_label_counts={"object_not_lifted": 6, "skill_failed:verify_grasp": 6, "verify_grasp": 6}; nominal.success_rate=0; nominal.success_rate_delta_vs_nominal=0; pose_noise_x_plus_2cm.success_rate=0; pose_noise_x_plus_2cm.success_rate_delta_vs_nominal=0; pose_noise_y_plus_2cm.success_rate=0; pose_noise_y_plus_2cm.success_rate_delta_vs_nominal=0; control_delay_3_steps.success_rate=0; control_delay_3_steps.success_rate_delta_vs_nominal=0; low_gain_0_75.success_rate=0; low_gain_0_75.success_rate_delta_vs_nominal=0; gripper_underclose_1cm.success_rate=0; gripper_underclose_1cm.success_rate_delta_vs_nominal=0

### C21-core-skill-defaults-are-audited-to-prefer-physic

- Status: `supported`
- Claim: Core skill defaults are audited to prefer physical actuator execution over direct-qpos shortcuts.
- Primary report: `results/memory/universal_pipeline_calibration_v1/physical_default_audit_report.json`
- Safe wording: Core base, torso, arm, gripper, and field atomic defaults have been audited to prefer physical control over direct-qpos shortcuts.
- Avoid wording: Do not claim that every explicit debug or test path is physical-only; the audit concerns defaults, not every possible parameter override.
- Key metrics: file_count=26; direct_qpos_true_total=0; direct_qpos_false_total=20

### C22-memory-writeback-is-auditable-through-explicit-w

- Status: `supported`
- Claim: Memory writeback is auditable through explicit write, skip, merge, and reject decisions.
- Primary report: `/tmp/write_policy_audit_report.json`
- Safe wording: The write policy is auditable: each candidate entry receives an explicit write, skip, merge, or reject decision with a reason.
- Avoid wording: Do not claim the write policy is learned or globally optimal; it is an explicit engineering gate.
- Key metrics: entry_count=3; decision_counts={"write": 3}; reason_counts={"preserve_field_atomic_experience": 3}; preserved_reason_counts={"preserve_field_atomic_experience": 3}; memory_role_counts={"field_atomic_success": 3}

### C23-write-policy-pressure-tests-exercise-all-lifecyc

- Status: `supported`
- Claim: Write-policy pressure tests exercise all lifecycle decisions: write, merge, skip, and reject.
- Primary report: `results/memory/universal_pipeline_calibration_v1/write_policy_pressure_report.json`
- Safe wording: A deterministic pressure test constructs representative experiences that trigger write, merge, skip, and reject decisions, showing that writeback is not a plain append-only log.
- Avoid wording: Do not claim the handcrafted pressure cases prove optimal memory lifecycle policy.
- Key metrics: case_count=6; stored_library_entry_count=3; decision_counts={"merge": 1, "reject": 1, "skip": 1, "write": 3}; reason_counts={"accepted": 1, "duplicate_low_risk_success": 1, "low_value_success": 1, "missing_required_fields": 1, "preserve_failure_taxonomy": 1, "preserve_field_atomic_experience": 1}; expected_decisions_present={"merge": true, "reject": true, "skip": true, "write": true}

### C24-runtime-scene-observations-have-a-fixed-schema-f

- Status: `supported`
- Claim: Runtime scene observations have a fixed schema for converting onsite RGB-D/LiDAR outputs into sandbox scene construction inputs.
- Primary report: `/home/mlx/mujoco/YOLO_World-SAM-GraspNet/experience_system/templates/field_runtime_scene_observation_template.json`
- Safe wording: The implementation defines a field runtime observation template for converting onsite perception outputs into sandbox scene inputs.
- Avoid wording: Do not claim automatic high-fidelity scene reconstruction until real perception outputs are mapped and validated onsite.
- Key metrics: schema_exists=True
