# Memory Type Coverage Report

This report measures field-level evidence for six robot memory types in the universal experience library.

## Summary

- Input: `results/memory/universal_pipeline_calibration_v1/universal_experience_library.json`
- Entry count: 12
- Average covered memory types per entry: 5.6667
- Entries covering all six types: 8 (0.6667)
- Source distribution: `{"pseudo_real": 8, "simulation": 4}`
- Memory role distribution: `{"pseudo_real_failure_case": 4, "sim_real_gap_memory": 8}`

## Coverage By Memory Type

| Memory type | Covered entries | Coverage rate | Main evidence fields | Example ids |
| --- | --- | --- | --- | --- |
| temporal_memory | 12 | 1.0 | skill_sequence=12 | exp_bbed526ab28b, exp_ddaf34a39edc, exp_6a115cd94118, exp_b7fb782456f2, pseudo_real_g3_clean_success_001 |
| spatial_memory | 12 | 1.0 | object_state.objects=12; object_state.occupancy=12; spatial_state=4; state_before=4; state_after=4; sim_real_gap.pose_gap=8 | exp_bbed526ab28b, exp_ddaf34a39edc, exp_6a115cd94118, exp_b7fb782456f2, pseudo_real_g3_clean_success_001 |
| episodic_memory | 12 | 1.0 | scenario=12; condition=12; task=12; result=12; source=12; backend=12; validation_status=12 | exp_bbed526ab28b, exp_ddaf34a39edc, exp_6a115cd94118, exp_b7fb782456f2, pseudo_real_g3_clean_success_001 |
| semantic_memory | 12 | 1.0 | anomaly=2; failure_taxonomy=6; memory_tags=12; retrieval_key=12; critic_result.rule_flags=12 | exp_bbed526ab28b, exp_ddaf34a39edc, exp_6a115cd94118, exp_b7fb782456f2, pseudo_real_g3_clean_success_001 |
| perceptual_memory | 12 | 1.0 | keyframes=12; sensor_summary.sensor_modalities=12; sensor_summary.raw_refs=4 | exp_bbed526ab28b, exp_ddaf34a39edc, exp_6a115cd94118, exp_b7fb782456f2, pseudo_real_g3_clean_success_001 |
| sim_real_gap_memory | 8 | 0.6667 | sim_real_pair=8; sim_real_gap.gap_id=8; sim_real_gap.outcome_gap=8; sim_real_gap.pose_gap=8; sim_real_gap.contact_gap=8; sandbox_calibration.calibration_id=8; sandbox_calibration.source_gap_ids=8; sandbox_calibration.object_pose_bias=8 | exp_bbed526ab28b, exp_ddaf34a39edc, exp_6a115cd94118, exp_b7fb782456f2, pseudo_real_g3_clean_success_001 |

## Paper Wording Boundary

- Safe claim: The experience library stores multi-type robot memories, with temporal, spatial, episodic, semantic, perceptual, and sim-real gap evidence represented by explicit fields.
- Avoid claim: Do not claim broad RoboMME-scale coverage or real-robot sensor validation from this report; the report measures field-level coverage in the current library.

## Entry Profiles

| Experience id | Source | Scenario | Condition | Success | Covered memory types |
| --- | --- | --- | --- | --- | --- |
| exp_bbed526ab28b | simulation | G3 | clean | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| exp_ddaf34a39edc | simulation | G3 | place_occupied | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| exp_6a115cd94118 | simulation | G4 | clean | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| exp_b7fb782456f2 | simulation | G4 | place_occupied | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| pseudo_real_g3_clean_success_001 | pseudo_real | G3 | clean | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| pseudo_real_g3_grasp_miss_001 | pseudo_real | G3 | grasp_miss | False | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory |
| pseudo_real_g3_grasp_slip_001 | pseudo_real | G3 | grasp_slip | False | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory |
| pseudo_real_g3_place_occupied_success_001 | pseudo_real | G3 | place_occupied | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| pseudo_real_g4_clean_success_001 | pseudo_real | G4 | clean | True | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| pseudo_real_g4_dual_arm_mismatch_001 | pseudo_real | G4 | dual_arm_mismatch | False | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory |
| pseudo_real_g4_place_occupied_failure_001 | pseudo_real | G4 | place_occupied | False | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory, sim_real_gap_memory |
| pseudo_real_g4_transport_collision_001 | pseudo_real | G4 | transport_collision | False | temporal_memory, spatial_memory, episodic_memory, semantic_memory, perceptual_memory |
