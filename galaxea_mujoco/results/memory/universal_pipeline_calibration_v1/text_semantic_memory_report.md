# Text Semantic Memory Report

This report summarizes explicit text-semantic evidence extracted from the universal experience library.

## Summary

- Input: `results/memory/universal_pipeline_calibration_v1/universal_experience_library.json`
- Retrieval backend: `faiss_tfidf`
- FAISS index: `{"dimension": 210, "indexed_entry_count": 12}`
- Entry count: 12
- Nonempty semantic summaries: 12 (1.0)
- Average token count: 148.75
- Semantic signal rate: 1.0
- Query count: 9
- Same-scenario top-k matches: 31
- Cross-scenario top-k matches: 14
- Cross-condition top-k matches: 36

## Field Coverage

- Scenario tokens: 12
- Condition tokens: 12
- Task tokens: 12
- Anomaly tokens: 2
- Failure tokens: 6
- Critic tokens: 12
- Gap tokens: 8
- Plan tokens: 12

## Paper Wording Boundary

- Safe claim: The system constructs explicit text-semantic summaries from scenario, condition, task, anomaly, failure taxonomy, critic, gap, and retrieval-key fields, and uses TF-IDF vectors indexed by FAISS as an auxiliary semantic retrieval signal.
- Avoid claim: Do not claim this is a learned language encoder or neural embedding benchmark; the report measures lightweight TF-IDF + FAISS semantic retrieval only.

## Retrieval Samples

### exp_bbed526ab28b

- Query: `G3 clean sortable_object task_chain warn sim_real_gap_memory`
- Top matches: 5
- pseudo_real_g3_clean_success_001 score=0.2626 (G3/clean/)
- pseudo_real_g3_place_occupied_success_001 score=0.1844 (G3/place_occupied/)
- pseudo_real_g4_clean_success_001 score=0.158 (G4/clean/)
- pseudo_real_g3_grasp_miss_001 score=0.1139 (G3/grasp_miss/grasp_miss)
- pseudo_real_g3_grasp_slip_001 score=0.1134 (G3/grasp_slip/grasp_slip)

### exp_ddaf34a39edc

- Query: `G3 place_occupied sortable_object task_chain warn sim_real_gap_memory`
- Top matches: 5
- pseudo_real_g3_place_occupied_success_001 score=0.2498 (G3/place_occupied/)
- pseudo_real_g3_clean_success_001 score=0.2368 (G3/clean/)
- pseudo_real_g4_place_occupied_failure_001 score=0.2033 (G4/place_occupied/place_occupied)
- pseudo_real_g3_grasp_miss_001 score=0.1415 (G3/grasp_miss/grasp_miss)
- pseudo_real_g3_grasp_slip_001 score=0.1408 (G3/grasp_slip/grasp_slip)

### exp_6a115cd94118

- Query: `G4 clean large_object task_chain warn sim_real_gap_memory`
- Top matches: 5
- pseudo_real_g4_clean_success_001 score=0.2533 (G4/clean/)
- pseudo_real_g3_clean_success_001 score=0.1638 (G3/clean/)
- pseudo_real_g4_place_occupied_failure_001 score=0.1464 (G4/place_occupied/place_occupied)
- pseudo_real_g4_dual_arm_mismatch_001 score=0.1123 (G4/dual_arm_mismatch/dual_arm_mismatch)
- pseudo_real_g3_place_occupied_success_001 score=0.0888 (G3/place_occupied/)

### exp_b7fb782456f2

- Query: `G4 place_occupied large_object task_chain sim_success_real_fail block sim_real_gap_memory`
- Top matches: 5
- pseudo_real_g4_place_occupied_failure_001 score=0.3678 (G4/place_occupied/place_occupied)
- pseudo_real_g4_transport_collision_001 score=0.1568 (G4/transport_collision/transport_collision)
- pseudo_real_g4_clean_success_001 score=0.1334 (G4/clean/)
- pseudo_real_g3_place_occupied_success_001 score=0.0715 (G3/place_occupied/)
- pseudo_real_g4_dual_arm_mismatch_001 score=0.064 (G4/dual_arm_mismatch/dual_arm_mismatch)

### pseudo_real_g3_grasp_miss_001

- Query: `G3 grasp_miss sortable_object task_chain grasp_miss warn pseudo_real_failure_case`
- Top matches: 5
- pseudo_real_g3_grasp_slip_001 score=0.0844 (G3/grasp_slip/grasp_slip)
- pseudo_real_g3_clean_success_001 score=0.0841 (G3/clean/)
- pseudo_real_g3_place_occupied_success_001 score=0.0814 (G3/place_occupied/)
- pseudo_real_g4_dual_arm_mismatch_001 score=0.0524 (G4/dual_arm_mismatch/dual_arm_mismatch)
- pseudo_real_g4_transport_collision_001 score=0.0367 (G4/transport_collision/transport_collision)

### pseudo_real_g3_grasp_slip_001

- Query: `G3 grasp_slip sortable_object task_chain grasp_slip warn pseudo_real_failure_case`
- Top matches: 5
- pseudo_real_g3_grasp_miss_001 score=0.0848 (G3/grasp_miss/grasp_miss)
- pseudo_real_g3_clean_success_001 score=0.0841 (G3/clean/)
- pseudo_real_g3_place_occupied_success_001 score=0.0814 (G3/place_occupied/)
- pseudo_real_g4_dual_arm_mismatch_001 score=0.0524 (G4/dual_arm_mismatch/dual_arm_mismatch)
- pseudo_real_g4_transport_collision_001 score=0.0367 (G4/transport_collision/transport_collision)

### pseudo_real_g4_dual_arm_mismatch_001

- Query: `G4 dual_arm_mismatch large_object task_chain dual_arm_mismatch warn pseudo_real_failure_case`
- Top matches: 5
- pseudo_real_g4_clean_success_001 score=0.0812 (G4/clean/)
- pseudo_real_g4_transport_collision_001 score=0.0698 (G4/transport_collision/transport_collision)
- pseudo_real_g4_place_occupied_failure_001 score=0.0671 (G4/place_occupied/place_occupied)
- pseudo_real_g3_grasp_miss_001 score=0.0532 (G3/grasp_miss/grasp_miss)
- pseudo_real_g3_grasp_slip_001 score=0.053 (G3/grasp_slip/grasp_slip)

### pseudo_real_g4_place_occupied_failure_001

- Query: `G4 place_occupied large_object task_chain place_occupied block sim_real_gap_memory`
- Top matches: 5
- pseudo_real_g4_transport_collision_001 score=0.1808 (G4/transport_collision/transport_collision)
- pseudo_real_g4_clean_success_001 score=0.161 (G4/clean/)
- pseudo_real_g3_place_occupied_success_001 score=0.1097 (G3/place_occupied/)
- pseudo_real_g3_clean_success_001 score=0.0804 (G3/clean/)
- pseudo_real_g4_dual_arm_mismatch_001 score=0.0792 (G4/dual_arm_mismatch/dual_arm_mismatch)
