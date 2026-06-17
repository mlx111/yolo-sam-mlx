# MVP Dual-Source Gap/Critic Report

- Config: `/home/mlx/mujoco/YOLO_World-SAM-GraspNet/experiment-sim-wrapper1/configs/mvp_dual_source_gap_critic_u3_u4_v1.json`
- Real entries: 8
- Sim entries: 8
- Pair count: 8

## Candidate Scoring

### U3-4

- candidate_score: `0.4148`
- decision: `rewrite_recommended`
- failure_overlap_risk: `0.4286`
- gap_uncertainty: `0.3643`
- critic_risk: `0.0`
- calibration_applied: `True`
- calibration_confidence: `0.7077`

| experience | source | validation | success | score | gap | critic | role |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `165d8af4` | simulation | simulation_validated | True | 1.77 | 0.0 | 0.0 | synthetic_sim_prior |
| `ff392fe0` | real | real_executed | False | 1.327 | 0.85 | 0.0 | sim_real_gap_memory |

### U4-2

- candidate_score: `0.7216`
- decision: `prefer`
- failure_overlap_risk: `0.0`
- gap_uncertainty: `0.15`
- critic_risk: `0.0`
- calibration_applied: `True`
- calibration_confidence: `0.6843`

| experience | source | validation | success | score | gap | critic | role |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `106a24b2` | real | real_executed | True | 1.822 | 0.35 | 0.0 | sim_real_gap_memory |
| `300d605b` | simulation | simulation_validated | True | 1.775 | 0.0 | 0.0 | synthetic_sim_prior |
| `76311216` | simulation | simulation_validated | True | 1.75 | 0.0 | 0.0 | synthetic_sim_prior |
| `0c49951d` | real | real_executed | False | 1.307 | 0.85 | 0.0 | sim_real_gap_memory |
