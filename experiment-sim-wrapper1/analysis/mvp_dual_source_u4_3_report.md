# MVP Dual-Source Gap/Critic Report

- Config: `/home/mlx/mujoco/YOLO_World-SAM-GraspNet/experiment-sim-wrapper1/configs/mvp_dual_source_gap_critic_u4_3_v1.json`
- Real entries: 40
- Sim entries: 40
- Pair count: 40

## Candidate Scoring

### U4-3

- candidate_score: `0.5292`
- decision: `allow`
- failure_overlap_risk: `0.8571`
- gap_uncertainty: `0.7286`
- critic_risk: `0.0`
- calibration_applied: `True`
- calibration_confidence: `0.5966`
- support_evidence_count: `6`
- risk_evidence_count: `6`

| experience | source | validation | success | score | gap | critic | role |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `d47389f0` | real | real_executed | True | 1.8364 | 0.27 | 0.0 | sim_real_gap_memory |
| `6e44c002` | real | real_executed | True | 1.8364 | 0.27 | 0.0 | sim_real_gap_memory |
| `cfa94309` | real | real_executed | True | 1.8364 | 0.27 | 0.0 | sim_real_gap_memory |
| `54abc220` | real | real_executed | True | 1.8364 | 0.27 | 0.0 | sim_real_gap_memory |
| `b8010bc3` | real | real_executed | True | 1.8364 | 0.27 | 0.0 | sim_real_gap_memory |
| `b9e3930a` | real | real_executed | True | 1.8364 | 0.27 | 0.0 | sim_real_gap_memory |
| `1aef5a8a` | real | real_executed | False | 1.332 | 0.85 | 0.0 | sim_real_gap_memory |
| `df021909` | real | real_executed | False | 1.332 | 0.85 | 0.0 | sim_real_gap_memory |
| `461186a4` | real | real_executed | False | 1.332 | 0.85 | 0.0 | sim_real_gap_memory |
| `aa1bad88` | real | real_executed | False | 1.3278 | 0.85 | 0.0 | sim_real_gap_memory |
| `6f372361` | real | real_executed | False | 1.3278 | 0.85 | 0.0 | sim_real_gap_memory |
| `cde6e398` | real | real_executed | False | 1.3278 | 0.85 | 0.0 | sim_real_gap_memory |
