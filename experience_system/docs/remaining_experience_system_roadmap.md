# Remaining Experience-System Roadmap

This document tracks work that is still genuinely remaining. Items that are
already implemented have been removed from the roadmap and should be documented
in `paper_implementation_alignment.md` or
`universal_experience_memory_current_implementation.md` instead.

## Current Baseline

The current `experience_system` already supports:

- Universal `ExperienceEntry` schema for simulation, pseudo-real, real-format,
  and real-episode imports.
- Real episode templates and collection guide for RGB-D, chassis lidar, wrist
  force, keyframes, logs, and result metadata.
- Real episode import report with validation, missing-reference checks, sensor
  coverage, optional import, write-policy records, and memory-partition counts.
- Sensor evidence quality gate that records completeness, risk-signal, and
  missing-reference signals in `memory_gate.explanation`.
- Sensor-derived sim-real gap extraction from RGB-D, lidar, wrist-force, and
  timestamp summaries for real-format episodes.
- Sim-real pair and gap signatures.
- Gap-derived sandbox calibration as object initial-pose shift and risk
  penalty.
- Candidate memory scoring with Top-O failure risk.
- Candidate sandbox rollout with motion-level critic.
- LLM-generated compact lessons with quality audit.
- Stage-aware retrieval and deterministic planner-context construction.
- Visual keyframe indexing and visual retrieval ablation.
- Repeated writeback benchmark tooling.
- Safety stress and harder adversarial safety stress reports.
- Real-format / pseudo-real evidence pack.
- Sandbox calibration ablation.
- Paper evidence summary generation with sensor evidence and sensor-gap claim
  boundary row.

The strongest safe paper claim remains:

```text
The system ranks anomaly-recovery candidates using memory support, failure risk,
stage-specific retrieval evidence, gap-calibrated sandbox rollout, and
motion-level critic feedback.
```

Avoid claiming real-robot success-rate improvement, learned critic, full digital
twin, or statistically proven long-horizon improvement until true real-robot
reports exist.

## Active Sandbox Fidelity Roadmap

The sandbox rollout layer is functional but intentionally not claimed as a full
digital twin. Continue sandbox-specific optimization in:

```text
experience_system/docs/sandbox_fidelity_optimization_roadmap.md
```

The next non-real-data step is sandbox initial-state construction from existing
simulation/pseudo-real episodes, followed by lightweight uncertainty sweeps and
contact/force-proxy critic metrics.

## Deferred Until Real Data Exists

These items should wait for actual robot logs:

- Claiming real-robot success-rate improvement.
- Claiming sensor-derived calibration improves real execution.
- Estimating statistically meaningful real-world failure reduction.
- Training or tuning thresholds from real wrist-force/lidar distributions.
- Reporting real sensor retrieval impact on candidate selection.

## Recommended Next Step

No non-real-data sensor-memory roadmap item remains in this document. The next
meaningful step is to import actual robot episodes and regenerate the real
episode import report, real-format evidence pack, and paper evidence summary so
the sensor row can move from `partial` to `supported`.
