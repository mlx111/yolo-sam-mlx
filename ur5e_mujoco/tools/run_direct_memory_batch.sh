#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

python run_experiment_batch.py \
  --config configs/ur5e_direct_memory_batch_5trials_v1.json \
  --no-viewer
