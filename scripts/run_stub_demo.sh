#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python src/slm_wafer_cell/demo_isaaclab_wafer_cell.py \
  --checkpoint checkpoints/best.pt \
  --device cuda \
  --episodes "${EPISODES:-2}" \
  --steps-per-episode "${STEPS_PER_EPISODE:-100}" \
  --obs-dim "${OBS_DIM:-64}" \
  --act-dim "${ACT_DIM:-18}"
