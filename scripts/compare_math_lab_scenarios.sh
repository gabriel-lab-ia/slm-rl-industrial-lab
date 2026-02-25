#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIOS=(line circle spiral figure8)
EPISODES="${EPISODES:-1}"
STEPS="${STEPS:-120}"
RATE="${RATE:-50.0}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT"
source .venv/bin/activate
set +u
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

mkdir -p docs

for sc in "${SCENARIOS[@]}"; do
  metrics="docs/results_math_lab_${sc}.json"
  rm -f "$metrics"
  echo "[run] scenario=$sc"

  python ros2_ws/src/wafer_cell_bringup/wafer_cell_bringup/wafer_cell_sim_node.py --ros-args \
    -p backend:=mujoco \
    -p mujoco_mode:=math_lab \
    -p mujoco_math_scenario:=$sc \
    -p mujoco_metrics_path:="$metrics" \
    -p headless:=true \
    -p publish_rate_hz:=$RATE \
    -p mujoco_realtime:=false &
  NODE_PID=$!

  cleanup_node() {
    kill "$NODE_PID" 2>/dev/null || true
  }
  trap cleanup_node EXIT

  sleep 1.2

  python src/slm_wafer_cell/demo_isaaclab_wafer_cell.py \
    --checkpoint checkpoints/best.pt \
    --device "$DEVICE" \
    --episodes "$EPISODES" \
    --steps-per-episode "$STEPS" \
    --obs-dim 64 \
    --act-dim 18 >"/tmp/math_lab_demo_${sc}.log" 2>&1 || true

  kill "$NODE_PID" 2>/dev/null || true
  wait "$NODE_PID" 2>/dev/null || true
  trap - EXIT

  echo "[done] scenario=$sc metrics=$metrics"
done

python - <<'PY'
import json
from pathlib import Path
rows = []
for sc in ['line','circle','spiral','figure8']:
    p = Path('docs') / f'results_math_lab_{sc}.json'
    if not p.exists():
        rows.append((sc, None, None, None, None))
        continue
    d = json.loads(p.read_text())
    rows.append((
        sc,
        d.get('tracking_error_rms'),
        d.get('manipulability_mean'),
        d.get('jacobian_cond_mean'),
        d.get('latency_ms_mean'),
    ))
print('\nScenario comparison (math_lab)')
print('scenario   tracking_error_rms   manipulability_mean   jacobian_cond_mean   latency_ms_mean')
for sc, e, m, c, l in rows:
    def fmt(x):
        return 'NA' if x is None else f'{x:.6f}'
    print(f'{sc:<9} {fmt(e):>18} {fmt(m):>22} {fmt(c):>21} {fmt(l):>16}')
PY
