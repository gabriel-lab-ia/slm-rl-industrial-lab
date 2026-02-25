slm-rl-industrial-lab

Purpose
Industrial robotics simulation project connecting a low-latency SLM RL controller (MathCore-based) to a ROS2 Jazzy bridge and a MuJoCo backend, with a future Isaac Lab backend planned under the same ROS2 contract.

Current state
- ROS2 Jazzy bridge operational (`wafer_cell_bringup`)
- MuJoCo visual backend operational (industrial scene, single active arm workflow)
- `math_lab` mode operational for kinematics and linear-algebra validation
- `IsaacLabWaferCellEnv` (Gym-like ROS2 environment) operational
- `SLMPolicy` integrates the real MathCore RL agent with MLP fallback
- End-to-end RL demo operational (ROS2 -> env -> policy -> MuJoCo)
- Stable ROS2 protocol: `wafer_cell_ros2_v1`

Architecture
- `MathCoreAgent` (base RL project): action inference (CUDA-capable)
- `SLMPolicy`: adapts native agent actions to N2 action space (`act_dim=18`)
- `IsaacLabWaferCellEnv`: publishes ROS2 commands, consumes state, computes reward
- `wafer_cell_sim_node`: backend routing (`stub`, `mujoco`, `isaaclab` skeleton)
- MuJoCo: visual simulation, physics stepping, task-space validation backend

Base RL brain integration
Default `SLMPolicy` import target:
- `/home/aipowerisraelense/AI/ML-Edge-AI-Cinematic-Dashboard/rl-isaac-mathcore-humanoid`

Expected local checkpoint in this repository:
- `checkpoints/best.pt`

MuJoCo backend modes
- `industrial`: simplified industrial wafer-cell visual/control loop mode
- `math_lab`: kinematics and linear-algebra validation mode

math_lab features
- Geometric targets: `line`, `circle`, `spiral`, `figure8`, `rotate_extend`
- Damped Least Squares inverse kinematics: `dq = J^T (J J^T + λ²I)^-1 e`
- Null-space posture regularization (simple)
- Metrics export (JSON): `tracking_error_rms`, `manipulability_mean`, `jacobian_cond_mean`, `smoothness_mean`, `energy_mean`, `latency_ms_mean`, `latency_ms_max`

Current math_lab baseline (from `docs/results_math_lab_*.json`)
- `line`: tracking RMS ~ `0.6279`
- `circle`: tracking RMS ~ `0.6191`
- `spiral`: tracking RMS ~ `0.6138` (best current baseline)
- `figure8`: tracking RMS ~ `0.6221`

Repository structure (key paths)
- `src/slm_wafer_cell/envs/isaaclab_wafer_cell_env.py`
- `src/slm_wafer_cell/policy/slm_policy_loader.py`
- `src/slm_wafer_cell/demo_isaaclab_wafer_cell.py`
- `src/slm_wafer_cell/ros2/protocol.py`
- `ros2_ws/src/wafer_cell_bringup/wafer_cell_bringup/wafer_cell_sim_node.py`
- `src/slm_wafer_cell/sim/isaaclab_scene_loader.py` (skeleton)
- `scripts/manual_joint_probe.sh`
- `scripts/compare_math_lab_scenarios.sh`
- `docs/N2_ROS2_PROTOCOL.md`
- `docs/ROADMAP.md`

Requirements
- Pop!_OS / Ubuntu 24.04
- ROS2 Jazzy in `/opt/ros/jazzy`
- Python 3.12
- `venv --system-site-packages` (for `rclpy`)
- PyTorch with CUDA (recommended)
- MuJoCo Python package

Python setup (N2)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

ROS2 build
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run MuJoCo visual backend (math_lab)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python ros2_ws/src/wafer_cell_bringup/wafer_cell_bringup/wafer_cell_sim_node.py --ros-args   -p backend:=mujoco   -p mujoco_mode:=math_lab   -p mujoco_math_scenario:=spiral   -p debug_log_every_n:=25   -p headless:=false   -p publish_rate_hz:=50.0   -p mujoco_realtime:=true
```

Manual joint motion probe (fold / yaw / extend)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
bash scripts/manual_joint_probe.sh fold
bash scripts/manual_joint_probe.sh yaw
bash scripts/manual_joint_probe.sh extend
```

Run RL demo (MathCoreAgent through `SLMPolicy`)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python src/slm_wafer_cell/demo_isaaclab_wafer_cell.py   --checkpoint checkpoints/best.pt   --device cuda   --episodes 2   --steps-per-episode 150   --obs-dim 64   --act-dim 18
```

Compare math_lab scenarios (headless)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
bash scripts/compare_math_lab_scenarios.sh
```

ROS2 topics
- `/arm_1/joint_states`, `/arm_2/joint_states`, `/arm_3/joint_states`
- `/arm_1/joint_command`, `/arm_2/joint_command`, `/arm_3/joint_command`
- `/cell/wafer_state`
- `/cell/reset`
- `/cell/math_metrics` (math_lab metrics snapshot topic)

Publication notes
- `checkpoints/*.pt` is ignored by default and not versioned
- Local closeout manifests and temporary MuJoCo artifacts are ignored
- To publish a demo checkpoint intentionally: `git add -f checkpoints/best.pt`

Roadmap summary
- MuJoCo-specific RL fine-tuning (RL / imitation learning)
- Real industrial arm model (URDF/MJCF + meshes)
- Simulated sensors (contact, F/T, RGB-D)
- Web dashboard (FastAPI + HTML/WebSocket)
- Isaac Lab backend implementation while preserving the same ROS2 protocol
