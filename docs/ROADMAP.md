N2 Roadmap

Phase 1 (Completed)
- ROS2 Jazzy bridge implemented with topics and reset service
- Deterministic `stub` backend implemented (3 arms + wafer/process state)
- Gym-like ROS2 environment implemented
- End-to-end RL demo implemented with metrics and latency logging
- Stable ROS2 contract defined: `wafer_cell_ros2_v1`
- MuJoCo backend integrated
- `math_lab` mode implemented (DLS IK + trajectory validation + metrics)
- MathCore RL integration via `SLMPolicy` implemented

Phase 2 (Current Focus)
- Tune DLS IK gains and regularization using `math_lab` metrics
- Compare `line/circle/spiral/figure8` trajectories systematically
- Improve robot kinematic realism and visual fidelity in MuJoCo
- Prepare data logging for MuJoCo-specific fine-tuning

Phase 3 (Near-Term Upgrades)
- Add structured contact / force proxies and additional ROS2 telemetry
- Add a baseline controller comparison (DLS vs RL policy)
- Add episode logging for imitation learning and RL fine-tuning datasets
- Add web dashboard for metrics streaming and scenario control

Phase 4 (Industrial-Grade Simulation Expansion)
- Replace simplified arm geometry with a real industrial robot model (URDF/MJCF + meshes)
- Add gripper/end-effector variants (pick/place, process tool, vacuum handling)
- Add realistic wafer handling tasks and multi-object scenes
- Add collision-rich scenarios and dynamic obstacles

Phase 5 (Isaac Lab Backend)
- Implement real `IsaacLabSceneLoader`
- Implement `backend:=isaaclab` in `wafer_cell_sim_node.py`
- Preserve `wafer_cell_ros2_v1` while swapping MuJoCo backend for Isaac Lab
- Validate metric and control parity across backends
