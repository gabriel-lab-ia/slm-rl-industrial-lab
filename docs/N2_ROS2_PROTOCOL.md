N2 ROS2 Protocol

Protocol version
- `wafer_cell_ros2_v1`

Purpose
This protocol defines the stable ROS2 interface between the simulation backend (`stub`, `mujoco`, future `isaaclab`) and the RL environment / policy runtime.

Topics
- `/arm_1/joint_states`
- `/arm_2/joint_states`
- `/arm_3/joint_states`
- `/arm_1/joint_command`
- `/arm_2/joint_command`
- `/arm_3/joint_command`
- `/cell/wafer_state`
- `/cell/math_metrics` (optional, active in `math_lab` mode)

Services
- `/cell/reset` (`std_srvs/Trigger`)

`/cell/wafer_state` message type
- `std_msgs/Float32MultiArray`

`/cell/wafer_state` layout (v1, 18 floats)
1. `wafer_x`
2. `wafer_y`
3. `wafer_z`
4. `wafer_roll`
5. `wafer_pitch`
6. `wafer_yaw`
7. `wafer_process_state`
8. `wafer_quality_score`
9. `coordination_score`
10. `contamination_risk`
11. `vibration_index`
12. `stage_norm`
13. `progress_norm`
14. `temperature_norm`
15. `pull_rate_norm`
16. `rotation_norm`
17. `energy_norm`
18. `cycle_count_norm_proxy`

math_lab compatibility note
In `math_lab` mode, the payload length remains `18` for compatibility. Selected fields are repurposed internally to carry normalized task-space and kinematic validation values. Full metrics are exported to JSON and can also be streamed through `/cell/math_metrics`.

Design rule
Backend implementations may change (`stub`, `mujoco`, `isaaclab`) without breaking the RL environment as long as topic names, service names, and `/cell/wafer_state` length/order remain protocol-compatible.
