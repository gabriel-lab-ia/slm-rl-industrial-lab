# N2 ROS2 Protocol (Stub/Isaac Lab Bridge)

Versão atual: `wafer_cell_ros2_v1`

## Tópicos
- `/arm_1/joint_states`, `/arm_2/joint_states`, `/arm_3/joint_states`
- `/arm_1/joint_command`, `/arm_2/joint_command`, `/arm_3/joint_command`
- `/cell/wafer_state`
- serviço `/cell/reset` (`std_srvs/Trigger`)

## `/cell/wafer_state` (`std_msgs/Float32MultiArray`)
Layout fixo (v1):
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

## Próxima etapa (Isaac Lab real)
- `wafer_cell_sim_node.py` mantém os mesmos tópicos.
- `isaaclab_scene_loader.py` substitui somente o backend interno.
- `IsaacLabWaferCellEnv` permanece estável (sem mudanças no contrato ROS2).
