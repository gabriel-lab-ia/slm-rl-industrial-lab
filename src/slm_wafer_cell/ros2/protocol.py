"""Stable ROS2 contract for the wafer-cell bridge (stub and future Isaac Lab backend)."""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = "wafer_cell_ros2_v1"
ARM_COUNT = 3
JOINTS_PER_ARM_DEFAULT = 6
ACTION_DIM_DEFAULT = ARM_COUNT * JOINTS_PER_ARM_DEFAULT

TOPICS = {
    "arm_1_joint_states": "/arm_1/joint_states",
    "arm_2_joint_states": "/arm_2/joint_states",
    "arm_3_joint_states": "/arm_3/joint_states",
    "arm_1_joint_command": "/arm_1/joint_command",
    "arm_2_joint_command": "/arm_2/joint_command",
    "arm_3_joint_command": "/arm_3/joint_command",
    "wafer_state": "/cell/wafer_state",
    "reset_service": "/cell/reset",
}

# Float32MultiArray /cell/wafer_state layout (stable contract v1)
WAFER_STATE_FIELDS = [
    "wafer_x",
    "wafer_y",
    "wafer_z",
    "wafer_roll",
    "wafer_pitch",
    "wafer_yaw",
    "wafer_process_state",
    "wafer_quality_score",
    "coordination_score",
    "contamination_risk",
    "vibration_index",
    "stage_norm",
    "progress_norm",
    "temperature_norm",
    "pull_rate_norm",
    "rotation_norm",
    "energy_norm",
    "cycle_count_norm_proxy",
]
WAFER_STATE_INDEX = {name: i for i, name in enumerate(WAFER_STATE_FIELDS)}
WAFER_STATE_DIM = len(WAFER_STATE_FIELDS)


@dataclass(frozen=True)
class WaferStateV1:
    wafer_x: float
    wafer_y: float
    wafer_z: float
    wafer_roll: float
    wafer_pitch: float
    wafer_yaw: float
    wafer_process_state: float
    wafer_quality_score: float
    coordination_score: float
    contamination_risk: float
    vibration_index: float
    stage_norm: float
    progress_norm: float
    temperature_norm: float
    pull_rate_norm: float
    rotation_norm: float
    energy_norm: float
    cycle_count_norm_proxy: float
