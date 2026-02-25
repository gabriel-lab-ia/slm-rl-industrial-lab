#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger

try:
    from slm_wafer_cell.ros2.protocol import PROTOCOL_VERSION, TOPICS
except Exception:
    PROTOCOL_VERSION = "wafer_cell_ros2_v1"  # fallback if package not on PYTHONPATH
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

try:
    from slm_wafer_cell.sim.isaaclab_scene_loader import IsaacLabSceneLoader, WaferCellSceneConfig
except Exception:
    IsaacLabSceneLoader = None  # type: ignore
    WaferCellSceneConfig = None  # type: ignore

try:
    import mujoco  # type: ignore
    from mujoco import viewer as mujoco_viewer  # type: ignore
except Exception:
    mujoco = None  # type: ignore
    mujoco_viewer = None  # type: ignore


class _InternalStubBackend:
    """Deterministic, smooth, ROS2-friendly wafer-cell stub backend."""

    def __init__(self) -> None:
        self.t = 0.0
        self.dt = 0.02
        self.joint_count = 6
        self._cmd = {f"arm_{i}": [0.0] * self.joint_count for i in (1, 2, 3)}
        self._pos = {k: [0.0] * self.joint_count for k in self._cmd}
        self._vel = {k: [0.0] * self.joint_count for k in self._cmd}
        self._stage = 0.0
        self._progress = 0.0
        self._cycle_count = 0
        self._joint_phase = {
            "arm_1": [0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
            "arm_2": [0.2, 0.5, 0.8, 1.1, 1.4, 1.7],
            "arm_3": [0.4, 0.7, 1.0, 1.3, 1.6, 1.9],
        }
        self._wafer_pose = [0.0, 0.0, 0.80, 0.0, 0.0, 0.0]
        self._wafer_process_state = 0.15
        self._wafer_quality_score = 0.85
        self._aux = {
            "temperature_norm": 0.75,
            "pull_rate_norm": 0.55,
            "rotation_norm": 0.45,
            "contamination_risk": 0.08,
            "vibration_index": 0.06,
            "coordination_score": 0.82,
            "energy_norm": 0.20,
        }

    def reset(self) -> None:
        self.t = 0.0
        self._stage = 0.0
        self._progress = 0.0
        self._cycle_count = 0
        for k in self._cmd:
            self._cmd[k] = [0.0] * self.joint_count
            self._pos[k] = [0.0] * self.joint_count
            self._vel[k] = [0.0] * self.joint_count
        self._wafer_pose = [0.0, 0.0, 0.80, 0.0, 0.0, 0.0]
        self._wafer_process_state = 0.15
        self._wafer_quality_score = 0.85
        self._aux.update(
            {
                "temperature_norm": 0.75,
                "pull_rate_norm": 0.55,
                "rotation_norm": 0.45,
                "contamination_risk": 0.08,
                "vibration_index": 0.06,
                "coordination_score": 0.82,
                "energy_norm": 0.20,
            }
        )

    def apply_joint_command(self, arm: str, cmd: list[float]) -> None:
        vals = list(cmd[: self.joint_count]) + [0.0] * max(0, self.joint_count - len(cmd))
        self._cmd[arm] = vals[: self.joint_count]

    def step(self) -> None:
        self.t += self.dt
        all_positions: list[float] = []
        all_velocities: list[float] = []
        for arm in ("arm_1", "arm_2", "arm_3"):
            for j in range(self.joint_count):
                target = float(self._cmd[arm][j])
                phase = self._joint_phase[arm][j]
                carrier = 0.12 * math.sin(0.6 * self.t + phase)
                micro = 0.03 * math.sin(1.7 * self.t + 0.5 * phase)
                guided_target = max(-1.0, min(1.0, target + carrier + micro))
                vel = 2.2 * (guided_target - self._pos[arm][j]) - 0.12 * self._vel[arm][j]
                self._vel[arm][j] = vel
                self._pos[arm][j] += self.dt * vel
                all_positions.append(self._pos[arm][j])
                all_velocities.append(self._vel[arm][j])

        pos_var = float(_variance(all_positions))
        vel_var = float(_variance(all_velocities))
        cmd_energy = float(sum(v * v for vals in self._cmd.values() for v in vals) / (3 * self.joint_count))
        coordination = max(0.0, min(1.0, 0.92 - 0.35 * pos_var - 0.08 * vel_var))

        process_gain = max(0.0, 0.010 + 0.010 * coordination - 0.004 * cmd_energy)
        self._progress += process_gain
        if self._progress >= 1.0:
            self._progress = 0.0
            self._stage = min(1.0, self._stage + 0.2)
            self._cycle_count += 1
            if self._stage >= 1.0:
                self._stage = 0.0

        self._wafer_pose[0] = 0.015 * math.sin(0.4 * self.t)
        self._wafer_pose[1] = 0.012 * math.cos(0.35 * self.t)
        self._wafer_pose[2] = 0.80 + 0.005 * math.sin(0.2 * self.t)
        self._wafer_pose[3] = 0.01 * math.sin(0.5 * self.t)
        self._wafer_pose[4] = 0.01 * math.cos(0.45 * self.t)
        self._wafer_pose[5] = 0.25 * self._stage + 0.10 * self._progress + 0.03 * math.sin(0.3 * self.t)

        temp = 0.72 + 0.05 * math.sin(0.25 * self.t + 0.2)
        pull = 0.53 + 0.06 * math.cos(0.33 * self.t + 0.4)
        rot = 0.47 + 0.05 * math.sin(0.28 * self.t + 0.6)
        contamination = max(0.0, min(1.0, 0.05 + 0.03 * math.sin(0.19 * self.t) + 0.03 * cmd_energy))
        vibration = max(0.0, min(1.0, 0.04 + 0.02 * math.cos(0.41 * self.t) + 0.04 * vel_var))
        wafer_process_state = max(0.0, min(1.0, 0.15 + 0.55 * self._stage + 0.25 * self._progress))
        wafer_quality = max(0.0, min(1.0, 0.90 * coordination - 0.40 * contamination - 0.15 * vibration + 0.15))
        self._wafer_process_state = wafer_process_state
        self._wafer_quality_score = wafer_quality
        self._aux.update(
            {
                "temperature_norm": max(0.0, min(1.0, temp)),
                "pull_rate_norm": max(0.0, min(1.0, pull)),
                "rotation_norm": max(0.0, min(1.0, rot)),
                "contamination_risk": contamination,
                "vibration_index": vibration,
                "coordination_score": coordination,
                "energy_norm": max(0.0, min(1.0, 0.12 + 0.45 * cmd_energy)),
            }
        )

    def read_joint_state(self, arm: str) -> dict[str, list[float]]:
        return {
            "name": [f"{arm}_joint_{i+1}" for i in range(self.joint_count)],
            "position": list(self._pos[arm]),
            "velocity": list(self._vel[arm]),
        }

    def read_wafer_state(self) -> list[float]:
        return [
            *[float(v) for v in self._wafer_pose],
            float(self._wafer_process_state),
            float(self._wafer_quality_score),
            float(self._aux["coordination_score"]),
            float(self._aux["contamination_risk"]),
            float(self._aux["vibration_index"]),
            float(self._stage),
            float(self._progress),
            float(self._aux["temperature_norm"]),
            float(self._aux["pull_rate_norm"]),
            float(self._aux["rotation_norm"]),
            float(self._aux["energy_norm"]),
            float(min(1.0, self._cycle_count / 20.0)),
        ]


class _MujocoWaferCellBackend:
    """MuJoCo backend with robust 3-arm industrial wafer cell, scripted contact targets and process indicators.

    Keeps ROS2 contract `wafer_cell_ros2_v1` unchanged. RL actions modulate the trajectory/IK targets.
    """

    def __init__(self, *, dt: float = 0.02, headless: bool = True, realtime: bool = True) -> None:
        if mujoco is None:
            raise RuntimeError("mujoco python package not installed")
        self.dt = float(dt)
        self.headless = bool(headless)
        self.realtime = bool(realtime)
        self.joint_count = 6
        self.t = 0.0
        self._stage = 0.0
        self._progress = 0.0
        self._cycle_count = 0
        self._cmd = {f"arm_{i}": [0.0] * self.joint_count for i in (1, 2, 3)}
        self.control_arms = ("arm_1",)
        self.hidden_arms = ("arm_2", "arm_3")
        self._build_model()
        self.viewer = None
        if not self.headless and mujoco_viewer is not None:
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            try:
                self.viewer.cam.lookat[:] = [-0.02, 0.00, 0.63]
                self.viewer.cam.distance = 1.12
                self.viewer.cam.azimuth = 122.0
                self.viewer.cam.elevation = -16.0
            except Exception:
                pass
        self._last_wall = time.perf_counter()
        self.reset()

    def _build_model(self) -> None:
        xml = self._build_xml_string()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt / 4.0
        self.arm_joint_names = {
            arm: [f"{arm}_joint_{j}" for j in range(1, self.joint_count + 1)]
            for arm in ("arm_1", "arm_2", "arm_3")
        }
        self.arm_joint_ids = {
            arm: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
            for arm, names in self.arm_joint_names.items()
        }
        self.arm_qpos_adr = {arm: [int(self.model.jnt_qposadr[j]) for j in ids] for arm, ids in self.arm_joint_ids.items()}
        self.arm_qvel_adr = {arm: [int(self.model.jnt_dofadr[j]) for j in ids] for arm, ids in self.arm_joint_ids.items()}
        self.arm_actuator_ids = {
            arm: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_act_{j}") for j in range(1, self.joint_count + 1)]
            for arm in ("arm_1", "arm_2", "arm_3")
        }
        self.arm_base_body_ids = {
            arm: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_base")
            for arm in ("arm_1", "arm_2", "arm_3")
        }
        self.arm_tool_site_ids = {
            arm: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_tool")
            for arm in ("arm_1", "arm_2", "arm_3")
        }
        self.arm_base_yaw = {
            "arm_1": math.pi,
            "arm_2": 0.80 * math.pi,
            "arm_3": 0.20 * math.pi,
        }

        self.wafer_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wafer_free")
        self.wafer_qpos_adr = int(self.model.jnt_qposadr[self.wafer_joint_id])
        self.wafer_qvel_adr = int(self.model.jnt_dofadr[self.wafer_joint_id])
        self.wafer_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wafer")
        self.process_hotspot_joint_id = None
        self.process_hotspot_qpos_adr = None
        self.process_hotspot_qvel_adr = None
        self.spark_joint_id = None
        self.spark_qpos_adr = None
        self.spark_qvel_adr = None
        self.spark_geom_id = None
        self.hotspot_geom_id = None
        try:
            self.process_hotspot_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "process_hotspot_free")
            self.process_hotspot_qpos_adr = int(self.model.jnt_qposadr[self.process_hotspot_joint_id])
            self.process_hotspot_qvel_adr = int(self.model.jnt_dofadr[self.process_hotspot_joint_id])
            self.spark_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "process_spark_free")
            self.spark_qpos_adr = int(self.model.jnt_qposadr[self.spark_joint_id])
            self.spark_qvel_adr = int(self.model.jnt_dofadr[self.spark_joint_id])
            self.spark_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "process_spark_geom")
            self.hotspot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "process_hotspot_geom")
        except Exception:
            # process visuals removed/disabled from MJCF; keep backend functional.
            pass
        self.track_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"track_dot_{i}") for i in range(24)
        ]
        self._track_local = []
        for i in range(24):
            r = 0.02 + 0.10 * (i / 23.0)
            a = 2.8 * math.pi * (i / 23.0)
            self._track_local.append((r * math.cos(a), r * math.sin(a), 0.010))

    def diagnostics_summary(self) -> dict[str, Any]:
        joint_ranges = {}
        actuator_ctrlranges = {}
        for arm in ("arm_1", "arm_2", "arm_3"):
            jr = []
            for jid in self.arm_joint_ids[arm]:
                has_lim = bool(self.model.jnt_limited[jid]) if hasattr(self.model, 'jnt_limited') else True
                rng = self.model.jnt_range[jid] if has_lim else (-float('inf'), float('inf'))
                jr.append([float(rng[0]), float(rng[1])])
            joint_ranges[arm] = jr
            cr = []
            for aid in self.arm_actuator_ids[arm]:
                r = self.model.actuator_ctrlrange[aid]
                cr.append([float(r[0]), float(r[1])])
            actuator_ctrlranges[arm] = cr
        return {
            'backend': 'mujoco',
            'mode': 'industrial',
            'nq': int(self.model.nq),
            'nv': int(self.model.nv),
            'nu': int(self.model.nu),
            'timestep': float(self.model.opt.timestep),
            'arm_joint_ranges': joint_ranges,
            'arm_actuator_ctrlranges': actuator_ctrlranges,
            'has_wafer_free_joint': True,
        }


    def _build_xml_string(self) -> str:
        def arm_xml(name: str, base_xyz: tuple[float, float, float], base_yaw: float) -> str:
            x, y, z = base_xyz
            yaw_deg = base_yaw * 180.0 / math.pi
            c = '0.36 0.37 0.40 1'  # visible industrial body
            white = '0.62 0.64 0.68 1'
            dark = '0.18 0.19 0.22 1'
            parts = [
                f'<body name="{name}_base" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw_deg:.3f}">',
                '  <geom type="cylinder" size="0.12 0.10" rgba="0.04 0.04 0.05 1"/>',
                '  <geom type="cylinder" pos="0 0 0.18" size="0.09 0.18" rgba="0.05 0.05 0.06 1"/>',
                f'  <geom type="capsule" fromto="0 0 0.36 0.16 0 0.48" size="0.06" rgba="{white}"/>',
                f'  <geom type="capsule" fromto="0.16 0 0.48 0.40 0 0.54" size="0.05" rgba="{white}"/>',
                f'  <geom type="box" pos="0.42 0 0.55" size="0.06 0.05 0.05" rgba="{c}"/>',
                f'  <geom type="capsule" fromto="0.44 0 0.55 0.56 0 0.45" size="0.04" rgba="{white}"/>',
                f'  <body name="{name}_kin_root" pos="0 0 0.36">',
                f'    <joint name="{name}_joint_1" type="hinge" axis="0 0 1" range="-3.0 3.0" damping="5.0"/>',
                f'    <body name="{name}_link_1" pos="0.00 0.00 0.00">',
                f'      <geom type="capsule" fromto="0 0 0 0.23 0 0.15" size="0.055" rgba="{white}"/>',
                f'      <geom type="sphere" pos="0 0 0" size="0.070" rgba="{dark}"/>',
                f'      <body name="{name}_link_2" pos="0.23 0 0.15">',
                f'        <joint name="{name}_joint_2" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="4.0"/>',
                f'        <geom type="capsule" fromto="0 0 0 0.28 0 0.05" size="0.045" rgba="{white}"/>',
                f'        <geom type="sphere" pos="0 0 0" size="0.060" rgba="{dark}"/>',
                f'        <body name="{name}_link_3" pos="0.28 0 0.05">',
                f'          <joint name="{name}_joint_3" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="3.0"/>',
                f'          <geom type="capsule" fromto="0 0 0 0.20 0 -0.08" size="0.038" rgba="{white}"/>',
                f'          <geom type="sphere" pos="0 0 0" size="0.050" rgba="{dark}"/>',
                f'          <body name="{name}_link_4" pos="0.20 0 -0.08">',
                f'            <joint name="{name}_joint_4" type="hinge" axis="1 0 0" range="-2.4 2.4" damping="2.0"/>',
                f'            <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.028" rgba="{c}"/>',
                f'            <body name="{name}_link_5" pos="0.10 0 0">',
                f'              <joint name="{name}_joint_5" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="2.0"/>',
                f'              <geom type="capsule" fromto="0 0 0 0.07 0 0" size="0.024" rgba="{c}"/>',
                f'              <body name="{name}_link_6" pos="0.07 0 0">',
                f'                <joint name="{name}_joint_6" type="hinge" axis="1 0 0" range="-3.0 3.0" damping="1.5"/>',
                '                <geom type="box" pos="0.040 0 0" size="0.040 0.020 0.018" rgba="0.04 0.04 0.05 1"/>',
                '                <geom type="capsule" fromto="0.03 0.020 0 0.16 0.020 0" size="0.008" rgba="0.95 0.45 0.08 1"/>',
                '                <geom type="capsule" fromto="0.03 -0.020 0 0.16 -0.020 0" size="0.008" rgba="0.95 0.45 0.08 1"/>',
                '                <geom type="capsule" fromto="0.16 -0.020 0 0.16 0.020 0" size="0.005" rgba="0.95 0.55 0.12 1"/>',
                f'                <site name="{name}_tool" pos="0.16 0 0" size="0.020" rgba="1.0 0.35 0.10 1"/>',
                '              </body>',
                '            </body>',
                '          </body>',
                '        </body>',
                '      </body>',
                '    </body>',
                '  </body>',
                '</body>',
            ]
            return "\n".join(parts)

        actuators = []
        for arm in ("arm_1", "arm_2", "arm_3"):
            for j in range(1, 7):
                actuators.append(
                    f'<position name="{arm}_act_{j}" joint="{arm}_joint_{j}" kp="120" ctrlrange="-2.0 2.0" forcerange="-220 220"/>'
                )

        track_geoms = []
        for i in range(24):
            r = 0.02 + 0.10 * (i / 23.0)
            a = 2.8 * math.pi * (i / 23.0)
            x = r * math.cos(a)
            y = r * math.sin(a)
            track_geoms.append(
                f'<geom name="track_dot_{i}" type="sphere" pos="{x:.4f} {y:.4f} 0.008" size="0.0035" rgba="0.25 0.18 0.08 0.80" contype="0" conaffinity="0"/>'
            )

        xml = f"""
<mujoco model="wafer_cell_3arms">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.005"/>
  <visual>
    <headlight diffuse="0.80 0.82 0.88" ambient="0.18 0.18 0.20" specular="0.20 0.20 0.22"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.12 0.13 0.16" rgb2="0.06 0.07 0.09" width="256" height="256"/>
    <material name="metal" rgba="0.34 0.35 0.38 1"/>
    <material name="wafer_mat" rgba="0.10 0.12 0.16 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.4 2.2" dir="-0.25 0.15 -1" diffuse="1.25 1.25 1.30"/>
    <light pos="-0.8 0.8 1.6" dir="0.4 -0.4 -0.7" diffuse="0.65 0.70 0.85"/>
    <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.16 0.17 0.19 1"/>
    <body name="guard_panel" pos="-0.56 0.02 0.66">
      <geom type="box" size="0.03 0.65 0.46" rgba="0.22 0.23 0.25 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.03 0.65 0" size="0.01 0.02 0.46" rgba="0.80 0.62 0.08 1" contype="0" conaffinity="0"/>
    </body>
    <body name="positioner" pos="-0.08 0.02 0.56" euler="0 0.95 0">
      <geom type="cylinder" size="0.24 0.12" rgba="0.28 0.29 0.32 1"/>
      <geom type="cylinder" pos="0 0 0.08" size="0.17 0.025" rgba="0.18 0.19 0.21 1"/>
      <geom type="cylinder" pos="0 0 0.110" size="0.14 0.012" rgba="0.16 0.17 0.19 1"/>
    </body>
    <body name="cell_table" pos="0 0 0.33">
      <geom type="box" size="0.9 0.75 0.05" rgba="0.30 0.31 0.34 1"/>
      <body name="chuck_support" pos="-0.08 0.02 0.21">
        <geom type="box" size="0.12 0.10 0.10" rgba="0.24 0.25 0.28 1"/>
      </body>
      {arm_xml('arm_1', (0.35, 0.08, 0.00), 0.95 * math.pi)}
      {arm_xml('arm_2', (3.5, 3.5, -3.0), 0.78 * math.pi)}
      {arm_xml('arm_3', (-3.5, 3.5, -3.0), 0.22 * math.pi)}
    </body>
    <body name="wafer_carrier" pos="0 0 0" gravcomp="1">
      <joint name="wafer_free" type="free"/>
      <body name="wafer" pos="0 0 0">
        <geom type="cylinder" size="0.15 0.006" material="wafer_mat" rgba="0.18 0.20 0.24 1" contype="0" conaffinity="0"/>
        {' '.join(track_geoms)}
      </body>
    </body>
  </worldbody>
  <actuator>
    {' '.join(actuators)}
  </actuator>
</mujoco>
"""
        return xml

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0.0
        self._stage = 0.0
        self._progress = 0.0
        self._cycle_count = 0
        self._active_arm = 'arm_1'
        for arm in self._cmd:
            self._cmd[arm] = [0.0] * self.joint_count
        adr = self.wafer_qpos_adr
        q = _quat_from_rpy(0.0, 0.95, 0.0)
        self.data.qpos[adr : adr + 7] = [-0.08, 0.02, 0.67, q[0], q[1], q[2], q[3]]
        self.data.qvel[self.wafer_qvel_adr : self.wafer_qvel_adr + 6] = 0.0
        self._set_process_indicator([-0.08, 0.02, 0.77], q, spark_alpha=0.0)
        for gid in self.track_geom_ids:
            self.model.geom_rgba[gid] = [0.10, 0.04, 0.02, 0.65]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                pass
        self._last_wall = time.perf_counter()

    def apply_joint_command(self, arm: str, cmd: list[float]) -> None:
        vals = list(cmd[: self.joint_count]) + [0.0] * max(0, self.joint_count - len(cmd))
        self._cmd[arm] = [max(-1.8, min(1.8, float(v))) for v in vals[: self.joint_count]]

    def _wafer_world_pose(self) -> tuple[list[float], tuple[float, float, float, float]]:
        adr = self.wafer_qpos_adr
        q = self.data.qpos[adr : adr + 7]
        return [float(q[0]), float(q[1]), float(q[2])], (float(q[3]), float(q[4]), float(q[5]), float(q[6]))

    def _spiral_target_local(self, arm_idx: int) -> tuple[float, float, float]:
        phase = self.t * (1.1 + 0.15 * arm_idx) + (2.15 * arm_idx)
        spiral = 0.035 + 0.035 * (0.5 + 0.5 * math.sin(0.17 * self.t + 0.9 * arm_idx))
        r = spiral + 0.018 * math.sin(2.4 * phase)
        a = phase % (2.0 * math.pi)
        x = r * math.cos(a)
        y = r * math.sin(a)
        z = 0.002 + 0.0015 * math.sin(3.0 * phase)
        return x, y, z

    def _wafer_local_to_world(self, p_local: tuple[float, float, float], q_wafer: tuple[float, float, float, float], p_wafer: list[float]) -> tuple[float, float, float]:
        # Only need tilt+yaw rotation already encoded in quaternion; use mujoco rotation matrix conversion via manual formula.
        w, x, y, z = q_wafer
        px, py, pz = p_local
        r00 = 1 - 2*(y*y + z*z); r01 = 2*(x*y - z*w); r02 = 2*(x*z + y*w)
        r10 = 2*(x*y + z*w); r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z - x*w)
        r20 = 2*(x*z - y*w); r21 = 2*(y*z + x*w); r22 = 1 - 2*(x*x + y*y)
        wx = p_wafer[0] + r00*px + r01*py + r02*pz
        wy = p_wafer[1] + r10*px + r11*py + r12*pz
        wz = p_wafer[2] + r20*px + r21*py + r22*pz
        return (wx, wy, wz)

    def _solve_simple_ik(self, arm: str, target_world: tuple[float, float, float], rl_vec: list[float]) -> list[float]:
        base_pos = self.data.xpos[self.arm_base_body_ids[arm]]
        bx, by, bz = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
        tx, ty, tz = target_world
        dx, dy = tx - bx, ty - by
        yaw_base = self.arm_base_yaw[arm]
        # rotate target into base frame (yaw only)
        c = math.cos(-yaw_base); s = math.sin(-yaw_base)
        x_local = c*dx - s*dy
        y_local = s*dx + c*dy
        z_local = tz - (bz + 0.36)

        q1 = math.atan2(y_local, max(1e-6, x_local))
        r = max(0.05, (x_local*x_local + y_local*y_local) ** 0.5)
        L1, L2 = 0.28, 0.22
        rr = min(max(0.05, (r*r + z_local*z_local) ** 0.5), L1 + L2 - 1e-3)
        ca = max(-1.0, min(1.0, (L1*L1 + rr*rr - L2*L2) / (2*L1*rr)))
        cb = max(-1.0, min(1.0, (L1*L1 + L2*L2 - rr*rr) / (2*L1*L2)))
        base_elev = math.atan2(z_local, r)
        shoulder = base_elev + math.acos(ca) - 0.6
        elbow = math.pi - math.acos(cb) - 1.0
        wrist_pitch = -0.7 * shoulder - 0.6 * elbow
        wrist_roll = 0.35 * math.sin(self.t * 3.0 + (0.6 if arm == 'arm_1' else 1.7))
        wrist_yaw = 0.15 * math.sin(self.t * 4.0 + (1.5 if arm == 'arm_2' else 0.2))

        # RL modulation (real model influences path/attitude)
        m = (rl_vec + [0.0] * 6)[:6]
        q = [
            q1 + 0.35 * float(m[0]),
            shoulder + 0.25 * float(m[1]),
            elbow + 0.25 * float(m[2]),
            wrist_roll + 0.25 * float(m[3]),
            wrist_pitch + 0.25 * float(m[4]),
            wrist_yaw + 0.35 * float(m[5]),
        ]
        return [max(-1.9, min(1.9, v)) for v in q]

    def _set_process_indicator(self, pos_xyz: list[float], q: tuple[float, float, float, float], spark_alpha: float) -> None:
        if self.process_hotspot_qpos_adr is None or self.spark_qpos_adr is None:
            return
        h = self.process_hotspot_qpos_adr
        self.data.qpos[h:h+7] = [pos_xyz[0], pos_xyz[1], pos_xyz[2], q[0], q[1], q[2], q[3]]
        self.data.qvel[self.process_hotspot_qvel_adr:self.process_hotspot_qvel_adr+6] = 0.0
        s = self.spark_qpos_adr
        jitter = [0.006 * math.sin(13.0*self.t), 0.006 * math.cos(17.0*self.t), 0.006 * math.sin(11.0*self.t)]
        self.data.qpos[s:s+7] = [pos_xyz[0]+jitter[0], pos_xyz[1]+jitter[1], pos_xyz[2]+jitter[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.spark_qvel_adr:self.spark_qvel_adr+6] = 0.0
        if self.hotspot_geom_id is not None:
            self.model.geom_rgba[self.hotspot_geom_id] = [1.0, 0.35 + 0.12*math.sin(9.0*self.t), 0.06, 0.99]
        if self.spark_geom_id is not None:
            self.model.geom_rgba[self.spark_geom_id] = [1.0, 0.85, 0.25, max(0.0, min(1.0, spark_alpha))]

    def _update_track_visual(self, active_index: int) -> None:
        for i, gid in enumerate(self.track_geom_ids):
            if i <= active_index:
                self.model.geom_rgba[gid] = [1.0, 0.30, 0.05, 0.98]
            else:
                self.model.geom_rgba[gid] = [0.10, 0.04, 0.02, 0.60]

    def _update_process_metrics(self) -> None:
        all_positions = []
        all_velocities = []
        for arm in self.control_arms:
            qpos_adr = self.arm_qpos_adr[arm]
            qvel_adr = self.arm_qvel_adr[arm]
            all_positions.extend(float(self.data.qpos[a]) for a in qpos_adr)
            all_velocities.extend(float(self.data.qvel[a]) for a in qvel_adr)
        pos_var = _variance(all_positions)
        vel_var = _variance(all_velocities)
        cmd_energy = sum(v*v for arm, vals in self._cmd.items() if arm in self.control_arms for v in vals) / (max(1, len(self.control_arms)) * self.joint_count)
        coordination = max(0.0, min(1.0, 0.97 - 0.22*pos_var - 0.10*vel_var))
        process_gain = max(0.0, 0.010 + 0.015*coordination - 0.003*cmd_energy)
        self._progress += process_gain
        if self._progress >= 1.0:
            self._progress = 0.0
            self._stage = min(1.0, self._stage + 0.2)
            self._cycle_count += 1
            if self._stage >= 1.0:
                self._stage = 0.0
        self._temp = max(0.0, min(1.0, 0.76 + 0.05*math.sin(0.22*self.t)))
        self._pull = max(0.0, min(1.0, 0.52 + 0.04*math.cos(0.30*self.t)))
        self._rot = max(0.0, min(1.0, 0.62 + 0.08*math.sin(0.27*self.t + 0.2)))
        self._contam = max(0.0, min(1.0, 0.025 + 0.015*cmd_energy + 0.01*math.sin(0.19*self.t)))
        self._vib = max(0.0, min(1.0, 0.02 + 0.06*vel_var + 0.01*math.cos(0.44*self.t)))
        self._coord = coordination
        self._energy = max(0.0, min(1.0, 0.08 + 0.38*cmd_energy))
        self._proc_state = max(0.0, min(1.0, 0.12 + 0.55*self._stage + 0.30*self._progress))
        self._quality = max(0.0, min(1.0, 0.95*self._coord - 0.35*self._contam - 0.10*self._vib + 0.08))

    def step(self) -> None:
        # Wafer fixed on tilted positioner (kinematic override, no flying).
        adr = self.wafer_qpos_adr
        yaw = 0.20 * self._stage + 0.12 * self._progress + 0.35 * self.t
        q_wafer = _quat_from_rpy(0.0, 0.95, yaw)
        wafer_pos = [-0.08, 0.02, 0.67]
        self.data.qpos[adr:adr+7] = [wafer_pos[0], wafer_pos[1], wafer_pos[2], q_wafer[0], q_wafer[1], q_wafer[2], q_wafer[3]]
        self.data.qvel[self.wafer_qvel_adr:self.wafer_qvel_adr+6] = 0.0

        # Contact targets (spiral/circular) + simplified IK for each arm.
        arm_order = self.control_arms
        active_idx = 0
        self._active_arm = "arm_1"
        active_track = 0
        active_target_world = None
        for i, arm in enumerate(arm_order):
            local = self._spiral_target_local(i)
            target_world = self._wafer_local_to_world(local, q_wafer, wafer_pos)
            if arm == self._active_arm:
                active_target_world = target_world
                radius_now = (local[0]**2 + local[1]**2) ** 0.5
                active_track = int(min(23, max(0, round((radius_now - 0.02) / 0.10 * 23))))
            q_target = self._solve_simple_ik(arm, target_world, self._cmd[arm])
            for j, aid in enumerate(self.arm_actuator_ids[arm]):
                self.data.ctrl[aid] = q_target[j]

        # Zero hidden arms so only one mechatronic arm appears active in the cell demo.
        for arm in self.hidden_arms:
            for aid in self.arm_actuator_ids[arm]:
                self.data.ctrl[aid] = 0.0

        # Multi-substep
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt

        # Re-apply wafer kinematic pose AFTER stepping to avoid gravity drift/"flying" artifacts.
        self.data.qpos[adr:adr+7] = [wafer_pos[0], wafer_pos[1], wafer_pos[2], q_wafer[0], q_wafer[1], q_wafer[2], q_wafer[3]]
        self.data.qvel[self.wafer_qvel_adr:self.wafer_qvel_adr+6] = 0.0

        # Visual process indicators on active contact target
        if active_target_world is None:
            active_target_world = (wafer_pos[0], wafer_pos[1], wafer_pos[2] + 0.10)
        hotspot_pos = [active_target_world[0], active_target_world[1], active_target_world[2] + 0.004]
        spark_alpha = 0.55 + 0.40 * (0.5 + 0.5 * math.sin(22.0 * self.t))
        self._set_process_indicator(hotspot_pos, (1.0, 0.0, 0.0, 0.0), spark_alpha=spark_alpha)
        self._update_track_visual(active_track)

        mujoco.mj_forward(self.model, self.data)
        self._update_process_metrics()

        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                self.viewer = None

        if self.realtime:
            target = self._last_wall + self.dt
            now = time.perf_counter()
            sleep_dur = target - now
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            self._last_wall = max(target, time.perf_counter())

    def read_joint_state(self, arm: str) -> dict[str, list[float]]:
        if arm in self.hidden_arms:
            return {
                "name": [f"{arm}_joint_{i+1}" for i in range(self.joint_count)],
                "position": [0.0] * self.joint_count,
                "velocity": [0.0] * self.joint_count,
            }
        return {
            "name": list(self.arm_joint_names[arm]),
            "position": [float(self.data.qpos[a]) for a in self.arm_qpos_adr[arm]],
            "velocity": [float(self.data.qvel[a]) for a in self.arm_qvel_adr[arm]],
        }

    def read_wafer_state(self) -> list[float]:
        q = self.data.qpos[self.wafer_qpos_adr : self.wafer_qpos_adr + 7]
        x, y, z = float(q[0]), float(q[1]), float(q[2])
        w, qx, qy, qz = float(q[3]), float(q[4]), float(q[5]), float(q[6])
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * qy - qz * qx))))
        roll = math.atan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        return [
            x, y, z, roll, pitch, yaw,
            float(self._proc_state),
            float(self._quality),
            float(self._coord),
            float(self._contam),
            float(self._vib),
            float(self._stage),
            float(self._progress),
            float(self._temp),
            float(self._pull),
            float(self._rot),
            float(self._energy),
            float(min(1.0, self._cycle_count / 20.0)),
        ]

    def shutdown(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


class _MujocoMathLabBackend(_MujocoWaferCellBackend):
    """Math/kinematics validation mode on MuJoCo with DLS IK and geometric targets.

    Keeps the ROS2 wafer contract length unchanged, but repurposes some process fields with math metrics.
    Writes detailed metrics JSON on shutdown.
    """

    def __init__(
        self,
        *,
        dt: float = 0.02,
        headless: bool = True,
        realtime: bool = True,
        scenario: str = 'spiral',
        metrics_path: str | None = None,
    ) -> None:
        self.math_scenario = scenario
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self._tool_trail_world: list[tuple[float, float, float]] = []
        self._target_trail_world: list[tuple[float, float, float]] = []
        self._math_last = {
            'target_world': [0.0, 0.0, 0.0],
            'tool_world': [0.0, 0.0, 0.0],
            'error_vec': [0.0, 0.0, 0.0],
            'error_norm': 0.0,
            'jacobian_cond': 1.0,
            'manipulability': 0.0,
            'smoothness': 0.0,
            'control_energy': 0.0,
            'phase': 0.0,
            'latency_ms': 0.0,
        }
        self._acc = {
            'steps': 0,
            'err2_sum': 0.0,
            'manip_sum': 0.0,
            'cond_sum': 0.0,
            'smooth_sum': 0.0,
            'energy_sum': 0.0,
            'lat_ms_sum': 0.0,
            'lat_ms_max': 0.0,
        }
        super().__init__(dt=dt, headless=headless, realtime=realtime)
        if self.viewer is not None:
            try:
                self.viewer.cam.lookat[:] = [-0.02, 0.00, 0.63]
                self.viewer.cam.distance = 1.18
                self.viewer.cam.azimuth = 128.0
                self.viewer.cam.elevation = -18.0
            except Exception:
                pass

    def reset(self) -> None:
        super().reset()
        self._stage = 0.0
        self._progress = 0.0
        self._cycle_count = 0
        self._tool_trail_world.clear()
        self._target_trail_world.clear()
        for k in self._acc:
            self._acc[k] = 0 if k == 'steps' else 0.0

    def _scenario_target_local(self, phase: float) -> tuple[float, float, float]:
        sc = (self.math_scenario or 'spiral').lower()
        # Work envelope over the wafer/fixture in local wafer coordinates.
        if sc == 'line':
            u = math.sin(phase)
            return (0.11 * u, 0.0, 0.004)
        if sc == 'circle':
            return (0.09 * math.cos(phase), 0.09 * math.sin(phase), 0.004)
        if sc in ('figure8', 'figure-8', 'lemniscate'):
            return (0.10 * math.sin(phase), 0.06 * math.sin(2.0 * phase), 0.004)
        if sc in ('rotate_extend', 'rotate+extend', 'rotate-extend'):
            r = 0.04 + 0.06 * (0.5 + 0.5 * math.sin(0.5 * phase))
            return (r * math.cos(phase), r * math.sin(phase), 0.004 + 0.006 * math.sin(phase))
        # spiral (default)
        r = 0.03 + 0.08 * (0.5 + 0.5 * math.sin(0.22 * phase))
        return (r * math.cos(phase), r * math.sin(phase), 0.004 + 0.002 * math.sin(2.4 * phase))

    def _compute_dls_step(self, arm: str, target_world: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        dof_ids = self.arm_qvel_adr[arm]
        qpos_ids = self.arm_qpos_adr[arm]
        site_id = self.arm_tool_site_ids[arm]
        tool = np.array(self.data.site_xpos[site_id], dtype=np.float64)
        e = np.asarray(target_world, dtype=np.float64) - tool

        nv = int(self.model.nv)
        jacp = np.zeros((3, nv), dtype=np.float64)
        jacr = np.zeros((3, nv), dtype=np.float64)
        try:
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            J = jacp[:, dof_ids]
        except Exception:
            # Fallback rough Jacobian proxy
            J = np.eye(3, len(dof_ids), dtype=np.float64) * 0.1

        JJt = J @ J.T
        lam = 0.06
        I3 = np.eye(3, dtype=np.float64)
        try:
            y = np.linalg.solve(JJt + (lam * lam) * I3, e)
        except np.linalg.LinAlgError:
            y = np.linalg.pinv(JJt + (lam * lam) * I3) @ e
        dq = J.T @ y

        # Null-space posture regularization (bias to moderate posture near zero)
        q = np.array([float(self.data.qpos[a]) for a in qpos_ids], dtype=np.float64)
        J_pinv = J.T @ np.linalg.pinv(JJt + 1e-6 * I3)
        N = np.eye(len(dof_ids)) - J_pinv @ J
        dq_post = N @ (-0.10 * q)
        dq = dq + dq_post

        # Metrics
        try:
            svals = np.linalg.svd(J, compute_uv=False)
            smax = float(np.max(svals)) if svals.size else 0.0
            smin = float(np.min(svals)) if svals.size else 0.0
            cond = float(smax / max(1e-6, smin)) if svals.size else 1.0
            manipulability = float(np.sqrt(max(0.0, np.linalg.det(JJt))))
        except Exception:
            cond = 1.0
            manipulability = 0.0

        return dq.astype(np.float64), {
            'error_norm': float(np.linalg.norm(e)),
            'cond': cond,
            'manipulability': manipulability,
            'tool_x': float(tool[0]),
            'tool_y': float(tool[1]),
            'tool_z': float(tool[2]),
            'e_x': float(e[0]),
            'e_y': float(e[1]),
            'e_z': float(e[2]),
        }

    def _accumulate_metrics(self, *, err: float, manip: float, cond: float, smooth: float, energy: float, lat_ms: float) -> None:
        self._acc['steps'] += 1
        self._acc['err2_sum'] += float(err * err)
        self._acc['manip_sum'] += float(manip)
        self._acc['cond_sum'] += float(min(cond, 1e4))
        self._acc['smooth_sum'] += float(smooth)
        self._acc['energy_sum'] += float(energy)
        self._acc['lat_ms_sum'] += float(lat_ms)
        self._acc['lat_ms_max'] = float(max(self._acc['lat_ms_max'], lat_ms))

    def step(self) -> None:
        t0 = time.perf_counter()
        # math_lab: simplify wafer dynamics and use explicit geometric targets.
        adr = self.wafer_qpos_adr
        yaw = 0.15 * self.t
        q_wafer = _quat_from_rpy(0.0, 0.95, yaw)
        wafer_pos = [-0.08, 0.02, 0.67]
        self.data.qpos[adr:adr+7] = [wafer_pos[0], wafer_pos[1], wafer_pos[2], q_wafer[0], q_wafer[1], q_wafer[2], q_wafer[3]]
        self.data.qvel[self.wafer_qvel_adr:self.wafer_qvel_adr+6] = 0.0

        arm = 'arm_1'
        phase = 1.35 * self.t
        local = self._scenario_target_local(phase)
        tgt_world = np.array(self._wafer_local_to_world(local, q_wafer, wafer_pos), dtype=np.float64)

        dq, meta = self._compute_dls_step(arm, tgt_world)
        q_now = np.array([float(self.data.qpos[a]) for a in self.arm_qpos_adr[arm]], dtype=np.float64)
        q_target = q_now + 0.85 * dq

        # RL modulation from commands published by env/policy (already in ROS2 bridge)
        rl = np.array((self._cmd.get(arm, []) + [0.0] * 6)[:6], dtype=np.float64)
        q_target = q_target + 0.22 * rl
        q_target = np.clip(q_target, -1.9, 1.9)

        for j, aid in enumerate(self.arm_actuator_ids[arm]):
            self.data.ctrl[aid] = float(q_target[j])
        for hidden in self.hidden_arms:
            for aid in self.arm_actuator_ids[hidden]:
                self.data.ctrl[aid] = 0.0

        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt

        # Re-apply wafer pose after stepping to keep fixture stable.
        self.data.qpos[adr:adr+7] = [wafer_pos[0], wafer_pos[1], wafer_pos[2], q_wafer[0], q_wafer[1], q_wafer[2], q_wafer[3]]
        self.data.qvel[self.wafer_qvel_adr:self.wafer_qvel_adr+6] = 0.0

        # Visuals: target trail on wafer + hotspot at contact + spark.
        self._target_trail_world.append(tuple(float(v) for v in tgt_world))
        if len(self._target_trail_world) > 120:
            self._target_trail_world.pop(0)
        tool_site = self.arm_tool_site_ids[arm]
        tool_world = np.array(self.data.site_xpos[tool_site], dtype=np.float64)
        self._tool_trail_world.append(tuple(float(v) for v in tool_world))
        if len(self._tool_trail_world) > 120:
            self._tool_trail_world.pop(0)

        track_idx = int((0.5 + 0.5 * math.sin(phase * 0.4)) * 23)
        self._update_track_visual(track_idx)
        hotspot = [float(tgt_world[0]), float(tgt_world[1]), float(tgt_world[2]) + 0.004]
        spark_alpha = 0.55 + 0.35 * (0.5 + 0.5 * math.sin(28.0 * self.t))
        self._set_process_indicator(hotspot, (1.0, 0.0, 0.0, 0.0), spark_alpha=spark_alpha)

        mujoco.mj_forward(self.model, self.data)

        # Math metrics packed into process fields (contract length unchanged)
        err = float(meta['error_norm'])
        cond = float(meta['cond'])
        manip = float(meta['manipulability'])
        qvel = np.array([float(self.data.qvel[a]) for a in self.arm_qvel_adr[arm]], dtype=np.float64)
        smooth = float(np.mean(np.abs(qvel)))
        energy = float(np.mean(np.square(q_target - q_now)))
        lat_ms = (time.perf_counter() - t0) * 1000.0
        self._accumulate_metrics(err=err, manip=manip, cond=cond, smooth=smooth, energy=energy, lat_ms=lat_ms)

        self._math_last = {
            'target_world': [float(v) for v in tgt_world],
            'tool_world': [float(v) for v in tool_world],
            'error_vec': [meta['e_x'], meta['e_y'], meta['e_z']],
            'error_norm': err,
            'jacobian_cond': cond,
            'manipulability': manip,
            'smoothness': smooth,
            'control_energy': energy,
            'phase': float((phase % (2.0 * math.pi)) / (2.0 * math.pi)),
            'latency_ms': lat_ms,
        }

        # Repurpose process metrics with math-lab semantics but preserve ranges and field count.
        self._proc_state = max(0.0, min(1.0, self._math_last['phase']))
        self._quality = max(0.0, min(1.0, 1.0 - min(1.0, err / 0.12)))
        self._coord = max(0.0, min(1.0, manip / 0.08))
        self._contam = max(0.0, min(1.0, err / 0.12))
        self._vib = max(0.0, min(1.0, smooth / 1.5))
        self._stage = 0.0
        self._progress = self._proc_state
        self._temp = max(0.0, min(1.0, min(cond, 100.0) / 100.0))
        self._pull = max(0.0, min(1.0, energy / 0.2))
        self._rot = max(0.0, min(1.0, self._math_last['phase']))
        self._energy = max(0.0, min(1.0, energy / 0.2))

        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception:
                self.viewer = None

        if self.realtime:
            target_wall = self._last_wall + self.dt
            now = time.perf_counter()
            slp = target_wall - now
            if slp > 0:
                time.sleep(slp)
            self._last_wall = max(target_wall, time.perf_counter())

    def read_wafer_state(self) -> list[float]:
        # math_lab payload layout (length preserved = 18):
        # [wafer_pose(6), p_tool(3), p_target(3), e_pos(3), J_cond_norm, manipulability_norm, phase]
        q = self.data.qpos[self.wafer_qpos_adr : self.wafer_qpos_adr + 7]
        x, y, z = float(q[0]), float(q[1]), float(q[2])
        w, qx, qy, qz = float(q[3]), float(q[4]), float(q[5]), float(q[6])
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * qy - qz * qx))))
        roll = math.atan2(2.0 * (w * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))

        pt = [float(v) for v in self._math_last.get('tool_world', [0.0, 0.0, 0.0])]
        pg = [float(v) for v in self._math_last.get('target_world', [0.0, 0.0, 0.0])]
        e = [float(v) for v in self._math_last.get('error_vec', [0.0, 0.0, 0.0])]
        j_cond = float(self._math_last.get('jacobian_cond', 1.0))
        manip = float(self._math_last.get('manipulability', 0.0))
        phase = float(self._math_last.get('phase', 0.0))

        # Normalize to stable ranges for transport/use in env.
        def npos(v: float) -> float:
            return max(-1.0, min(1.0, v / 1.25))
        def nerr(v: float) -> float:
            return max(-1.0, min(1.0, v / 0.25))
        jn = max(0.0, min(1.0, min(j_cond, 100.0) / 100.0))
        mn = max(0.0, min(1.0, manip / 0.08))

        return [
            x, y, z, roll, pitch, yaw,
            npos(pt[0]), npos(pt[1]), npos(pt[2]),
            npos(pg[0]), npos(pg[1]), npos(pg[2]),
            nerr(e[0]), nerr(e[1]), nerr(e[2]),
            jn, mn, phase,
        ]

    def _final_metrics(self) -> dict[str, Any]:
        n = max(1, int(self._acc['steps']))
        rms = math.sqrt(float(self._acc['err2_sum']) / n)
        out = {
            'mode': 'math_lab',
            'scenario': self.math_scenario,
            'steps': int(self._acc['steps']),
            'tracking_error_rms': rms,
            'manipulability_mean': float(self._acc['manip_sum']) / n,
            'jacobian_cond_mean': float(self._acc['cond_sum']) / n,
            'smoothness_mean': float(self._acc['smooth_sum']) / n,
            'energy_mean': float(self._acc['energy_sum']) / n,
            'latency_ms_mean': float(self._acc['lat_ms_sum']) / n,
            'latency_ms_max': float(self._acc['lat_ms_max']),
            'state_vector_layout': ['q(6)', 'qdot(6)', 'p_tool(3)', 'p_target(3)', 'e_pos(3)', 'J_cond(1)', 'manipulability(1)'],
            'state_vector_dim': 23,
            'state_vector_last': self._build_state_vector_last(),
            'last': dict(self._math_last),
        }
        return out

    def _build_state_vector_last(self) -> list[float]:
        arm = 'arm_1'
        q = [float(self.data.qpos[a]) for a in self.arm_qpos_adr[arm]]
        qd = [float(self.data.qvel[a]) for a in self.arm_qvel_adr[arm]]
        pt = [float(v) for v in self._math_last.get('tool_world', [0.0, 0.0, 0.0])]
        pg = [float(v) for v in self._math_last.get('target_world', [0.0, 0.0, 0.0])]
        e = [float(v) for v in self._math_last.get('error_vec', [0.0, 0.0, 0.0])]
        return q + qd + pt + pg + e + [
            float(self._math_last.get('jacobian_cond', 1.0)),
            float(self._math_last.get('manipulability', 0.0)),
        ]

    def diagnostics_summary(self) -> dict[str, Any]:
        base = super().diagnostics_summary()
        base.update({
            'mode': 'math_lab',
            'scenario': self.math_scenario,
            'state_vector_layout': ['q(6)','qdot(6)','p_tool(3)','p_target(3)','e_pos(3)','J_cond(1)','manipulability(1)'],
            'state_vector_dim': 23,
        })
        return base

    def math_metrics_snapshot(self) -> dict[str, float]:
        return {
            'tracking_error': float(self._math_last.get('error_norm', 0.0)),
            'jacobian_cond': float(self._math_last.get('jacobian_cond', 1.0)),
            'manipulability': float(self._math_last.get('manipulability', 0.0)),
            'smoothness': float(self._math_last.get('smoothness', 0.0)),
            'control_energy': float(self._math_last.get('control_energy', 0.0)),
            'latency_ms': float(self._math_last.get('latency_ms', 0.0)),
            'phase': float(self._math_last.get('phase', 0.0)),
        }

    def shutdown(self) -> None:
        # save JSON metrics for offline validation
        try:
            path = self.metrics_path
            if path is None:
                root = None
                here = Path(__file__).resolve()
                for parent in [here.parent, *here.parents]:
                    if (parent / 'docs').exists() and (parent / 'ros2_ws').exists():
                        root = parent
                        break
                if root is None:
                    root = Path.cwd()
                path = root / 'docs' / 'results_math_lab_backend.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._final_metrics(), indent=2), encoding='utf-8')
        except Exception:
            pass
        super().shutdown()



class _IsaacLabWrapperBackend:
    """Minimal adapter to keep IsaacLab loader compatible with the same ROS2 contract."""

    def __init__(self, loader: Any):
        self.loader = loader
        self._last_cmd = {f"arm_{i}": [0.0] * 6 for i in (1, 2, 3)}

    def reset(self) -> None:
        self.loader.reset()

    def apply_joint_command(self, arm: str, cmd: list[float]) -> None:
        self._last_cmd[arm] = list(cmd)
        # TODO (Isaac Lab real): map arm commands into loader handles.

    def step(self) -> None:
        self.loader.step()

    def read_joint_state(self, arm: str) -> dict[str, list[float]]:
        joints = self.loader.read_joint_states().get(arm, {})
        return {
            "name": joints.get("name", [f"{arm}_joint_{i+1}" for i in range(6)]),
            "position": joints.get("position", [0.0] * 6),
            "velocity": joints.get("velocity", [0.0] * 6),
        }

    def read_wafer_state(self) -> list[float]:
        wafer = self.loader.read_wafer_state()
        pose = wafer.get("pose", {})
        proc = wafer.get("process", {})
        pos = pose.get("position_xyz", [0.0, 0.0, 0.0])
        rpy = pose.get("rpy", [0.0, 0.0, 0.0])
        return [
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(rpy[0]), float(rpy[1]), float(rpy[2]),
            float(proc.get("process_state", 0.0)),
            float(proc.get("quality_score", 0.0)),
            float(proc.get("coordination_score", 0.0)),
            float(proc.get("contamination_risk", 0.0)),
            float(proc.get("vibration_index", 0.0)),
            float(proc.get("stage_norm", 0.0)),
            float(proc.get("progress_norm", 0.0)),
            float(proc.get("temp_norm", 0.0)),
            float(proc.get("pull_rate_norm", 0.0)),
            float(proc.get("rotation_norm", 0.0)),
            float(proc.get("energy_norm", 0.0)),
            float(proc.get("cycle_count_norm", 0.0)),
        ]

    def shutdown(self) -> None:
        if hasattr(self.loader, "shutdown"):
            self.loader.shutdown()


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


class WaferCellSimNode(Node):
    def __init__(self):
        super().__init__("wafer_cell_sim")
        self.declare_parameter("backend", "stub")  # stub | mujoco | isaaclab
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("headless", True)
        self.declare_parameter("mujoco_realtime", True)
        self.declare_parameter("mujoco_mode", "industrial")  # industrial | math_lab
        self.declare_parameter("mujoco_math_scenario", "spiral")  # line|circle|spiral|figure8|rotate_extend
        self.declare_parameter("mujoco_metrics_path", "")
        self.declare_parameter("debug_log_every_n", 0)  # 0 disables periodic command/state logs
        self.declare_parameter("publish_math_metrics_topic", True)

        self.backend_name = str(self.get_parameter("backend").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.headless = bool(self.get_parameter("headless").value)
        self.mujoco_realtime = bool(self.get_parameter("mujoco_realtime").value)
        self.mujoco_mode = str(self.get_parameter("mujoco_mode").value)
        self.mujoco_math_scenario = str(self.get_parameter("mujoco_math_scenario").value)
        self.mujoco_metrics_path = str(self.get_parameter("mujoco_metrics_path").value)
        self.debug_log_every_n = int(self.get_parameter("debug_log_every_n").value)
        self.publish_math_metrics_topic = bool(self.get_parameter("publish_math_metrics_topic").value)
        self._tick_count = 0
        self._last_cmd_arm1 = [0.0] * 6

        self.pub_arm1 = self.create_publisher(JointState, TOPICS["arm_1_joint_states"], 10)
        self.pub_arm2 = self.create_publisher(JointState, TOPICS["arm_2_joint_states"], 10)
        self.pub_arm3 = self.create_publisher(JointState, TOPICS["arm_3_joint_states"], 10)
        self.pub_wafer = self.create_publisher(Float32MultiArray, TOPICS["wafer_state"], 10)
        self.pub_math_metrics = self.create_publisher(Float32MultiArray, "/cell/math_metrics", 10)
        self.create_subscription(JointState, TOPICS["arm_1_joint_command"], self._on_cmd_arm1, 10)
        self.create_subscription(JointState, TOPICS["arm_2_joint_command"], self._on_cmd_arm2, 10)
        self.create_subscription(JointState, TOPICS["arm_3_joint_command"], self._on_cmd_arm3, 10)
        self.reset_srv = self.create_service(Trigger, TOPICS["reset_service"], self._on_reset)

        self._backend: Any = self._build_backend()
        try:
            if hasattr(self._backend, "diagnostics_summary"):
                ds = self._backend.diagnostics_summary()
                self.get_logger().info(
                    "MuJoCo diagnostics: nu=%s nq=%s nv=%s timestep=%.6f mode=%s"
                    % (ds.get("nu"), ds.get("nq"), ds.get("nv"), float(ds.get("timestep", 0.0)), ds.get("mode"))
                )
        except Exception as exc:
            self.get_logger().warn(f"Diagnostics summary unavailable: {exc}")
        self._timer = self.create_timer(1.0 / max(1e-3, self.publish_rate_hz), self._on_timer)
        extra = ""
        if self.backend_name == "mujoco":
            extra = f", mujoco_mode={self.mujoco_mode}, scenario={self.mujoco_math_scenario}"
        self.get_logger().info(
            f"Wafer Cell Simulation Node started (backend={self.backend_name}{extra}, rate={self.publish_rate_hz:.1f}Hz, protocol={PROTOCOL_VERSION})."
        )

    def _build_backend(self) -> Any:
        if self.backend_name == "mujoco":
            try:
                if self.mujoco_mode == "math_lab":
                    backend = _MujocoMathLabBackend(
                        dt=1.0 / max(1e-3, self.publish_rate_hz),
                        headless=self.headless,
                        realtime=self.mujoco_realtime,
                        scenario=self.mujoco_math_scenario,
                        metrics_path=(self.mujoco_metrics_path or None),
                    )
                    self.get_logger().info(f"MuJoCo backend initialized (math_lab, scenario={self.mujoco_math_scenario}).")
                else:
                    backend = _MujocoWaferCellBackend(
                        dt=1.0 / max(1e-3, self.publish_rate_hz),
                        headless=self.headless,
                        realtime=self.mujoco_realtime,
                    )
                    self.get_logger().info("MuJoCo backend initialized (3-arm wafer cell simplified scene).")
                return backend
            except Exception as exc:
                self.get_logger().error(f"MuJoCo backend init failed: {exc}. Falling back to stub.")
                self.backend_name = "stub"
                return _InternalStubBackend()

        if self.backend_name == "isaaclab":
            if IsaacLabSceneLoader is None:
                self.get_logger().warning("IsaacLabSceneLoader import failed; falling back to stub backend.")
                self.backend_name = "stub"
                return _InternalStubBackend()
            try:
                loader = IsaacLabSceneLoader(WaferCellSceneConfig())  # type: ignore[misc]
                loader.initialize(headless=self.headless)
                self.get_logger().info("Isaac Lab scene loader initialized.")
                return _IsaacLabWrapperBackend(loader)
            except Exception as exc:
                self.get_logger().error(f"Isaac Lab backend init failed: {exc}. Falling back to stub.")
                self.backend_name = "stub"
                return _InternalStubBackend()

        return _InternalStubBackend()

    def _on_cmd_arm1(self, msg: JointState) -> None:
        self._last_cmd_arm1 = list(msg.position)
        self._backend.apply_joint_command("arm_1", list(msg.position))

    def _on_cmd_arm2(self, msg: JointState) -> None:
        self._backend.apply_joint_command("arm_2", list(msg.position))

    def _on_cmd_arm3(self, msg: JointState) -> None:
        self._backend.apply_joint_command("arm_3", list(msg.position))

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        try:
            self._backend.reset()
            self._publish_current_state()  # immediate state after reset for RL client
            response.success = True
            response.message = "wafer cell reset ok"
        except Exception as exc:
            response.success = False
            response.message = f"reset failed: {exc}"
        return response

    def _publish_joint_state(self, pub, state: dict[str, list[float]]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = state.get("name", [])
        msg.position = state.get("position", [])
        msg.velocity = state.get("velocity", [])
        msg.effort = []
        pub.publish(msg)

    def _on_timer(self) -> None:
        self._backend.step()
        self._tick_count += 1
        self._publish_current_state()

    def _publish_current_state(self) -> None:
        s1 = self._backend.read_joint_state("arm_1")
        s2 = self._backend.read_joint_state("arm_2")
        s3 = self._backend.read_joint_state("arm_3")
        wafer_data = self._backend.read_wafer_state()
        self._publish_joint_state(self.pub_arm1, s1)
        self._publish_joint_state(self.pub_arm2, s2)
        self._publish_joint_state(self.pub_arm3, s3)
        msg = Float32MultiArray()
        msg.data = [float(v) for v in wafer_data]
        self.pub_wafer.publish(msg)

        if self.publish_math_metrics_topic and hasattr(self._backend, "math_metrics_snapshot"):
            try:
                mm = self._backend.math_metrics_snapshot()
                mmsg = Float32MultiArray()
                mmsg.data = [
                    float(mm.get("tracking_error", 0.0)),
                    float(mm.get("jacobian_cond", 0.0)),
                    float(mm.get("manipulability", 0.0)),
                    float(mm.get("smoothness", 0.0)),
                    float(mm.get("control_energy", 0.0)),
                    float(mm.get("latency_ms", 0.0)),
                    float(mm.get("phase", 0.0)),
                ]
                self.pub_math_metrics.publish(mmsg)
            except Exception:
                pass

        if self.debug_log_every_n > 0 and (self._tick_count % self.debug_log_every_n == 0):
            try:
                q = s1.get("position", [])
                c = self._last_cmd_arm1
                self.get_logger().info(
                    "tick=%d arm1 cmd[0:3]=[%.3f,%.3f,%.3f] q[0:3]=[%.3f,%.3f,%.3f]" % (
                        self._tick_count,
                        float(c[0]) if len(c) > 0 else 0.0,
                        float(c[1]) if len(c) > 1 else 0.0,
                        float(c[2]) if len(c) > 2 else 0.0,
                        float(q[0]) if len(q) > 0 else 0.0,
                        float(q[1]) if len(q) > 1 else 0.0,
                        float(q[2]) if len(q) > 2 else 0.0,
                    )
                )
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = WaferCellSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        backend = getattr(node, "_backend", None)
        if backend is not None and hasattr(backend, "shutdown"):
            try:
                backend.shutdown()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
