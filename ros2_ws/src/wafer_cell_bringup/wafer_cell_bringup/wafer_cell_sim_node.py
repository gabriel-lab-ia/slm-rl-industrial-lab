#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Any

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


class _InternalStubBackend:
    """Deterministic, smooth, ROS2-friendly wafer-cell stub backend.

    Simula 3 braços mecatrônicos + estado de wafer/processo com dinâmica fake porém coerente.
    Contrato de tópicos permanece estável para substituição futura por Isaac Lab real.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self.dt = 0.02
        self.joint_count = 6
        self._cmd = {
            "arm_1": [0.0] * self.joint_count,
            "arm_2": [0.0] * self.joint_count,
            "arm_3": [0.0] * self.joint_count,
        }
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
        self._wafer_pose = [0.0, 0.0, 0.80, 0.0, 0.0, 0.0]  # x,y,z,roll,pitch,yaw
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
        self._cmd[arm] = list(cmd[: self.joint_count]) + [0.0] * max(0, self.joint_count - len(cmd))
        self._cmd[arm] = self._cmd[arm][: self.joint_count]

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

        # Suaviza progresso de processo em função de coordenação e esforço.
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

        # Wafer pose (fake, suave): x/y pequeno, z quase fixo, yaw seguindo rotação do processo.
        self._wafer_pose[0] = 0.015 * math.sin(0.4 * self.t)
        self._wafer_pose[1] = 0.012 * math.cos(0.35 * self.t)
        self._wafer_pose[2] = 0.80 + 0.005 * math.sin(0.2 * self.t)
        self._wafer_pose[3] = 0.01 * math.sin(0.5 * self.t)
        self._wafer_pose[4] = 0.01 * math.cos(0.45 * self.t)
        self._wafer_pose[5] = 0.25 * self._stage + 0.10 * self._progress + 0.03 * math.sin(0.3 * self.t)

        # Processo fake coerente (normalizado 0..1)
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
        # Float32MultiArray convention for N2 stub v1:
        # [0:6]  wafer_pose = [x,y,z,roll,pitch,yaw]
        # [6]    wafer_process_state (0..1)
        # [7]    wafer_quality_score (0..1)
        # [8]    coordination_score
        # [9]    contamination_risk
        # [10]   vibration_index
        # [11]   stage_norm
        # [12]   progress_norm
        # [13]   temperature_norm
        # [14]   pull_rate_norm
        # [15]   rotation_norm
        # [16]   energy_norm
        # [17]   cycle_count_norm_proxy
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


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class WaferCellSimNode(Node):
    def __init__(self):
        super().__init__("wafer_cell_sim")
        # Parameters
        self.declare_parameter("backend", "stub")  # isaaclab | stub
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("headless", True)
        self.backend_name = str(self.get_parameter("backend").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.headless = bool(self.get_parameter("headless").value)

        # ROS interfaces: joint states / commands for 3 arms + wafer state.
        self.pub_arm1 = self.create_publisher(JointState, TOPICS["arm_1_joint_states"], 10)
        self.pub_arm2 = self.create_publisher(JointState, TOPICS["arm_2_joint_states"], 10)
        self.pub_arm3 = self.create_publisher(JointState, TOPICS["arm_3_joint_states"], 10)
        self.pub_wafer = self.create_publisher(Float32MultiArray, TOPICS["wafer_state"], 10)
        self.sub_cmd1 = self.create_subscription(JointState, TOPICS["arm_1_joint_command"], self._on_cmd_arm1, 10)
        self.sub_cmd2 = self.create_subscription(JointState, TOPICS["arm_2_joint_command"], self._on_cmd_arm2, 10)
        self.sub_cmd3 = self.create_subscription(JointState, TOPICS["arm_3_joint_command"], self._on_cmd_arm3, 10)
        self.reset_srv = self.create_service(Trigger, TOPICS["reset_service"], self._on_reset)

        self._backend: Any = self._build_backend()
        self._timer = self.create_timer(1.0 / max(1e-3, self.publish_rate_hz), self._on_timer)
        self.get_logger().info(
            f"Wafer Cell Simulation Node started (backend={self.backend_name}, rate={self.publish_rate_hz:.1f}Hz, protocol={PROTOCOL_VERSION})."
        )

    def _build_backend(self) -> Any:
        if self.backend_name == "isaaclab":
            if IsaacLabSceneLoader is None:
                self.get_logger().warning("IsaacLabSceneLoader import failed; falling back to stub backend.")
                self.backend_name = "stub"
                return _InternalStubBackend()
            try:
                loader = IsaacLabSceneLoader(WaferCellSceneConfig())  # type: ignore[misc]
                loader.initialize(headless=self.headless)
                self.get_logger().info("Isaac Lab scene loader initialized.")
                # TODO: wrap loader into a richer backend object with process metrics.
                return loader
            except Exception as exc:
                self.get_logger().error(f"Isaac Lab backend init failed: {exc}. Falling back to stub.")
                self.backend_name = "stub"
                return _InternalStubBackend()
        return _InternalStubBackend()

    def _on_cmd_arm1(self, msg: JointState) -> None:
        self._apply_arm_cmd("arm_1", msg)

    def _on_cmd_arm2(self, msg: JointState) -> None:
        self._apply_arm_cmd("arm_2", msg)

    def _on_cmd_arm3(self, msg: JointState) -> None:
        self._apply_arm_cmd("arm_3", msg)

    def _apply_arm_cmd(self, arm: str, msg: JointState) -> None:
        cmd = list(msg.position)
        if self.backend_name == "stub":
            self._backend.apply_joint_command(arm, cmd)
            return
        # TODO (isaaclab): collect commands for all 3 arms and call loader.apply_joint_commands(...)
        # Example target map structure:
        # {"arm_1": [...], "arm_2": [...], "arm_3": [...]}

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        try:
            if self.backend_name == "stub":
                self._backend.reset()
            else:
                self._backend.reset()  # TODO: IsaacLabSceneLoader.reset should restore scene
            # Publica imediatamente o primeiro estado após reset para reduzir latência do cliente RL.
            self._publish_current_state()
            response.success = True
            response.message = "wafer cell reset ok"
        except Exception as exc:
            response.success = False
            response.message = f"reset failed: {exc}"
        return response

    def _publish_joint_state(self, pub, arm: str, state: dict[str, list[float]]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = state.get("name", [])
        msg.position = state.get("position", [])
        msg.velocity = state.get("velocity", [])
        msg.effort = []
        pub.publish(msg)

    def _on_timer(self) -> None:
        # Step simulation backend.
        if self.backend_name == "stub":
            self._backend.step()
            self._publish_current_state()
        else:
            # TODO (isaaclab): replace with actual loader methods and process metric extraction.
            self._backend.step()
            self._publish_current_state()

    def _publish_current_state(self) -> None:
        if self.backend_name == "stub":
            s1 = self._backend.read_joint_state("arm_1")
            s2 = self._backend.read_joint_state("arm_2")
            s3 = self._backend.read_joint_state("arm_3")
            wafer_data = self._backend.read_wafer_state()
        else:
            joints = self._backend.read_joint_states()
            s1 = joints.get("arm_1", {"name": [], "position": [], "velocity": []})
            s2 = joints.get("arm_2", {"name": [], "position": [], "velocity": []})
            s3 = joints.get("arm_3", {"name": [], "position": [], "velocity": []})
            wafer = self._backend.read_wafer_state()
            pose = wafer.get("pose", {})
            proc = wafer.get("process", {})
            # TODO(isaaclab): publicar contrato final equivalente ao stub v1.
            wafer_data = [
                *pose.get("position_xyz", [0.0, 0.0, 0.0]),
                0.0,
                0.0,
                0.0,
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
        self._publish_joint_state(self.pub_arm1, "arm_1", s1)
        self._publish_joint_state(self.pub_arm2, "arm_2", s2)
        self._publish_joint_state(self.pub_arm3, "arm_3", s3)
        msg = Float32MultiArray()
        msg.data = [float(v) for v in wafer_data]
        self.pub_wafer.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaferCellSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Best-effort backend shutdown if available.
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
