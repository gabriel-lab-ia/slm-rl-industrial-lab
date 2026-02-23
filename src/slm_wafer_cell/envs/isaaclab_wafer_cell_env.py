"""Gym-like Isaac Lab wafer-cell environment bridged over ROS2 Jazzy."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from slm_wafer_cell.ros2.protocol import PROTOCOL_VERSION, TOPICS, WAFER_STATE_INDEX

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - import depends on ROS2 env
    rclpy = None  # type: ignore
    SingleThreadedExecutor = None  # type: ignore
    Node = object  # type: ignore
    JointState = object  # type: ignore
    Float32MultiArray = object  # type: ignore
    Trigger = object  # type: ignore


@dataclass
class IsaacLabWaferCellEnvConfig:
    obs_dim: int = 64
    act_dim: int = 18  # 3 arms x 6 joints (adjust to your robot)
    control_rate_hz: float = 50.0
    reset_timeout_s: float = 3.0
    state_timeout_s: float = 1.0
    action_clip: float = 1.0
    max_episode_steps: int = 500
    namespace: str = ""
    use_ros_thread: bool = True


class _WaferCellRosClientNode(Node):  # type: ignore[misc]
    """ROS2 client node used by the Gym-like env."""

    def __init__(self, cfg: IsaacLabWaferCellEnvConfig) -> None:
        super().__init__("wafer_cell_rl_client")
        self.cfg = cfg
        ns = cfg.namespace.rstrip("/")

        def topic(name: str) -> str:
            return f"{ns}{name}" if ns else name

        # Subscribers (state from sim)
        self.sub_arm1 = self.create_subscription(JointState, topic(TOPICS["arm_1_joint_states"]), self._on_arm1, 10)
        self.sub_arm2 = self.create_subscription(JointState, topic(TOPICS["arm_2_joint_states"]), self._on_arm2, 10)
        self.sub_arm3 = self.create_subscription(JointState, topic(TOPICS["arm_3_joint_states"]), self._on_arm3, 10)
        self.sub_wafer = self.create_subscription(Float32MultiArray, topic(TOPICS["wafer_state"]), self._on_wafer, 10)

        # Publishers (commands to sim)
        self.pub_cmd1 = self.create_publisher(JointState, topic(TOPICS["arm_1_joint_command"]), 10)
        self.pub_cmd2 = self.create_publisher(JointState, topic(TOPICS["arm_2_joint_command"]), 10)
        self.pub_cmd3 = self.create_publisher(JointState, topic(TOPICS["arm_3_joint_command"]), 10)

        # Reset service (optional but recommended)
        self.reset_cli = self.create_client(Trigger, topic(TOPICS["reset_service"]))

        self._lock = threading.Lock()
        self._state_cv = threading.Condition(self._lock)
        self._arm_state: dict[str, dict[str, np.ndarray]] = {}
        self._wafer_state: np.ndarray | None = None
        self._last_state_time = 0.0
        self._last_info: dict[str, float] = {}
        self._state_seq = 0
        self._last_arm_positions = {
            "arm_1": np.zeros(0, dtype=np.float32),
            "arm_2": np.zeros(0, dtype=np.float32),
            "arm_3": np.zeros(0, dtype=np.float32),
        }

    def _on_arm1(self, msg: JointState) -> None:
        self._on_arm("arm_1", msg)

    def _on_arm2(self, msg: JointState) -> None:
        self._on_arm("arm_2", msg)

    def _on_arm3(self, msg: JointState) -> None:
        self._on_arm("arm_3", msg)

    def _on_arm(self, key: str, msg: JointState) -> None:
        with self._state_cv:
            pos = np.asarray(list(msg.position), dtype=np.float32)
            vel = np.asarray(list(msg.velocity), dtype=np.float32)
            self._arm_state[key] = {
                "position": pos,
                "velocity": vel,
            }
            self._last_arm_positions[key] = pos
            self._last_state_time = time.time()
            self._state_seq += 1
            self._state_cv.notify_all()

    def _on_wafer(self, msg: Float32MultiArray) -> None:
        with self._state_cv:
            arr = np.asarray(list(msg.data), dtype=np.float32)
            self._wafer_state = arr
            idx = WAFER_STATE_INDEX
            self._last_info = {
                "wafer_x": float(arr[idx["wafer_x"]]) if arr.size > idx["wafer_x"] else 0.0,
                "wafer_y": float(arr[idx["wafer_y"]]) if arr.size > idx["wafer_y"] else 0.0,
                "wafer_z": float(arr[idx["wafer_z"]]) if arr.size > idx["wafer_z"] else 0.0,
                "wafer_roll": float(arr[idx["wafer_roll"]]) if arr.size > idx["wafer_roll"] else 0.0,
                "wafer_pitch": float(arr[idx["wafer_pitch"]]) if arr.size > idx["wafer_pitch"] else 0.0,
                "wafer_yaw": float(arr[idx["wafer_yaw"]]) if arr.size > idx["wafer_yaw"] else 0.0,
                "wafer_process_state": float(arr[idx["wafer_process_state"]]) if arr.size > idx["wafer_process_state"] else 0.0,
                "wafer_quality_score": float(arr[idx["wafer_quality_score"]]) if arr.size > idx["wafer_quality_score"] else 0.0,
                "coord_quality": float(arr[idx["coordination_score"]]) if arr.size > idx["coordination_score"] else 0.0,
                "defect_risk": float(arr[idx["contamination_risk"]]) if arr.size > idx["contamination_risk"] else 0.0,
                "vibration_index": float(arr[idx["vibration_index"]]) if arr.size > idx["vibration_index"] else 0.0,
                "stage_norm": float(arr[idx["stage_norm"]]) if arr.size > idx["stage_norm"] else 0.0,
                "progress_norm": float(arr[idx["progress_norm"]]) if arr.size > idx["progress_norm"] else 0.0,
                "temp_norm": float(arr[idx["temperature_norm"]]) if arr.size > idx["temperature_norm"] else 0.0,
                "pull_rate_norm": float(arr[idx["pull_rate_norm"]]) if arr.size > idx["pull_rate_norm"] else 0.0,
                "rotation_norm": float(arr[idx["rotation_norm"]]) if arr.size > idx["rotation_norm"] else 0.0,
                "energy_norm": float(arr[idx["energy_norm"]]) if arr.size > idx["energy_norm"] else 0.0,
                "cycle_count_norm_proxy": float(arr[idx["cycle_count_norm_proxy"]]) if arr.size > idx["cycle_count_norm_proxy"] else 0.0,
            }
            self._last_state_time = time.time()
            self._state_seq += 1
            self._state_cv.notify_all()

    def publish_joint_command(self, action: np.ndarray) -> None:
        """Publish split joint commands to the 3 arms.

        Action layout (default): [arm1_joints..., arm2_joints..., arm3_joints...]
        TODO: adapt `joints_per_arm` to the actual arm model in Isaac Lab scene.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        joints_per_arm = action.size // 3
        splits = [action[0:joints_per_arm], action[joints_per_arm:2 * joints_per_arm], action[2 * joints_per_arm:]]
        pubs = [self.pub_cmd1, self.pub_cmd2, self.pub_cmd3]
        for i, (pub, vec) in enumerate(zip(pubs, splits, strict=False), start=1):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [f"arm_{i}_joint_{j+1}" for j in range(vec.size)]
            msg.position = [float(v) for v in vec]
            # TODO: optionally use velocity/effort commands depending on controller mode.
            msg.velocity = []
            msg.effort = []
            pub.publish(msg)

    def wait_for_fresh_state(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        with self._state_cv:
            start_t = self._last_state_time
            start_seq = self._state_seq
            while time.time() < deadline:
                have_arms = all(k in self._arm_state for k in ("arm_1", "arm_2", "arm_3"))
                have_wafer = self._wafer_state is not None
                if have_arms and have_wafer and (self._last_state_time > start_t or self._state_seq > start_seq):
                    return True
                remain = max(0.0, deadline - time.time())
                self._state_cv.wait(timeout=min(0.05, remain))
        return False

    def compose_obs(self, obs_dim: int) -> np.ndarray:
        with self._lock:
            chunks: list[np.ndarray] = []
            for key in ("arm_1", "arm_2", "arm_3"):
                st = self._arm_state.get(key, {})
                pos = st.get("position", np.zeros(0, dtype=np.float32))
                vel = st.get("velocity", np.zeros_like(pos))
                chunks.append(pos.astype(np.float32))
                chunks.append(vel.astype(np.float32))
            chunks.append((self._wafer_state if self._wafer_state is not None else np.zeros(12, dtype=np.float32)).astype(np.float32))
        flat = np.concatenate(chunks, dtype=np.float32) if chunks else np.zeros(obs_dim, dtype=np.float32)
        if flat.size < obs_dim:
            out = np.zeros(obs_dim, dtype=np.float32)
            out[: flat.size] = flat
            return out
        return flat[:obs_dim]

    def latest_info(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_info)

    def joint_coordination_score(self) -> float:
        with self._lock:
            positions = []
            for key in ("arm_1", "arm_2", "arm_3"):
                pos = self._arm_state.get(key, {}).get("position", np.zeros(0, dtype=np.float32))
                if pos.size:
                    positions.append(pos)
            if len(positions) < 3:
                return 0.0
            # Score baseado na dispersão média das posições por junta.
            min_len = min(p.size for p in positions)
            mat = np.stack([p[:min_len] for p in positions], axis=0)
            spread = float(np.mean(np.var(mat, axis=0)))
            return float(max(0.0, min(1.0, 1.0 - 1.8 * spread)))


class IsaacLabWaferCellEnv:
    """Gym-like environment that communicates with the ROS2 simulation node."""

    def __init__(self, cfg: IsaacLabWaferCellEnvConfig | None = None) -> None:
        if rclpy is None:
            raise RuntimeError("ROS2 Python (rclpy) is not available in this environment.")
        self.cfg = cfg or IsaacLabWaferCellEnvConfig()
        self._owns_rclpy = False
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._node: _WaferCellRosClientNode | None = None
        self._episode_step = 0
        self._last_action = np.zeros(self.cfg.act_dim, dtype=np.float32)
        self._connect_ros()

    def _connect_ros(self) -> None:
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._node = _WaferCellRosClientNode(self.cfg)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        if self.cfg.use_ros_thread:
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

    def _spin_once(self, timeout_s: float = 0.01) -> None:
        if self._executor is None:
            return
        self._executor.spin_once(timeout_sec=timeout_s)

    def _ensure_state(self, timeout_s: float | None = None) -> None:
        timeout = self.cfg.state_timeout_s if timeout_s is None else timeout_s
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.cfg.use_ros_thread:
                self._spin_once(timeout_s=0.01)
            assert self._node is not None
            if self._node.wait_for_fresh_state(timeout_s=0.05):
                return
        raise TimeoutError("Timed out waiting for wafer-cell ROS2 state topics.")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._episode_step = 0
        self._last_action[:] = 0.0
        assert self._node is not None

        # TODO: add structured reset options (stage, randomized wafer pose, disturbance profile).
        if not self._node.reset_cli.wait_for_service(timeout_sec=self.cfg.reset_timeout_s):
            self._node.get_logger().warn("/cell/reset service unavailable; proceeding without explicit reset.")
        elif self._node.reset_cli.service_is_ready():
            req = Trigger.Request()
            fut = self._node.reset_cli.call_async(req)
            deadline = time.time() + self.cfg.reset_timeout_s
            while time.time() < deadline and not fut.done():
                if not self.cfg.use_ros_thread:
                    self._spin_once(timeout_s=0.01)
                time.sleep(0.01)
        self._ensure_state(timeout_s=self.cfg.reset_timeout_s)
        obs = self._node.compose_obs(self.cfg.obs_dim)
        info = self._node.latest_info()
        info["joint_coordination_stub"] = self._node.joint_coordination_score()
        info["reset_ok"] = 1.0
        info["protocol_version"] = PROTOCOL_VERSION
        return obs, info

    def step(self, action: np.ndarray):
        assert self._node is not None
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.cfg.act_dim:
            raise ValueError(f"Expected action shape ({self.cfg.act_dim},), got {a.shape}")
        a = np.clip(a, -self.cfg.action_clip, self.cfg.action_clip)

        self._node.publish_joint_command(a)
        self._ensure_state(timeout_s=self.cfg.state_timeout_s)
        obs = self._node.compose_obs(self.cfg.obs_dim)
        info = self._node.latest_info()

        # Reward stub coerente com o backend futuro: processo + coordenação + suavidade + energia.
        wafer_process_state = float(info.get("wafer_process_state", 0.0))
        wafer_quality_score = float(info.get("wafer_quality_score", 0.0))
        coord_q_ros = float(info.get("coord_quality", 0.0))
        coord_q_joint = float(self._node.joint_coordination_score())
        coord_q = 0.5 * coord_q_ros + 0.5 * coord_q_joint
        defect_risk = float(info.get("defect_risk", 0.0))
        vibration_index = float(info.get("vibration_index", 0.0))
        stage_norm = float(info.get("stage_norm", 0.0))
        progress_norm = float(info.get("progress_norm", 0.0))
        energy_norm = float(info.get("energy_norm", 0.0))

        smooth_pen = float(np.mean((a - self._last_action) ** 2))
        energy_pen = float(0.02 * np.mean(a ** 2))
        reward = (
            1.4 * wafer_process_state
            + 1.2 * wafer_quality_score
            + 1.0 * coord_q
            + 0.3 * stage_norm
            + 0.4 * progress_norm
            - 1.4 * defect_risk
            - 0.3 * vibration_index
            - 0.2 * energy_norm
            - 0.4 * smooth_pen
            - energy_pen
        )
        self._last_action = a
        self._episode_step += 1

        # Stub policy: done quando o processo completa um ciclo (proxy) ou truncado por steps.
        done = bool(stage_norm >= 0.95 and progress_norm >= 0.95)
        truncated = bool(self._episode_step >= self.cfg.max_episode_steps)
        info.update(
            {
                "proc_quality": wafer_quality_score,  # compatibilidade com métricas antigas
                "wafer_process_state": wafer_process_state,
                "wafer_quality_score": wafer_quality_score,
                "coord_quality_joint": coord_q_joint,
                "coord_quality_fused": coord_q,
                "throughput_cycles_per_env_proxy": float(info.get("cycle_count_norm_proxy", 0.0)),
                "reward_terms/process_state": wafer_process_state,
                "reward_terms/quality": wafer_quality_score,
                "reward_terms/coord_q": coord_q,
                "reward_terms/defect_risk": defect_risk,
                "reward_terms/vibration": vibration_index,
                "reward_terms/energy_norm": energy_norm,
                "reward_terms/smooth_pen": smooth_pen,
                "reward_terms/energy_pen": energy_pen,
                "episode_step": self._episode_step,
            }
        )
        return obs, reward, done, truncated, info

    def close(self) -> None:
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
