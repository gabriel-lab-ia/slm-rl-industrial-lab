"""Demo skeleton: SLM RL controlling the Isaac Lab wafer-cell via ROS2 bridge."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from slm_wafer_cell.envs.isaaclab_wafer_cell_env import IsaacLabWaferCellEnv, IsaacLabWaferCellEnvConfig
from slm_wafer_cell.policy.slm_policy_loader import SLMPolicy
from slm_wafer_cell.ros2.protocol import PROTOCOL_VERSION, WAFER_STATE_FIELDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SLM RL demo against IsaacLabWaferCellEnv (ROS2 bridge).")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps-per-episode", type=int, default=300)
    p.add_argument("--obs-dim", type=int, default=64)
    p.add_argument("--act-dim", type=int, default=18)
    p.add_argument("--control-rate-hz", type=float, default=50.0)
    p.add_argument("--out", type=Path, default=Path("docs/results_stub_backend.json"))
    return p.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int((len(values) - 1) * p)
    return sorted(values)[idx]


def main() -> None:
    args = parse_args()
    env = IsaacLabWaferCellEnv(
        IsaacLabWaferCellEnvConfig(
            obs_dim=args.obs_dim,
            act_dim=args.act_dim,
            control_rate_hz=args.control_rate_hz,
        )
    )
    policy = SLMPolicy(args.checkpoint, device=args.device, obs_dim=args.obs_dim, act_dim=args.act_dim)

    ep_rewards: list[float] = []
    infer_lat_ms: list[float] = []
    control_loop_lat_ms: list[float] = []
    proc_q: list[float] = []
    coord_q: list[float] = []
    defect_risk_vals: list[float] = []
    throughput_proxy: list[float] = []
    process_state_vals: list[float] = []
    quality_vals: list[float] = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total_reward = 0.0
            ep_proc = []
            ep_coord = []
            ep_defect = []
            ep_proc_state = []
            ep_quality = []

            for t in range(args.steps_per_episode):
                loop_t0 = time.perf_counter()
                action = policy.act(obs)
                infer_lat_ms.append(policy.last_infer_ms)
                obs, reward, done, truncated, info = env.step(action)
                loop_t1 = time.perf_counter()

                total_reward += float(reward)
                ep_proc.append(float(info.get("proc_quality", 0.0)))
                ep_coord.append(float(info.get("coord_quality_fused", info.get("coord_quality", 0.0))))
                ep_defect.append(float(info.get("defect_risk", 0.0)))
                ep_proc_state.append(float(info.get("wafer_process_state", 0.0)))
                ep_quality.append(float(info.get("wafer_quality_score", 0.0)))
                control_loop_lat_ms.append((loop_t1 - loop_t0) * 1000.0)

                # TODO (backend Isaac Lab real): substituir por contador explícito de ciclos concluídos.
                throughput_proxy.append(float(info.get("progress_norm", 0.0)))

                if done or truncated:
                    break

            ep_rewards.append(total_reward)
            proc_q.extend(ep_proc)
            coord_q.extend(ep_coord)
            defect_risk_vals.extend(ep_defect)
            process_state_vals.extend(ep_proc_state)
            quality_vals.extend(ep_quality)
            print(
                f"episode={ep+1:02d} reward={total_reward:.4f} "
                f"process_state_mean={statistics.mean(ep_proc_state) if ep_proc_state else 0.0:.4f} "
                f"quality_mean={statistics.mean(ep_quality) if ep_quality else 0.0:.4f} "
                f"coord_q_mean={statistics.mean(ep_coord) if ep_coord else 0.0:.4f} "
                f"defect_risk_mean={statistics.mean(ep_defect) if ep_defect else 0.0:.4f}"
            )
    finally:
        env.close()

    report = {
        "checkpoint": str(args.checkpoint),
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps_per_episode),
        "metrics": {
            "reward_mean": statistics.mean(ep_rewards) if ep_rewards else 0.0,
            "reward_p50": statistics.median(ep_rewards) if ep_rewards else 0.0,
            "proc_quality_mean": statistics.mean(proc_q) if proc_q else 0.0,  # compat com backend futuro
            "wafer_process_state_mean": statistics.mean(process_state_vals) if process_state_vals else 0.0,
            "wafer_quality_score_mean": statistics.mean(quality_vals) if quality_vals else 0.0,
            "coord_quality_mean": statistics.mean(coord_q) if coord_q else 0.0,
            # TODO (backend Isaac Lab real): converter para taxa de defeito real por ciclo.
            "defect_rate_proxy": statistics.mean(defect_risk_vals) if defect_risk_vals else 0.0,
            # TODO (backend Isaac Lab real): throughput real via contadores de ciclos concluídos.
            "throughput_cycles_per_env_proxy": statistics.mean(throughput_proxy) if throughput_proxy else 0.0,
        },
        "latency_ms": {
            "infer_p50": statistics.median(infer_lat_ms) if infer_lat_ms else 0.0,
            "infer_p95": percentile(infer_lat_ms, 0.95),
            "infer_mean": statistics.mean(infer_lat_ms) if infer_lat_ms else 0.0,
            "control_loop_p50": statistics.median(control_loop_lat_ms) if control_loop_lat_ms else 0.0,
            "control_loop_p95": percentile(control_loop_lat_ms, 0.95),
            "control_loop_mean": statistics.mean(control_loop_lat_ms) if control_loop_lat_ms else 0.0,
        },
        "policy": policy.summary(),
        "notes": [
            "Stub backend ROS2 ativo (nao Isaac Lab real nesta etapa).",
            "TODO: substituir SLMPolicy fallback MLP pelo SLM RL real (MathCoreAgent + adapter de acao).",
            "TODO: substituir throughput/defect proxies por contadores reais vindos do backend Isaac Lab.",
        ],
        "ros2_contract": {
            "protocol_version": PROTOCOL_VERSION,
            "wafer_state_fields": WAFER_STATE_FIELDS,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        f"summary reward_mean={report['metrics']['reward_mean']:.4f} "
        f"infer_mean_ms={report['latency_ms']['infer_mean']:.4f} "
        f"infer_p50_ms={report['latency_ms']['infer_p50']:.4f} "
        f"control_loop_mean_ms={report['latency_ms']['control_loop_mean']:.4f}"
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
