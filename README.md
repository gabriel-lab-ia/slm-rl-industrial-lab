# slm-rl-isaaclab-wafer-cell-3arms (N2)

Projeto N2 para integrar um SLM RL de baixa latência com uma célula industrial de wafer (3 braços mecatrônicos) via ROS2 Jazzy e, na próxima etapa, Isaac Lab.

## Estado atual (publicável hoje)
- ROS2 Jazzy bridge funcional (`wafer_cell_bringup`)
- Backend `stub` profissional (determinístico, com estados de juntas + estado de wafer/processo)
- `IsaacLabWaferCellEnv` (Gym-like) funcional em modo stub (`reset/step/reward/info`)
- `SLMPolicy` com fallback MLP temporário em PyTorch/CUDA (pronto para trocar pelo SLM RL real)
- Demo RL ponta a ponta com métricas e latência
- Contrato ROS2 estável: `wafer_cell_ros2_v1`

## Resultado real (stub backend)
Arquivo: `docs/results_stub_backend.json`

- `reward_mean`: ~267.16
- `wafer_quality_score_mean`: ~0.9468
- `coord_quality_mean`: ~0.9598
- `defect_rate_proxy`: ~0.0560
- `infer_p50_ms`: ~0.8241
- `control_loop_mean_ms`: ~20.19 (alinhado a ~50 Hz)

## Arquitetura
Fluxo de dados atual:
- `SLMPolicy.act(obs)` -> ações contínuas (3 braços)
- `IsaacLabWaferCellEnv.step(action)` -> publica ROS2 / lê estados / calcula reward
- `wafer_cell_sim_node` -> backend stub (futuro: backend Isaac Lab real)
- tópicos ROS2 -> `/arm_i/joint_states`, `/arm_i/joint_command`, `/cell/wafer_state`

## Estrutura relevante
- `src/slm_wafer_cell/envs/isaaclab_wafer_cell_env.py`
- `src/slm_wafer_cell/policy/slm_policy_loader.py`
- `src/slm_wafer_cell/demo_isaaclab_wafer_cell.py`
- `src/slm_wafer_cell/ros2/protocol.py`
- `ros2_ws/src/wafer_cell_bringup/wafer_cell_bringup/wafer_cell_sim_node.py`
- `src/slm_wafer_cell/sim/isaaclab_scene_loader.py` (skeleton para integração Isaac Lab)

## Requisitos
- Pop!_OS / Ubuntu 24.04
- ROS2 Jazzy instalado em `/opt/ros/jazzy`
- Python 3.12
- venv com `--system-site-packages` para enxergar `rclpy`
- PyTorch (CUDA) na venv

## Setup Python (N2)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Exemplo GPU (ajuste conforme sua máquina)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Build ROS2
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Rodar stub backend (ROS2)
```bash
ros2 run wafer_cell_bringup wafer_cell_sim --ros-args -p backend:=stub -p publish_rate_hz:=50.0
```

## Rodar demo RL (stub backend)
Em outro terminal:
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python src/slm_wafer_cell/demo_isaaclab_wafer_cell.py \
  --checkpoint checkpoints/best.pt \
  --device cuda \
  --episodes 2 \
  --steps-per-episode 100 \
  --obs-dim 64 \
  --act-dim 18
```

## Contrato ROS2 (estável)
Ver:
- `docs/N2_ROS2_PROTOCOL.md`
- `src/slm_wafer_cell/config/ros2_contract_stub_v1.yaml`

## Próxima etapa (quando Isaac Lab estiver instalado)
- Implementar `IsaacLabSceneLoader` real (`src/slm_wafer_cell/sim/isaaclab_scene_loader.py`)
- Ligar `backend:=isaaclab` no `wafer_cell_sim_node.py`
- Substituir stub por estados reais da cena (mantendo o mesmo contrato ROS2)

## Observações
- `checkpoints/*.pt` está no `.gitignore` por padrão (peso local). Se quiser publicar o checkpoint, remova a regra e adicione explicitamente.
