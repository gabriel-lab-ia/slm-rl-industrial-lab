# slm-rl-isaaclab-wafer-cell-3arms (N2)

Projeto N2 para integrar um **SLM RL (MathCore)** de baixa latência a uma célula mecatrônica de wafer via **ROS2 Jazzy** e backend de simulação **MuJoCo** (com roadmap para Isaac Lab).

## Estado atual (publicável)
- Bridge ROS2 Jazzy funcional (`wafer_cell_bringup`)
- Backend `mujoco` visual funcional (cena industrial simplificada)
- Modo `math_lab` para validação cinemática/álgebra linear
- `IsaacLabWaferCellEnv` (Gym-like via ROS2) funcional
- `SLMPolicy` integrado ao **MathCoreAgent real** (projeto base), com fallback MLP
- Demo RL ponta a ponta (ROS2 -> env -> policy -> MuJoCo)
- Contrato ROS2 estável: `wafer_cell_ros2_v1`

## Arquitetura (alto nível)
- `MathCoreAgent` (projeto base) -> inferência de ação (CUDA)
- `SLMPolicy` -> adapta ação nativa do modelo para `act_dim=18` (3 braços x 6)
- `IsaacLabWaferCellEnv` -> publica comandos ROS2, lê estados, calcula reward
- `wafer_cell_sim_node` -> backend `stub`/`mujoco`/`isaaclab` (skeleton)
- MuJoCo -> cena visual + física + stepping

## Projeto base (cérebro RL)
Por padrão o `SLMPolicy` tenta carregar o agente real de:
- `/home/aipowerisraelense/AI/ML-Edge-AI-Cinematic-Dashboard/rl-isaac-mathcore-humanoid`

Checkpoint esperado no N2 (copiado localmente):
- `checkpoints/best.pt`

## Backend MuJoCo: modos disponíveis
### `industrial`
Cena industrial simplificada com wafer/posicionador e braço ativo.

### `math_lab`
Modo de validação cinemática com:
- alvos geométricos (`line`, `circle`, `spiral`, `figure8`, `rotate_extend`)
- DLS IK (`dq = J^T (J J^T + λ²I)^-1 e`)
- regularização de postura (null-space simples)
- métricas exportadas em JSON

## Resultados atuais (math_lab baseline)
Arquivos em `docs/`:
- `results_math_lab_line.json`
- `results_math_lab_circle.json`
- `results_math_lab_spiral.json`
- `results_math_lab_figure8.json`

Resumo (tracking RMS, menor é melhor):
- `line`: ~0.6279
- `circle`: ~0.6191
- `spiral`: ~0.6138 (melhor no baseline atual)
- `figure8`: ~0.6221

## Estrutura relevante
- `src/slm_wafer_cell/envs/isaaclab_wafer_cell_env.py`
- `src/slm_wafer_cell/policy/slm_policy_loader.py`
- `src/slm_wafer_cell/demo_isaaclab_wafer_cell.py`
- `src/slm_wafer_cell/ros2/protocol.py`
- `ros2_ws/src/wafer_cell_bringup/wafer_cell_bringup/wafer_cell_sim_node.py`
- `src/slm_wafer_cell/sim/isaaclab_scene_loader.py` (skeleton)
- `scripts/manual_joint_probe.sh`
- `scripts/compare_math_lab_scenarios.sh`

## Requisitos
- Pop!_OS / Ubuntu 24.04
- ROS2 Jazzy instalado em `/opt/ros/jazzy`
- Python 3.12
- `venv` com `--system-site-packages` (para `rclpy`)
- PyTorch com CUDA (opcional, recomendado)

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

## Rodar MuJoCo (visual) em `math_lab`
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python ros2_ws/src/wafer_cell_bringup/wafer_cell_bringup/wafer_cell_sim_node.py --ros-args \
  -p backend:=mujoco \
  -p mujoco_mode:=math_lab \
  -p mujoco_math_scenario:=spiral \
  -p debug_log_every_n:=25 \
  -p headless:=false \
  -p publish_rate_hz:=50.0 \
  -p mujoco_realtime:=true
```

## Teste manual (dobrar/girar/esticar)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
bash scripts/manual_joint_probe.sh fold
bash scripts/manual_joint_probe.sh yaw
bash scripts/manual_joint_probe.sh extend
```

## Rodar demo RL (MathCoreAgent via `SLMPolicy`)
Em outro terminal:
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python src/slm_wafer_cell/demo_isaaclab_wafer_cell.py \
  --checkpoint checkpoints/best.pt \
  --device cuda \
  --episodes 2 \
  --steps-per-episode 150 \
  --obs-dim 64 \
  --act-dim 18
```

## Comparar cenários `math_lab` (headless)
```bash
cd ~/projects/slm-rl-isaaclab-wafer-cell-3arms
bash scripts/compare_math_lab_scenarios.sh
```

## Tópicos ROS2 principais
- `/arm_1/joint_states`, `/arm_2/joint_states`, `/arm_3/joint_states`
- `/arm_1/joint_command`, `/arm_2/joint_command`, `/arm_3/joint_command`
- `/cell/wafer_state`
- `/cell/reset`
- `/cell/math_metrics` (snapshot de métricas no `math_lab`)

## Roadmap (próximas etapas)
- Treino/fine-tuning específico no MuJoCo (RL/IL)
- Modelo de braço real (URDF/MJCF + malhas)
- Sensores simulados (contato, F/T, RGB-D)
- Dashboard web em tempo real (FastAPI + HTML/WebSocket)
- Backend Isaac Lab real (mantendo o contrato ROS2)

## Observações de publicação
- `checkpoints/*.pt` está no `.gitignore` por padrão (não versionado)
- Artefatos locais de fechamento (`manifests/*closeout*`) e logs temporários do MuJoCo também ficam ignorados
- Se quiser publicar um checkpoint demo, adicione explicitamente com `git add -f checkpoints/best.pt`
