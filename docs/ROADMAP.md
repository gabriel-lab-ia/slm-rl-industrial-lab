# Roadmap N2 (Stub -> Isaac Lab)

## Fase atual (concluída)
- [x] ROS2 Jazzy bridge funcional com tópicos e reset
- [x] Backend stub determinístico de 3 braços + wafer/processo
- [x] Env Gym-like sobre ROS2
- [x] Demo RL ponta a ponta com métricas e latência
- [x] Contrato ROS2 estável `wafer_cell_ros2_v1`

## Próxima fase (Isaac Lab real)
- [ ] Instalar/validar `isaaclab` ou `omni.isaac.lab` no host
- [ ] Implementar scene loader real (spawn 3 braços + wafer/chuck)
- [ ] Publicar estados reais via `/cell/wafer_state`
- [ ] Adapter de comandos de junta -> controladores Isaac Lab
- [ ] Demo visual com `backend:=isaaclab`

## Fase de produção
- [ ] Trocar `SLMPolicy` fallback MLP pelo SLM RL real (MathCoreAgent + adapter 12->18)
- [ ] Métricas reais de throughput/defect por ciclo
- [ ] Cenários de teste / benchmark de cadência
- [ ] Vídeo e screenshots para GitHub
