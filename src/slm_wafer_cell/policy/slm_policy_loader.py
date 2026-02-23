"""SLM RL policy loader (checkpoint -> low-latency act(obs) interface)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class SLMPolicy:
    """Minimal policy adapter for the wafer-cell project.

    This skeleton intentionally decouples the N2 project from the previous repository internals.
    TODO: copy/import the exact `MathCoreAgent` architecture from the base project and finalize mapping.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        obs_dim: int = 64,
        act_dim: int = 18,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_str = device
        self.model: Any | None = None
        self.obs_dim: int | None = int(obs_dim)
        self.act_dim: int | None = int(act_dim)
        self.device = None
        self.ckpt_metrics: dict[str, Any] = {}
        self.ckpt_update: int | None = None
        self._uses_fallback_mlp = True
        self._last_infer_ms: float = 0.0
        self._load_model()

    def _load_model(self) -> None:
        """Load checkpoint and instantiate the SLM RL policy.

        TODO (integração real):
        1. Copiar `mathcore/` + config do projeto base ou transformar em dependência instalável.
        2. Instanciar `MathCoreAgent(**model_cfg)` correto.
        3. Carregar `model_state_dict` de `checkpoints/best.pt`.
        4. Definir `self.obs_dim` e `self.act_dim` a partir do modelo/config.
        5. Opcional: `torch.compile` (somente se estável com memória temporal desabilitada).
        """
        if torch is None:
            raise RuntimeError("PyTorch não disponível no ambiente Python.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {self.checkpoint_path}")

        req = str(self.device_str).lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(req)
        elif req.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(req)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            self.ckpt_metrics = dict(ckpt.get("metrics", {}) or {})
            if "update" in ckpt:
                try:
                    self.ckpt_update = int(ckpt["update"])
                except Exception:
                    self.ckpt_update = None

        # Fallback MLP temporário (determinístico) para fechar loop do N2 enquanto a arquitetura SLM real é portada.
        # TODO (próxima etapa): substituir por MathCoreAgent real + adapter 12->18 juntas.
        if self.obs_dim is None or self.act_dim is None:
            self.obs_dim = 64
            self.act_dim = 18

        seed = 1234
        if self.ckpt_update is not None:
            seed = 1234 + int(self.ckpt_update)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(self.obs_dim), 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, int(self.act_dim)),
            torch.nn.Tanh(),
        ).to(self.device).eval()

        # Inicialização pequena para ação suave e baixa oscilação no stub.
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.25)
                    torch.nn.init.zeros_(m.bias)

    @property
    def last_infer_ms(self) -> float:
        return self._last_infer_ms

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Run a single inference step and return continuous action vector.

        Args:
            obs: np.array 1D normalizado/compatível com o ambiente.

        Returns:
            np.array 1D de comandos contínuos para juntas dos 3 braços.
        """
        if torch is None or self.model is None or self.obs_dim is None or self.act_dim is None:
            raise RuntimeError("SLMPolicy não inicializada corretamente.")
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.size < self.obs_dim:
            padded = np.zeros((self.obs_dim,), dtype=np.float32)
            padded[: x.size] = x
            x = padded
        elif x.size > self.obs_dim:
            x = x[: self.obs_dim]
        t0 = time.perf_counter()

        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device, dtype=torch.float32).unsqueeze(0)
            yt = self.model(xt)
            action = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        self._last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return action

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "obs_dim": int(self.obs_dim or 0),
            "act_dim": int(self.act_dim or 0),
            "uses_fallback_mlp": bool(self._uses_fallback_mlp),
            "ckpt_update": self.ckpt_update,
            "ckpt_metrics": self.ckpt_metrics,
            # TODO: quando integrar o SLM real, expor nome da arquitetura e adapter 12->18.
        }
