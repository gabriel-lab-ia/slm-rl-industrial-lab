"""SLM RL policy loader (checkpoint -> low-latency act(obs) interface)."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


class SLMPolicy:
    """Policy adapter for the wafer-cell project.

    Preferred path: dynamically import the real `MathCoreAgent` from the base RL project and load the
    checkpoint weights. If that fails, fallback to a small local MLP so the pipeline still runs.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        obs_dim: int = 64,
        act_dim: int = 18,
        base_project_root: str | Path | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_str = device
        self.model: Any | None = None
        self.obs_dim: int | None = int(obs_dim)
        self.act_dim: int | None = int(act_dim)
        self.native_obs_dim: int | None = None
        self.native_act_dim: int | None = None
        self.device = None
        self.ckpt_metrics: dict[str, Any] = {}
        self.ckpt_update: int | None = None
        self._uses_fallback_mlp = True
        self._last_infer_ms: float = 0.0
        self._act_mode = "fallback_mlp"
        self._mathcore_agent = None
        self._mathcore_action_map: str | None = None
        self.base_project_root = Path(base_project_root) if base_project_root else Path(
            "/home/aipowerisraelense/AI/ML-Edge-AI-Cinematic-Dashboard/rl-isaac-mathcore-humanoid"
        )
        self._load_model()

    def _resolve_device(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch não disponível no ambiente Python.")
        req = str(self.device_str).lower()
        if req.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(req)
        elif req.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(req)

    def _infer_dims_from_state(self, state: dict[str, Any]) -> tuple[int | None, int | None]:
        obs_dim = None
        act_dim = None
        try:
            b = state.get("encoder.fourier.b")
            if hasattr(b, "shape") and len(b.shape) == 2:
                obs_dim = int(b.shape[0])
        except Exception:
            pass
        try:
            lstd = state.get("policy_head.log_std")
            if hasattr(lstd, "shape") and len(lstd.shape) >= 1:
                act_dim = int(lstd.shape[0])
        except Exception:
            pass
        return obs_dim, act_dim

    def _load_yaml_model_cfg(self) -> dict[str, Any] | None:
        if yaml is None:
            return None
        candidates = [
            self.base_project_root / "configs" / "model_ultralite.yaml",
            self.base_project_root / "configs" / "model.yaml",
        ]
        for p in candidates:
            if not p.exists():
                continue
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("model"), dict):
                    return dict(data["model"])
            except Exception:
                continue
        return None

    def _try_load_mathcore_agent(self, ckpt: dict[str, Any]) -> bool:
        if torch is None:
            return False
        if not self.base_project_root.exists():
            return False

        state = ckpt.get("model_state_dict", ckpt)
        if not isinstance(state, dict):
            return False

        # Add base project root to import path
        root_str = str(self.base_project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            mod = importlib.import_module("mathcore.agent")
            MathCoreAgent = getattr(mod, "MathCoreAgent")
        except Exception:
            return False

        model_cfg = self._load_yaml_model_cfg() or {}
        inf_obs, inf_act = self._infer_dims_from_state(state)
        if inf_obs is not None:
            model_cfg["obs_dim"] = inf_obs
        if inf_act is not None:
            model_cfg["act_dim"] = inf_act
        if "obs_dim" not in model_cfg or "act_dim" not in model_cfg:
            return False

        try:
            agent = MathCoreAgent(**model_cfg).to(self.device).eval()
            agent.load_state_dict(state, strict=False)
        except Exception:
            return False

        self.model = agent
        self._mathcore_agent = agent
        self._uses_fallback_mlp = False
        self._act_mode = "mathcore_agent"
        self.native_obs_dim = int(model_cfg["obs_dim"])
        self.native_act_dim = int(model_cfg["act_dim"])
        try:
            agent.reset_memory(batch_size=1, device=self.device)
        except Exception:
            pass
        return True

    def _load_model(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch não disponível no ambiente Python.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {self.checkpoint_path}")

        self._resolve_device()
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            self.ckpt_metrics = dict(ckpt.get("metrics", {}) or {})
            if "update" in ckpt:
                try:
                    self.ckpt_update = int(ckpt["update"])
                except Exception:
                    self.ckpt_update = None

        if isinstance(ckpt, dict) and self._try_load_mathcore_agent(ckpt):
            # Adapter dimensions for N2 wrapper still come from requested env interface.
            if self.obs_dim is None:
                self.obs_dim = 64
            if self.act_dim is None:
                self.act_dim = 18
            return

        # Fallback MLP temporário (determinístico) se MathCoreAgent não puder ser importado.
        if self.obs_dim is None or self.act_dim is None:
            self.obs_dim = 64
            self.act_dim = 18

        seed = 1234 + int(self.ckpt_update or 0)
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
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.25)
                    torch.nn.init.zeros_(m.bias)

    @property
    def last_infer_ms(self) -> float:
        return self._last_infer_ms

    def _adapt_obs(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.size < target_dim:
            out = np.zeros((target_dim,), dtype=np.float32)
            out[: x.size] = x
            return out
        return x[:target_dim]

    def _map_action_native_to_env(self, raw: np.ndarray, env_act_dim: int) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.size == env_act_dim:
            return np.clip(raw, -1.0, 1.0)
        if raw.size == 12 and env_act_dim == 18:
            # Common case: 3 arms x 4 controls -> 3 arms x 6 joints. Deterministic geometric expansion.
            out = np.zeros((18,), dtype=np.float32)
            for arm in range(3):
                a = raw[arm * 4 : (arm + 1) * 4]
                j = arm * 6
                out[j + 0] = a[0]
                out[j + 1] = a[1]
                out[j + 2] = a[2]
                out[j + 3] = a[3]
                out[j + 4] = 0.5 * (a[1] - a[2])
                out[j + 5] = 0.7 * a[3] + 0.2 * a[0]
            self._mathcore_action_map = "12_to_18_geometric"
            return np.clip(out, -1.0, 1.0)
        # Generic fallback: repeat/truncate
        out = np.resize(raw, env_act_dim).astype(np.float32, copy=False)
        self._mathcore_action_map = f"resize_{raw.size}_to_{env_act_dim}"
        return np.clip(out, -1.0, 1.0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None or self.act_dim is None:
            raise RuntimeError("SLMPolicy não inicializada corretamente.")

        t0 = time.perf_counter()
        with torch.no_grad():
            if self._mathcore_agent is not None and self.native_obs_dim is not None and self.native_act_dim is not None:
                x = self._adapt_obs(obs, self.native_obs_dim)
                xt = torch.from_numpy(x).to(self.device, dtype=torch.float32).unsqueeze(0)
                # Deterministic low-latency path
                mean, _log_std, _value, _z = self._mathcore_agent(xt, temporal=True)
                raw = torch.tanh(mean).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                action = self._map_action_native_to_env(raw, int(self.act_dim))
            else:
                x = self._adapt_obs(obs, int(self.obs_dim or 64))
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
            "native_obs_dim": int(self.native_obs_dim or 0),
            "native_act_dim": int(self.native_act_dim or 0),
            "uses_fallback_mlp": bool(self._uses_fallback_mlp),
            "act_mode": self._act_mode,
            "mathcore_action_map": self._mathcore_action_map,
            "ckpt_update": self.ckpt_update,
            "ckpt_metrics": self.ckpt_metrics,
            "base_project_root": str(self.base_project_root),
        }
