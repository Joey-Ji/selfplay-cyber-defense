"""Shared utilities: model loading, agent factory."""

from __future__ import annotations

from pathlib import Path

import torch

from selfplay.env.agents import (
    BLineAgent,
    BlueSleepAgent,
    MeanderAgent,
    ReactRestoreAgent,
    RestoreDecoysAgent,
)
from selfplay.training.ppo import PPO as CustomPPO


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_ppo(path: Path, obs_dim: int, act_n: int) -> CustomPPO:
    """Create a custom PPO model and load a checkpoint into it."""
    model = CustomPPO(obs_dim=obs_dim, act_n=act_n, device="cpu")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.actor_critic.load_state_dict(state_dict["model_state_dict"])
    else:
        model.actor_critic.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# Scripted agent factory
# ---------------------------------------------------------------------------

SCRIPTED_AGENT_REGISTRY: dict[str, type] = {
    "red_bline": BLineAgent,
    "red_meander": MeanderAgent,
    "blue_sleep": BlueSleepAgent,
    "blue_react_restore": ReactRestoreAgent,
    "blue_restore_decoys": RestoreDecoysAgent,
}


def make_scripted_agent(name: str):
    """Create a scripted agent by name.

    Supports red agents: red_bline, red_meander
    Supports blue agents: blue_sleep, blue_react_restore, blue_restore_decoys
    """
    cls = SCRIPTED_AGENT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown scripted agent: {name!r}. "
            f"Available: {sorted(SCRIPTED_AGENT_REGISTRY)}"
        )
    return cls()
