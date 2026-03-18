"""Evaluate Red vs Blue agents using batched SimplifiedCAGE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from selfplay.training.ppo import PPO as CustomPPO, ActorCritic

from selfplay.env import SimplifiedCAGE, HOSTS, RED_OBS_DIM, RED_ACT_N, BLUE_OBS_DIM, BLUE_ACT_N
from selfplay.env.agents import BaseAgent
from selfplay.utils import make_scripted_agent


class PPOAgent:
    """Wraps a custom PPO model to match scripted agent interface."""

    def __init__(self, model):
        self.model = model
        if hasattr(model, 'actor_critic'):
            model.actor_critic.eval()
        elif hasattr(model, 'eval'):
            model.eval()

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        with torch.no_grad():
            a, _ = self.model.predict(observation, deterministic=False)
        return np.array(a).reshape(-1, 1).astype(np.int32)

    def reset(self) -> None:
        pass


def load_agent(path_or_name: str, role: str):
    """Load an agent — either a custom PPO model from .pt path or a scripted agent by name."""
    path = Path(path_or_name)
    if path.exists() and path.suffix == ".pt":
        if role.lower() == "red":
            obs_dim, act_n = RED_OBS_DIM, RED_ACT_N
        else:
            obs_dim, act_n = BLUE_OBS_DIM, BLUE_ACT_N
        model = CustomPPO.load(path, device="cpu", obs_dim=obs_dim, act_n=act_n)
        return PPOAgent(model)
    else:
        return make_scripted_agent(path_or_name)


def _is_scripted(agent) -> bool:
    """Check whether an agent is a scripted BaseAgent (not a PPOAgent wrapper)."""
    return isinstance(agent, BaseAgent)


def evaluate(
    red_agent,
    blue_agent,
    num_episodes: int = 100,
    max_steps: int = 100,
    remove_bugs: bool = True,
) -> dict:
    """Evaluate red vs blue over num_episodes. Returns dict with mean/std rewards."""
    sim = SimplifiedCAGE(num_envs=num_episodes, remove_bugs=remove_bugs)

    if hasattr(red_agent, "reset"):
        red_agent.reset()
    if hasattr(blue_agent, "reset"):
        blue_agent.reset()

    obs_dict, _ = sim.reset()
    red_obs = obs_dict["Red"]
    blue_obs = obs_dict["Blue"]

    total_red_reward = np.zeros(num_episodes)
    total_blue_reward = np.zeros(num_episodes)

    for step in range(max_steps):
        red_action = red_agent.get_action(red_obs)
        blue_action = blue_agent.get_action(blue_obs)

        red_action = np.array(red_action, dtype=np.int32).reshape(num_episodes, 1)
        blue_action = np.array(blue_action, dtype=np.int32).reshape(num_episodes, 1)

        obs_dict, reward_dict, _, _ = sim.step(
            red_action=red_action,
            blue_action=blue_action,
            red_agent=red_agent if _is_scripted(red_agent) else None,
        )

        red_obs = obs_dict["Red"]
        blue_obs = obs_dict["Blue"]
        total_red_reward += reward_dict["Red"].reshape(-1)
        total_blue_reward += reward_dict["Blue"].reshape(-1)

    red_wins = (total_red_reward > 0).mean()

    return {
        "red_reward_mean": float(total_red_reward.mean()),
        "red_reward_std": float(total_red_reward.std()),
        "blue_reward_mean": float(total_blue_reward.mean()),
        "blue_reward_std": float(total_blue_reward.std()),
        "red_win_rate": float(red_wins),
        "num_episodes": num_episodes,
    }
