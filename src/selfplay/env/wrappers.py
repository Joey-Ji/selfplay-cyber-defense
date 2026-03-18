"""Self-play gym wrappers that load opponents from checkpoint pools."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from selfplay.training.ppo import ActorCritic

from selfplay.env.cage import HOSTS, SimplifiedCAGE


# Obs/action dims per role.
_ROLE_DIMS: dict[str, tuple[int, int]] = {
    "Blue": (6 * len(HOSTS), None),  # act_n filled from sim
    "Red": (1 + 3 * len(HOSTS), None),
}


class SelfPlayBlueWrapper(gym.Env):
    """Blue trains, Red opponent loaded from a checkpoint pool or scripted agent."""

    metadata = {"render_modes": []}

    def __init__(self, remove_bugs: bool = True, max_steps: int = 100):
        super().__init__()

        self.role = "Blue"
        self.sim = SimplifiedCAGE(num_envs=1, remove_bugs=remove_bugs)

        obs_dim = 6 * len(HOSTS)
        self.action_space = spaces.Discrete(len(self.sim.action_mapping["Blue"]))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self.max_steps = max_steps
        self.steps_done = 0

        self._red_model: ActorCritic | None = None
        self._red_obs = None
        self._scripted_red = None

    def set_opponent(self, checkpoint_path: Path | str | None) -> None:
        self._scripted_red = None  # learned model takes over

        if checkpoint_path is None:
            self._red_model = None
            return

        if isinstance(checkpoint_path, str):
            candidate = checkpoint_path.strip()
            if candidate and not _looks_like_checkpoint_path(candidate):
                from selfplay.utils import make_scripted_agent
                self._scripted_red = make_scripted_agent(candidate)
                self._red_model = None
                return
            checkpoint_path = Path(candidate)
        else:
            checkpoint_path = Path(checkpoint_path)

        if self._red_model is None:
            red_obs_dim = 1 + 3 * len(HOSTS)
            red_act_n = len(self.sim.action_mapping["Red"])
            self._red_model = ActorCritic(red_obs_dim, red_act_n)

        self._scripted_red = None
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self._red_model.load_state_dict(state_dict)
        self._red_model.eval()

    def set_scripted_opponent(self, agent):
        """Use a scripted red agent instead of learned checkpoint."""
        self._scripted_red = agent

    def _get_red_action(self, red_obs: np.ndarray) -> np.ndarray:
        if self._scripted_red is not None:
            action = self._scripted_red.get_action(red_obs)
            return np.array(action, dtype=np.int32).reshape(1, 1)
        if self._red_model is None:
            return np.array([[0]], dtype=np.int32)  # sleep
        with torch.no_grad():
            obs_t = torch.tensor(red_obs.astype(np.float32)).unsqueeze(0)
            logits, _ = self._red_model(obs_t)
            action = torch.distributions.Categorical(logits=logits).sample()
        return np.array([[int(action.item())]], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_done = 0

        if self._scripted_red is not None and hasattr(self._scripted_red, "reset"):
            self._scripted_red.reset()

        obs_dict, info = self.sim.reset()
        self._red_obs = obs_dict["Red"][0]
        blue_obs = obs_dict["Blue"][0].astype(np.float32)
        mask = self.sim.get_mask(self.sim.state, self.sim.current_decoys)
        return blue_obs, {"action_mask": mask["Blue"][0].astype(np.float32)}

    def step(self, blue_action):
        self.steps_done += 1

        red_action = self._get_red_action(self._red_obs)
        blue_action = np.array([[blue_action]], dtype=np.int32)

        obs_dict, reward_dict, terminated, info = self.sim.step(
            red_action=red_action, blue_action=blue_action,
        )

        self._red_obs = obs_dict["Red"][0]
        blue_obs = obs_dict["Blue"][0].astype(np.float32)
        reward = float(reward_dict["Blue"][0][0])
        done = self.steps_done >= self.max_steps
        mask = self.sim.get_mask(self.sim.state, self.sim.current_decoys)
        info["action_mask"] = mask["Blue"][0].astype(np.float32)
        return blue_obs, reward, done, False, info


class SelfPlayRedWrapper(gym.Env):
    """Red trains, Blue opponent loaded from a checkpoint pool."""

    metadata = {"render_modes": []}

    def __init__(self, remove_bugs: bool = True, max_steps: int = 100):
        super().__init__()

        self.sim = SimplifiedCAGE(num_envs=1, remove_bugs=remove_bugs)

        self.role = "Red"
        self.action_space = spaces.Discrete(len(self.sim.action_mapping["Red"]))

        obs_dim = 1 + 3 * len(HOSTS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.max_steps = max_steps
        self.steps_done = 0

        self._blue_model: ActorCritic | None = None
        self._blue_obs = None
        self._scripted_blue = None

    def set_opponent(self, checkpoint_path: Path | str | None) -> None:
        self._scripted_blue = None  # learned model takes over
        if checkpoint_path is None:
            self._blue_model = None
            return

        checkpoint_path = Path(checkpoint_path)

        if self._blue_model is None:
            blue_obs_dim = 6 * len(HOSTS)
            blue_act_n = len(self.sim.action_mapping["Blue"])
            self._blue_model = ActorCritic(blue_obs_dim, blue_act_n)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self._blue_model.load_state_dict(state_dict)
        self._blue_model.eval()

    def set_scripted_opponent(self, agent):
        """Use a scripted blue agent instead of learned checkpoint."""
        self._scripted_blue = agent

    def _get_blue_action(self, blue_obs: np.ndarray) -> np.ndarray:
        if self._scripted_blue is not None:
            action = self._scripted_blue.get_action(blue_obs)
            return np.array(action, dtype=np.int32).reshape(1, 1)
        if self._blue_model is None:
            return np.array([[0]], dtype=np.int32)
        with torch.no_grad():
            obs_t = torch.tensor(blue_obs.astype(np.float32)).unsqueeze(0)
            logits, _ = self._blue_model(obs_t)
            action = torch.distributions.Categorical(logits=logits).sample()
        return np.array([[int(action.item())]], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_done = 0
        obs_dict, info = self.sim.reset()
        self._blue_obs = obs_dict["Blue"][0]
        red_obs = obs_dict["Red"][0].astype(np.float32)
        mask = self.sim.get_mask(self.sim.state, self.sim.current_decoys)
        return red_obs, {"action_mask": mask["Red"][0].astype(np.float32)}

    def step(self, red_action):
        self.steps_done += 1

        blue_action = self._get_blue_action(self._blue_obs)
        red_action = np.array([[red_action]], dtype=np.int32)

        obs_dict, reward_dict, terminated, info = self.sim.step(
            red_action=red_action, blue_action=blue_action,
        )

        self._blue_obs = obs_dict["Blue"][0]
        red_obs = obs_dict["Red"][0].astype(np.float32)
        reward = float(reward_dict["Red"][0][0])
        done = self.steps_done >= self.max_steps
        mask = self.sim.get_mask(self.sim.state, self.sim.current_decoys)
        info["action_mask"] = mask["Red"][0].astype(np.float32)
        return red_obs, reward, done, False, info


def _looks_like_checkpoint_path(spec: str) -> bool:
    path = Path(spec)
    return (
        path.exists()
        or path.suffix == ".pt"
        or any(sep in spec for sep in ("/", "\\"))
    )
