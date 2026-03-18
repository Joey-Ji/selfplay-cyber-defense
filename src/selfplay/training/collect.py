"""Collect (red_obs, red_action) demonstrations from scripted B-line agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from absl import logging

from selfplay.env.agents import BLineAgent
from selfplay.env.cage import SimplifiedCAGE
from selfplay.utils import make_scripted_agent


def collect(
    num_episodes: int = 10000,
    max_steps: int = 100,
    blue_agents: list[str] | None = None,
    remove_bugs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run B-line vs scripted blue agents and collect red demonstrations."""

    if blue_agents is None:
        blue_agents = ["blue_sleep", "blue_react_restore"]

    all_obs = []
    all_actions = []
    eps_per_agent = num_episodes // len(blue_agents)

    for blue_name in blue_agents:
        blue = make_scripted_agent(blue_name)

        batch_size = min(eps_per_agent, 256)
        sim = SimplifiedCAGE(num_envs=batch_size, remove_bugs=remove_bugs)

        collected = 0
        while collected < eps_per_agent:
            red = BLineAgent()
            if hasattr(blue, "reset"):
                blue.reset()

            obs_dict, _ = sim.reset()
            red_obs = obs_dict["Red"]
            blue_obs = obs_dict["Blue"]

            for step in range(max_steps):
                red_action = red.get_action(red_obs)
                blue_action = blue.get_action(blue_obs)

                for i in range(batch_size):
                    all_obs.append(red_obs[i].copy())
                    all_actions.append(int(red_action[i, 0]))

                obs_dict, _, _, _ = sim.step(
                    red_action=red_action.reshape(batch_size, 1).astype(np.int32),
                    blue_action=blue_action.reshape(batch_size, 1).astype(np.int32),
                    red_agent=red,
                )
                red_obs = obs_dict["Red"]
                blue_obs = obs_dict["Blue"]

            collected += batch_size

        logging.info("Collected %d episodes vs %s", collected, blue_name)

    obs_array = np.array(all_obs, dtype=np.float32)
    act_array = np.array(all_actions, dtype=np.int64)
    logging.info("Total demonstrations: %d transitions", len(obs_array))
    return obs_array, act_array
