"""Collect (blue_obs, blue_action) demonstrations from React-Restore agent."""

from __future__ import annotations

import numpy as np
from absl import logging

from selfplay.env.agents import BLineAgent, MeanderAgent, ReactRestoreAgent
from selfplay.env.cage import SimplifiedCAGE


def collect_blue(
    num_episodes: int = 10000,
    max_steps: int = 100,
    red_agents: list[str] | None = None,
    remove_bugs: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run React-Restore as Blue vs scripted red agents and collect blue demonstrations."""

    if red_agents is None:
        red_agents = ["bline", "meander"]

    all_obs = []
    all_actions = []
    eps_per_agent = num_episodes // len(red_agents)

    red_factories = {"bline": BLineAgent, "b_line": BLineAgent, "meander": MeanderAgent}

    for red_name in red_agents:
        make_red = red_factories.get(red_name)
        if make_red is None:
            raise ValueError(f"Unknown red agent: {red_name}")

        batch_size = min(eps_per_agent, 256)
        sim = SimplifiedCAGE(num_envs=batch_size, remove_bugs=remove_bugs)
        blue = ReactRestoreAgent()

        collected = 0
        while collected < eps_per_agent:
            red = make_red()
            blue.reset()

            obs_dict, _ = sim.reset()
            red_obs = obs_dict["Red"]
            blue_obs = obs_dict["Blue"]

            for step in range(max_steps):
                red_action = red.get_action(red_obs)
                blue_action = blue.get_action(blue_obs)

                for i in range(batch_size):
                    all_obs.append(blue_obs[i].copy())
                    all_actions.append(int(blue_action[i, 0]))

                obs_dict, _, _, _ = sim.step(
                    red_action=red_action.reshape(batch_size, 1).astype(np.int32),
                    blue_action=blue_action.reshape(batch_size, 1).astype(np.int32),
                    red_agent=red if isinstance(red, BLineAgent) else None,
                )
                red_obs = obs_dict["Red"]
                blue_obs = obs_dict["Blue"]

            collected += batch_size

        logging.info("Collected %d episodes vs %s", collected, red_name)

    obs_array = np.array(all_obs, dtype=np.float32)
    act_array = np.array(all_actions, dtype=np.int64)
    logging.info("Total Blue demonstrations: %d transitions", len(obs_array))
    return obs_array, act_array
