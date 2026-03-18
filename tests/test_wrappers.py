"""Smoke tests for gym wrappers."""

import numpy as np

from selfplay.env.blue_wrapper import MiniCageBlue
from selfplay.env.wrappers import SelfPlayBlueWrapper, SelfPlayRedWrapper


def test_red_selfplay_wrapper():
    env = SelfPlayRedWrapper(max_steps=100)
    obs, info = env.reset()

    assert obs.shape == (40,), f"Red obs shape {obs.shape} != (40,)"
    assert env.action_space.n == 56, f"Red action space {env.action_space.n} != 56"

    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (40,)
        if done:
            break


def test_blue_wrapper():
    env = MiniCageBlue(red_policy="red_bline", max_steps=100)
    obs, info = env.reset()

    assert obs.shape == (78,), f"Blue obs shape {obs.shape} != (78,)"
    assert env.action_space.n == 53, f"Blue action space {env.action_space.n} != 53"

    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (78,)
        if done:
            break


def test_blue_selfplay_wrapper():
    env = SelfPlayBlueWrapper(max_steps=100)
    obs, info = env.reset()

    assert obs.shape == (78,), f"Blue obs shape {obs.shape} != (78,)"
    assert env.action_space.n == 53, f"Blue action space {env.action_space.n} != 53"

    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (78,)
        if done:
            break
