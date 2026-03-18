"""Environment layer: MiniCAGE simulation, agents, and gym wrappers.

Public API:
    SimplifiedCAGE  — Core vectorized CAGE2 environment
    HOSTS           — List of host names in the network
    SelfPlayBlueWrapper / SelfPlayRedWrapper — Self-play gym wrappers
    MiniCageBlue    — Single-agent Blue gym wrapper (scripted Red)
"""

from selfplay.env.cage import SimplifiedCAGE, HOSTS
from selfplay.env.wrappers import SelfPlayBlueWrapper, SelfPlayRedWrapper
from selfplay.env.blue_wrapper import MiniCageBlue

# Shared observation/action space dimensions.
RED_OBS_DIM = 1 + 3 * len(HOSTS)  # 40
RED_ACT_N = 56
BLUE_OBS_DIM = 6 * len(HOSTS)     # 78
BLUE_ACT_N = 53

__all__ = [
    "SimplifiedCAGE",
    "HOSTS",
    "SelfPlayBlueWrapper",
    "SelfPlayRedWrapper",
    "MiniCageBlue",
    "RED_OBS_DIM",
    "RED_ACT_N",
    "BLUE_OBS_DIM",
    "BLUE_ACT_N",
]
