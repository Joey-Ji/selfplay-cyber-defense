"""Train a self-play Red vs Blue agent pair.

Usage:
    python scripts/train.py --config.selfplay.strategy=vanilla --config.selfplay.total_rounds=50
"""

from absl import app
import fancyflags as ff

from selfplay import config
from selfplay.training.selfplay import run_selfplay

_CONFIG = ff.DEFINE_from_instance("config", config.ExperimentConfig())


def main(_):
    run_selfplay(_CONFIG.value())


if __name__ == "__main__":
    app.run(main)
