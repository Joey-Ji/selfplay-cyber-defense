"""Compute cross-play matrices for all trained configurations.

Usage::

    uv run python scripts/eval_crossplay.py
    uv run python scripts/eval_crossplay.py --config experiments/quick.toml
"""

from absl import app, flags

from selfplay import config
from selfplay.eval.crossplay import evaluate_crossplay

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "experiments/paper.toml", "Path to experiment config.")


def main(_):
    batch_config = config.load(FLAGS.config)
    evaluate_crossplay(batch_config)


if __name__ == "__main__":
    app.run(main)
