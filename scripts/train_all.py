"""Train all configurations defined in an experiment config.

Usage::

    uv run python scripts/train_all.py
    uv run python scripts/train_all.py --config experiments/quick.toml --parallel
"""

from absl import app, flags

from selfplay import config
from selfplay.training.batch import train_all

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "experiments/paper.toml", "Path to experiment config.")
flags.DEFINE_bool("parallel", False, "Run experiments in parallel.")


def main(_):
    batch_config = config.load(FLAGS.config)
    train_all(batch_config, parallel=FLAGS.parallel)


if __name__ == "__main__":
    app.run(main)
