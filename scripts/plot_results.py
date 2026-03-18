"""Generate comparative reward-curve plots across all experiments.

Usage::

    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --config experiments/quick.toml
"""

from absl import app, flags

from selfplay import config
from selfplay.viz.plots import plot_comparative_results

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "experiments/paper.toml", "Path to experiment config.")


def main(_):
    batch_config = config.load(FLAGS.config)
    plot_comparative_results(batch_config)


if __name__ == "__main__":
    app.run(main)
