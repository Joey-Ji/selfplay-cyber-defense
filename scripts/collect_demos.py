"""Collect B-line demonstrations for behavioral cloning."""

from pathlib import Path

from absl import app, flags, logging
import numpy as np
import torch

from selfplay.training.collect import collect

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of episodes to collect.")
flags.DEFINE_string("output", "data/bline_demos.npz", "Output path for demonstrations.")
flags.DEFINE_integer("seed", 42, "Random seed.")


def main(_):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    Path(FLAGS.output).parent.mkdir(parents=True, exist_ok=True)

    obs, actions = collect(num_episodes=FLAGS.num_episodes)
    np.savez(FLAGS.output, observations=obs, actions=actions)
    logging.info("Saved to %s", FLAGS.output)


if __name__ == "__main__":
    app.run(main)
