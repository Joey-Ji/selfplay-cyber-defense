"""Collect Blue (React-Restore) demonstrations for BC pre-training."""

from pathlib import Path

from absl import app, flags
import numpy as np

from selfplay.training.collect_blue import collect_blue

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 10000, "Number of episodes to collect.")
flags.DEFINE_integer("max_steps", 100, "Max steps per episode.")
flags.DEFINE_string("output", "data/blue_react_restore_demos.npz", "Output path.")


def main(_):
    obs, actions = collect_blue(
        num_episodes=FLAGS.num_episodes,
        max_steps=FLAGS.max_steps,
    )
    Path(FLAGS.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(FLAGS.output, observations=obs, actions=actions)
    print(f"Saved to {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
