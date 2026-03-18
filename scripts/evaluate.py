"""Evaluate Red vs Blue agents."""

from absl import app, flags
import numpy as np
import torch

from selfplay.eval.evaluate import load_agent, evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string("red", None, "Red agent (path or name).", required=True)
flags.DEFINE_string("blue", None, "Blue agent (path or name).", required=True)
flags.DEFINE_integer("episodes", 1000, "Number of evaluation episodes.")
flags.DEFINE_integer("max_steps", 100, "Max steps per episode.")
flags.DEFINE_integer("seed", 42, "Random seed.")


def main(_):
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    red = load_agent(FLAGS.red, "red")
    blue = load_agent(FLAGS.blue, "blue")

    results = evaluate(red, blue, num_episodes=FLAGS.episodes, max_steps=FLAGS.max_steps)

    print(f"\nResults ({FLAGS.episodes} episodes):")
    print(f"  Red  reward: {results['red_reward_mean']:.2f} +/- {results['red_reward_std']:.2f}")
    print(f"  Blue reward: {results['blue_reward_mean']:.2f} +/- {results['blue_reward_std']:.2f}")
    print(f"  Red win rate: {results['red_win_rate']:.3f}")


if __name__ == "__main__":
    app.run(main)
