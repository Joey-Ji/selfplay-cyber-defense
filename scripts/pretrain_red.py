"""Pre-train Red via behavioral cloning."""

from absl import app, flags

from selfplay.env import RED_OBS_DIM, RED_ACT_N
from selfplay.training.pretrain import pretrain

FLAGS = flags.FLAGS
flags.DEFINE_string("data", "data/bline_demos.npz", "Path to demonstration data.")
flags.DEFINE_integer("epochs", 50, "Training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("lr", 1e-3, "Learning rate.")
flags.DEFINE_string("output", "out/models/red_pretrained.pt", "Output model path.")
flags.DEFINE_integer("seed", 42, "Random seed.")


def main(_):
    pretrain(
        data_path=FLAGS.data,
        output_path=FLAGS.output,
        obs_dim=RED_OBS_DIM,
        act_n=RED_ACT_N,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        lr=FLAGS.lr,
        seed=FLAGS.seed,
    )


if __name__ == "__main__":
    app.run(main)
