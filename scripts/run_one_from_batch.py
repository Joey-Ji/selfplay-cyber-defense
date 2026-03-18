from absl import app, flags

from selfplay import config
from selfplay.training.batch import _run_one

FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Batch config path.", required=True)
flags.DEFINE_integer("seed", None, "Seed to run.", required=True)
flags.DEFINE_integer("strategy_index", 0, "Strategy index in the batch config.")


def main(_):
    batch_config = config.load(FLAGS.config)
    strategy = batch_config.strategies[FLAGS.strategy_index]
    _run_one(batch_config, strategy, FLAGS.seed)


if __name__ == "__main__":
    app.run(main)
