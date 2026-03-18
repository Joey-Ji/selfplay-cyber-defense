"""Batch-train all strategies x seeds from an experiment batch_config."""

from __future__ import annotations

import shutil
from multiprocessing import Process, set_start_method
from pathlib import Path

from absl import logging

from selfplay import config
from selfplay.training.selfplay import run_selfplay


def _run_dir(cfg: config.ExperimentConfig) -> Path:
    """Full output directory for a single experiment run."""
    return Path(cfg.save_dir) / cfg.selfplay.strategy


def _run_one(batch_config: config.ExperimentBatchConfig, strategy: config.Strategy, seed: int) -> None:
    cfg = batch_config.build_experiment(strategy, seed)
    run_dir = _run_dir(cfg)

    if (run_dir / "metrics.json").exists():
        logging.info("Skipping %s seed=%d (already complete)", strategy.tag, seed)
        return

    if run_dir.exists():
        logging.info("Cleaning incomplete %s seed=%d", strategy.tag, seed)
        shutil.rmtree(run_dir)

    run_selfplay(cfg)


def train_all(batch_config: config.ExperimentBatchConfig, *, parallel: bool = False) -> None:
    """Train every (strategy, seed) combination in *config*."""
    experiments = [
        (s, seed)
        for s in batch_config.strategies
        for seed in batch_config.seeds
    ]

    mode = "parallel" if parallel else "sequential"
    logging.info("Launching %d experiments (%s)", len(experiments), mode)
    for s, seed in experiments:
        logging.info("  %s seed=%d", s.tag, seed)

    if parallel:
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        processes = []
        for strategy, seed in experiments:
            p = Process(
                target=_run_one,
                args=(batch_config, strategy, seed),
                daemon=False,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        for strategy, seed in experiments:
            logging.info("--- %s seed=%d ---", strategy.tag, seed)
            _run_one(batch_config, strategy, seed)

    logging.info("Training complete.")
