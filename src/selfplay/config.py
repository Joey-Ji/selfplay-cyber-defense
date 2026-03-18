"""Configuration dataclasses and experiment config loader.

Single-run configs (:class:`ExperimentConfig`) are used by ``train.py``.
Batch specs (:class:`ExperimentBatchConfig`) are loaded from TOML files and used
by all batch scripts (``train_all.py``, ``eval_crossplay.py``, etc.).

Usage::

    from selfplay import config

    config = config.load("experiments/paper.toml")
    for strategy in config.strategies:
        for seed in config.seeds:
            cfg = config.build_experiment(strategy, seed)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

VALID_STRATEGIES = {"vanilla", "delta_uniform", "pfsp"}


@dataclass
class PPOConfig:
    learning_rate: float = 0.002
    gamma: float = 0.99
    clip_range: float = 0.2
    n_epochs: int = 6
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                f"n_steps ({self.n_steps}) must be divisible by "
                f"batch_size ({self.batch_size})"
            )


@dataclass
class GRPOConfig:
    learning_rate: float = 0.001
    gamma: float = 0.99
    clip_range: float = 0.2
    n_epochs: int = 4
    group_size: int = 8
    batch_size: int = 64
    ent_coef: float = 0.01
    kl_coef: float = 0.1
    max_grad_norm: float = 0.5
    ref_update_freq: int = 10

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be non-negative, got {self.kl_coef}")


@dataclass
class SelfPlayConfig:
    strategy: str = "vanilla"
    delta: float = 0.5
    checkpoint_freq: int = 1
    max_pool_size: int = 200
    rounds_per_side: int = 10
    total_rounds: int = 500
    max_steps: int = 100
    num_envs: int = 1
    red_pretrain_steps: int = 0
    pfsp_eval_episodes: int = 50
    pfsp_p: float = 1.0
    blue_pretrain_path: str = ""
    red_pretrain_path: str = ""
    blue_scripted_opponent_prob: float = 0.0
    blue_scripted_opponents: list[str] = field(
        default_factory=lambda: ["red_bline", "red_meander"]
    )
    blue_scripted_opponent_weights: list[float] = field(
        default_factory=lambda: [0.5, 0.5]
    )

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}. "
                f"Must be one of {sorted(VALID_STRATEGIES)}"
            )
        if not 0.0 <= self.delta <= 1.0:
            raise ValueError(f"delta must be in [0, 1], got {self.delta}")
        if len(self.blue_scripted_opponent_weights) != len(
            self.blue_scripted_opponents
        ):
            raise ValueError(
                f"blue_scripted_opponent_weights length "
                f"({len(self.blue_scripted_opponent_weights)}) must match "
                f"blue_scripted_opponents length "
                f"({len(self.blue_scripted_opponents)})"
            )

@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "mini-cage-selfplay"
    entity: str = "YTRewards"
    group: str = ""


@dataclass
class ExperimentConfig:
    """Single experiment: one (strategy, seed) run."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    algorithm: str = "ppo"  # "ppo" or "grpo"
    seed: int = 42
    save_dir: str = "out/models"


@dataclass(frozen=True)
class Strategy:
    """One opponent-sampling strategy (one ``[[strategies]]`` block in TOML)."""

    name: str
    delta: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.name!r}. "
                f"Must be one of {sorted(VALID_STRATEGIES)}"
            )
        if not 0.0 <= self.delta <= 1.0:
            raise ValueError(f"delta must be in [0, 1], got {self.delta}")

    @property
    def tag(self) -> str:
        """Directory-safe label, e.g. ``delta_uniform_d0.3``."""
        if self.name == "delta_uniform":
            return f"delta_uniform_d{self.delta}"
        return self.name


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation hyperparameters shared across all strategies."""

    br_steps: int = 200_000
    crossplay_episodes: int = 200
    crossplay_checkpoints: int = 10
    exploit_checkpoints: int = 5


@dataclass
class ExperimentBatchConfig:
    """Everything needed to run a batch of experiments.

    Loaded from a TOML file via :func:`load`.  Provides typed access to
    strategies, seeds, hyperparameters, and output paths — plus a
    convenience method to build a single-run :class:`ExperimentConfig`.
    """

    strategies: list[Strategy]
    seeds: list[int]
    ppo: PPOConfig
    grpo: GRPOConfig
    selfplay: SelfPlayConfig
    wandb: WandbConfig
    eval: EvalConfig
    algorithm: str = "ppo"  # "ppo" or "grpo"
    save_dir: str = "out/models"
    results_dir: str = "out/results"

    def build_experiment(
        self,
        strategy: Strategy,
        seed: int,
    ) -> ExperimentConfig:
        """Construct an :class:`ExperimentConfig` for one (strategy, seed) run."""
        return ExperimentConfig(
            ppo=self.ppo,
            grpo=self.grpo,
            selfplay=replace(
                self.selfplay,
                strategy=strategy.name,
                delta=strategy.delta,
            ),
            wandb=replace(self.wandb, group=f"selfplay_{strategy.tag}"),
            algorithm=self.algorithm,
            seed=seed,
            save_dir=str(Path(self.save_dir) / f"{strategy.tag}_seed{seed}"),
        )


def load(path: str | Path) -> ExperimentBatchConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    raw_strategies = raw.get("strategies", [])
    if not raw_strategies:
        raise ValueError(f"{path}: must define at least one [[strategies]] entry")

    seeds = raw.get("seeds", [42])
    if not seeds or not all(isinstance(s, int) and s > 0 for s in seeds):
        raise ValueError(f"{path}: seeds must be a non-empty list of positive integers")

    algorithm = raw.get("algorithm", "ppo")
    if algorithm not in ("ppo", "grpo"):
        raise ValueError(f"{path}: algorithm must be 'ppo' or 'grpo', got {algorithm!r}")

    return ExperimentBatchConfig(
        strategies=[Strategy(**s) for s in raw_strategies],
        seeds=seeds,
        ppo=PPOConfig(**raw.get("ppo", {})),
        grpo=GRPOConfig(**raw.get("grpo", {})),
        selfplay=SelfPlayConfig(**raw.get("selfplay", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        algorithm=algorithm,
        save_dir=raw.get("save_dir", "out/models"),
        results_dir=raw.get("results_dir", "out/results"),
    )
