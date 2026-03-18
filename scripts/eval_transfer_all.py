"""Evaluate all VM-trained Blue agents against scripted Red baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from absl import app, flags

from selfplay.eval.evaluate import evaluate, load_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("root", "out_vm", "Root directory containing VM experiment outputs.")
flags.DEFINE_string(
    "output",
    "out_vm/results/transfer_eval.json",
    "Path to the JSON output file.",
)
flags.DEFINE_integer("episodes", 200, "Number of batched evaluation episodes per matchup.")
flags.DEFINE_integer("max_steps", 100, "Maximum environment steps per episode.")
flags.DEFINE_integer("seed", 42, "Numpy seed for deterministic scripted-agent sampling.")

SCRIPTED_REDS = ("bline", "meander")
RUN_ORDER = (
    "no_bc_no_curriculum",
    "bc_only",
    "bc_curriculum_p03",
    "bc_curriculum_p05",
    "delta_uniform_d0.5",
    "pfsp",
)


@dataclass(frozen=True)
class RunSpec:
    condition: str
    seed: int
    strategy: str
    blue_path: Path
    metrics_path: Path


def _parse_seed(token: str) -> int:
    if not token.startswith("seed"):
        raise ValueError(f"Unexpected seed token: {token}")
    return int(token.removeprefix("seed"))


def discover_runs(root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []

    patterns = {
        "no_bc_no_curriculum": "models_ablation_no_bc_no_curriculum/*/vanilla/blue/latest.pt",
        "bc_only": "models_ablation_bc_only/*/vanilla/blue/latest.pt",
        "bc_curriculum_p05": "models_ablation_bc_curriculum_p05/*/vanilla/blue/latest.pt",
        "bc_curriculum_p03": "models_final/vanilla_seed*/vanilla/blue/latest.pt",
        "delta_uniform_d0.5": "models_final/delta_uniform_d0.5_seed*/delta_uniform/blue/latest.pt",
        "pfsp": "models_final/pfsp_seed*/pfsp/blue/latest.pt",
    }

    for condition, pattern in patterns.items():
        for blue_path in sorted(root.glob(pattern)):
            exp_dir = blue_path.parent.parent
            seed_token = exp_dir.parent.name.rsplit("_", 1)[-1]
            seed = _parse_seed(seed_token)
            metrics_path = exp_dir / "metrics.json"
            runs.append(
                RunSpec(
                    condition=condition,
                    seed=seed,
                    strategy=exp_dir.name,
                    blue_path=blue_path,
                    metrics_path=metrics_path,
                )
            )

    return sorted(runs, key=lambda run: (RUN_ORDER.index(run.condition), run.seed))


def load_selfplay_blue_reward(metrics_path: Path) -> float:
    with metrics_path.open() as f:
        metrics = json.load(f)
    if not metrics:
        raise ValueError(f"No metrics in {metrics_path}")
    return float(metrics[-1]["blue_mean_reward"])


def summarize_condition(condition_runs: list[dict]) -> dict:
    vs_bline = [run["transfer"]["bline"]["red_reward_mean"] for run in condition_runs]
    vs_meander = [run["transfer"]["meander"]["red_reward_mean"] for run in condition_runs]
    blue_selfplay = [run["selfplay_blue_reward"] for run in condition_runs]
    return {
        "n_seeds": len(condition_runs),
        "vs_bline_red_reward_mean": float(np.mean(vs_bline)),
        "vs_bline_red_reward_std": float(np.std(vs_bline)),
        "vs_meander_red_reward_mean": float(np.mean(vs_meander)),
        "vs_meander_red_reward_std": float(np.std(vs_meander)),
        "selfplay_blue_reward_mean": float(np.mean(blue_selfplay)),
        "selfplay_blue_reward_std": float(np.std(blue_selfplay)),
    }


def print_summary_table(summary: dict[str, dict]) -> None:
    header = (
        f"{'Condition':<24} {'vs B-line Red':>16} {'vs Meander Red':>18} "
        f"{'Self-play Blue':>18}"
    )
    print("\nTransfer Summary")
    print(header)
    print("-" * len(header))
    for condition in RUN_ORDER:
        if condition not in summary:
            continue
        row = summary[condition]
        print(
            f"{condition:<24} "
            f"{row['vs_bline_red_reward_mean']:>8.1f} +/- {row['vs_bline_red_reward_std']:<5.1f} "
            f"{row['vs_meander_red_reward_mean']:>8.1f} +/- {row['vs_meander_red_reward_std']:<5.1f} "
            f"{row['selfplay_blue_reward_mean']:>8.1f} +/- {row['selfplay_blue_reward_std']:<5.1f}"
        )


def main(_):
    np.random.seed(FLAGS.seed)

    root = Path(FLAGS.root)
    runs = discover_runs(root)
    if not runs:
        raise FileNotFoundError(f"No Blue checkpoints found under {root}")

    print(f"Discovered {len(runs)} Blue checkpoints under {root}")
    run_results: list[dict] = []
    for idx, run in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] {run.condition} seed={run.seed}")
        blue_agent = load_agent(str(run.blue_path), "blue")
        transfer = {}
        for red_name in SCRIPTED_REDS:
            red_agent = load_agent(red_name, "red")
            transfer[red_name] = evaluate(
                red_agent,
                blue_agent,
                num_episodes=FLAGS.episodes,
                max_steps=FLAGS.max_steps,
            )

        run_results.append(
            {
                "condition": run.condition,
                "seed": run.seed,
                "strategy": run.strategy,
                "blue_checkpoint": str(run.blue_path),
                "metrics_path": str(run.metrics_path),
                "selfplay_blue_reward": load_selfplay_blue_reward(run.metrics_path),
                "transfer": transfer,
            }
        )

    summary = {
        condition: summarize_condition(
            [result for result in run_results if result["condition"] == condition]
        )
        for condition in RUN_ORDER
        if any(result["condition"] == condition for result in run_results)
    }

    output = Path(FLAGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(
            {
                "episodes": FLAGS.episodes,
                "max_steps": FLAGS.max_steps,
                "scripted_reds": list(SCRIPTED_REDS),
                "runs": run_results,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print_summary_table(summary)
    print(f"\nSaved transfer evaluation to {output}")


if __name__ == "__main__":
    app.run(main)
