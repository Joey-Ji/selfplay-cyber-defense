"""Evaluate a cross-condition transfer matrix for representative Blue agents."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import app, flags

from selfplay.eval.evaluate import evaluate, load_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("transfer_json", "out_vm/results/transfer_eval.json", "Transfer eval JSON.")
flags.DEFINE_string("output_json", "out_vm/results/cross_condition_matrix.json", "Matrix JSON output.")
flags.DEFINE_string("output_png", "out_vm/results/cross_condition_matrix.png", "Matrix heatmap output.")
flags.DEFINE_integer("episodes", 200, "Number of evaluation episodes per matchup.")
flags.DEFINE_integer("max_steps", 100, "Max environment steps per episode.")

BLUE_LABELS = {
    "no_bc_no_curriculum": "No BC",
    "bc_only": "BC Only",
    "bc_curriculum_p03": "BC + Curr p=0.3",
    "bc_curriculum_p05": "BC + Curr p=0.5",
    "delta_uniform_d0.5": "delta_uniform",
    "pfsp": "PFSP",
}


def choose_blue_representatives(transfer_path: Path) -> list[dict]:
    with transfer_path.open() as f:
        runs = json.load(f)["runs"]

    best_by_condition = {}
    for run in runs:
        score = (
            run["transfer"]["bline"]["red_reward_mean"]
            + run["transfer"]["meander"]["red_reward_mean"]
        ) / 2.0
        current = best_by_condition.get(run["condition"])
        if current is None or score < current["selection_score"]:
            best_by_condition[run["condition"]] = {
                "condition": run["condition"],
                "seed": run["seed"],
                "blue_checkpoint": run["blue_checkpoint"],
                "selection_score": score,
            }

    order = [
        "no_bc_no_curriculum",
        "bc_only",
        "bc_curriculum_p03",
        "bc_curriculum_p05",
        "delta_uniform_d0.5",
        "pfsp",
    ]
    return [best_by_condition[key] for key in order if key in best_by_condition]


def build_rows(root: Path) -> list[dict]:
    return [
        {"label": "B-line", "kind": "scripted", "agent": "bline"},
        {"label": "Meander", "kind": "scripted", "agent": "meander"},
        {
            "label": "Learned Red (No BC s42)",
            "kind": "learned",
            "agent": str(root / "models_ablation_no_bc_no_curriculum" / "vanilla_seed42" / "vanilla" / "red" / "latest.pt"),
        },
        {
            "label": "Learned Red (No BC s123)",
            "kind": "learned",
            "agent": str(root / "models_ablation_no_bc_no_curriculum" / "vanilla_seed123" / "vanilla" / "red" / "latest.pt"),
        },
    ]


def plot_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
    )
    ax.set_title("Cross-Condition Transfer Matrix")
    ax.set_xlabel("Representative Blue Agent")
    ax.set_ylabel("Red Opponent")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(_):
    transfer_path = Path(FLAGS.transfer_json)
    root = transfer_path.parent.parent
    blue_reps = choose_blue_representatives(transfer_path)
    rows = build_rows(root)

    row_labels = [row["label"] for row in rows]
    col_labels = [BLUE_LABELS.get(rep["condition"], rep["condition"]) for rep in blue_reps]
    matrix = np.zeros((len(rows), len(blue_reps)))

    for row_idx, row in enumerate(rows):
        red_agent = load_agent(row["agent"], "red")
        for col_idx, rep in enumerate(blue_reps):
            blue_agent = load_agent(rep["blue_checkpoint"], "blue")
            result = evaluate(
                red_agent,
                blue_agent,
                num_episodes=FLAGS.episodes,
                max_steps=FLAGS.max_steps,
            )
            matrix[row_idx, col_idx] = result["red_reward_mean"]

    output_json = Path(FLAGS.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(
            {
                "episodes": FLAGS.episodes,
                "max_steps": FLAGS.max_steps,
                "rows": rows,
                "columns": blue_reps,
                "matrix": matrix.tolist(),
            },
            f,
            indent=2,
        )

    plot_matrix(matrix, row_labels, col_labels, Path(FLAGS.output_png))
    print(f"Saved matrix JSON to {output_json}")
    print(f"Saved matrix heatmap to {FLAGS.output_png}")


if __name__ == "__main__":
    app.run(main)
