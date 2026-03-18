"""Summarize anchored-self-play evals against the pure delta=0.7 baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor-dir", required=True, help="Dir with anchor_eval_seed*.json"
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Dir with baseline_delta07_eval_seed*.json or eval_seed*.json",
    )
    parser.add_argument(
        "--anchor-analysis", required=True, help="Anchored analysis.json"
    )
    parser.add_argument(
        "--baseline-analysis", required=True, help="Baseline analysis.json"
    )
    parser.add_argument(
        "--output", required=True, help="Where to save the summary JSON"
    )
    return parser.parse_args()


def infer_opponent_name(spec: str) -> str:
    if spec in {"red_bline", "red_meander"}:
        return spec
    if "vanilla_seed" in spec:
        return "learned_vanilla"
    if "pfsp_seed" in spec:
        return "learned_pfsp"
    if "delta_uniform_d0.3" in spec:
        return "learned_delta_uniform_d0.3"
    if "delta_uniform_d0.5" in spec:
        return "learned_delta_uniform_d0.5"
    if "delta_uniform_d0.7" in spec:
        return "learned_delta_uniform_d0.7"
    return Path(spec).stem


def load_eval(path: Path) -> dict:
    data = json.loads(path.read_text())
    matchups = {}
    for row in data["matchups"]:
        name = infer_opponent_name(row["opponent_spec"])
        matchups[name] = row
    return {
        "average_red_reward_eval_set": data["average_red_reward_eval_set"],
        "matchups": matchups,
        "action_snapshot": data.get("action_snapshot"),
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(statistics.stdev(values))


def load_eval_series(directory: Path, stem_prefix: str) -> dict[int, dict]:
    rows = {}
    for path in sorted(directory.glob(f"{stem_prefix}_seed*.json")):
        seed = int(path.stem.rsplit("seed", 1)[1])
        rows[seed] = load_eval(path)
    return rows


def summarize_deltas(
    anchor_series: dict[int, dict], baseline_series: dict[int, dict]
) -> dict:
    seeds = sorted(set(anchor_series) & set(baseline_series))
    opponent_names = sorted(
        {
            name
            for seed in seeds
            for name in set(anchor_series[seed]["matchups"])
            | set(baseline_series[seed]["matchups"])
        }
    )

    per_seed = []
    opponent_aggregate: dict[str, dict[str, float]] = {}

    for name in opponent_names:
        anchor_values = []
        baseline_values = []
        delta_values = []
        for seed in seeds:
            anchor_row = anchor_series[seed]["matchups"].get(name)
            baseline_row = baseline_series[seed]["matchups"].get(name)
            if anchor_row is None or baseline_row is None:
                continue
            anchor_values.append(float(anchor_row["red_reward_mean"]))
            baseline_values.append(float(baseline_row["red_reward_mean"]))
            delta_values.append(
                float(anchor_row["red_reward_mean"])
                - float(baseline_row["red_reward_mean"])
            )
        if anchor_values:
            opponent_aggregate[name] = {
                "anchor_mean": mean(anchor_values),
                "anchor_std": std(anchor_values),
                "baseline_mean": mean(baseline_values),
                "baseline_std": std(baseline_values),
                "delta_mean": mean(delta_values),
                "delta_std": std(delta_values),
                "n_seeds": len(anchor_values),
            }

    for seed in seeds:
        anchor = anchor_series[seed]
        baseline = baseline_series[seed]
        row = {
            "seed": seed,
            "average_red_reward_eval_set_anchor": anchor["average_red_reward_eval_set"],
            "average_red_reward_eval_set_baseline": baseline[
                "average_red_reward_eval_set"
            ],
            "average_red_reward_eval_set_delta": (
                anchor["average_red_reward_eval_set"]
                - baseline["average_red_reward_eval_set"]
            ),
            "opponents": {},
        }
        for name in opponent_names:
            anchor_row = anchor["matchups"].get(name)
            baseline_row = baseline["matchups"].get(name)
            if anchor_row is None or baseline_row is None:
                continue
            row["opponents"][name] = {
                "anchor_red_reward_mean": anchor_row["red_reward_mean"],
                "baseline_red_reward_mean": baseline_row["red_reward_mean"],
                "delta_red_reward_mean": (
                    anchor_row["red_reward_mean"] - baseline_row["red_reward_mean"]
                ),
                "anchor_red_win_rate": anchor_row["red_win_rate"],
                "baseline_red_win_rate": baseline_row["red_win_rate"],
            }
        per_seed.append(row)

    anchor_avgs = [anchor_series[seed]["average_red_reward_eval_set"] for seed in seeds]
    baseline_avgs = [
        baseline_series[seed]["average_red_reward_eval_set"] for seed in seeds
    ]
    delta_avgs = [a - b for a, b in zip(anchor_avgs, baseline_avgs)]
    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "opponent_aggregate": opponent_aggregate,
        "average_red_reward_eval_set": {
            "anchor_mean": mean(anchor_avgs),
            "anchor_std": std(anchor_avgs),
            "baseline_mean": mean(baseline_avgs),
            "baseline_std": std(baseline_avgs),
            "delta_mean": mean(delta_avgs),
            "delta_std": std(delta_avgs),
            "n_seeds": len(seeds),
        },
    }


def main() -> None:
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)
    baseline_dir = Path(args.baseline_dir)

    anchor_series = load_eval_series(anchor_dir, "anchor_eval")
    baseline_series = load_eval_series(baseline_dir, "eval")
    if not baseline_series:
        baseline_series = load_eval_series(baseline_dir, "baseline_delta07_eval")

    summary = {
        "anchor_dir": str(anchor_dir),
        "baseline_dir": str(baseline_dir),
        "anchor_analysis": json.loads(Path(args.anchor_analysis).read_text()),
        "baseline_analysis": json.loads(Path(args.baseline_analysis).read_text()),
        "eval_comparison": summarize_deltas(anchor_series, baseline_series),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    avg = summary["eval_comparison"]["average_red_reward_eval_set"]
    print(f"Saved summary to {out_path}")
    print(
        "Eval-set average Red reward "
        f"anchor={avg['anchor_mean']:.2f} baseline={avg['baseline_mean']:.2f} "
        f"delta={avg['delta_mean']:.2f}"
    )
    for name, row in summary["eval_comparison"]["opponent_aggregate"].items():
        print(
            f"{name:28s} "
            f"anchor={row['anchor_mean']:.2f} baseline={row['baseline_mean']:.2f} "
            f"delta={row['delta_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
