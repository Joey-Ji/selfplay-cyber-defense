"""5-seed significance analysis for anchor_p05_equal vs baseline.

Combines original 3 seeds (42, 123, 456) with extra seeds (789, 1024).
The extra baseline uses rounds_per_side=5 to match the anchor configs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS_ORIGINAL = [42, 123, 456]
SEEDS_EXTRA = [789, 1024]


def load_original_data() -> list[dict]:
    """Load per-seed data from the original 3-seed summary."""
    summary = json.loads(
        Path("out/results_anchor_p05_equal/summary.json").read_text()
    )
    per_seed = summary["eval_comparison"]["per_seed"]
    results = []
    for s in per_seed:
        results.append({
            "seed": s["seed"],
            "anchor_avg": s["average_red_reward_eval_set_anchor"],
            "baseline_avg": s["average_red_reward_eval_set_baseline"],
            "anchor_bline": s["opponents"]["red_bline"]["anchor_red_reward_mean"],
            "baseline_bline": s["opponents"]["red_bline"]["baseline_red_reward_mean"],
            "anchor_meander": s["opponents"]["red_meander"]["anchor_red_reward_mean"],
            "baseline_meander": s["opponents"]["red_meander"]["baseline_red_reward_mean"],
        })
    return results


def load_extra_seed(seed: int) -> dict:
    """Load a single extra seed's eval data."""
    baseline_path = Path(f"out/results_baseline_extra/eval_seed{seed}.json")
    anchor_path = Path(f"out/results_anchor_p05_equal_extra/anchor_eval_seed{seed}.json")

    if not baseline_path.exists() or not anchor_path.exists():
        raise FileNotFoundError(
            f"Missing eval files for seed {seed}:\n"
            f"  baseline: {baseline_path} (exists={baseline_path.exists()})\n"
            f"  anchor: {anchor_path} (exists={anchor_path.exists()})"
        )

    baseline_data = json.loads(baseline_path.read_text())
    anchor_data = json.loads(anchor_path.read_text())

    def get_opponent(data, opp_name):
        for m in data["matchups"]:
            if m["opponent_spec"] == opp_name:
                return m["red_reward_mean"]
        return None

    return {
        "seed": seed,
        "anchor_avg": anchor_data["average_red_reward_eval_set"],
        "baseline_avg": baseline_data["average_red_reward_eval_set"],
        "anchor_bline": get_opponent(anchor_data, "red_bline"),
        "baseline_bline": get_opponent(baseline_data, "red_bline"),
        "anchor_meander": get_opponent(anchor_data, "red_meander"),
        "baseline_meander": get_opponent(baseline_data, "red_meander"),
    }


def paired_test(anchor_vals, baseline_vals, label=""):
    """Run paired t-test and return results dict."""
    deltas = [a - b for a, b in zip(anchor_vals, baseline_vals)]
    n = len(deltas)
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    se = std_d / math.sqrt(n)
    t_stat, p_val = stats.ttest_rel(anchor_vals, baseline_vals)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean_d, scale=se)
    return {
        "label": label,
        "n": n,
        "mean_delta": mean_d,
        "sample_std": std_d,
        "se": se,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "ci_95_lower": float(ci[0]),
        "ci_95_upper": float(ci[1]),
        "significant_at_05": float(p_val) < 0.05,
        "significant_at_10": float(p_val) < 0.10,
        "deltas": deltas,
    }


def main() -> None:
    # Load original 3 seeds
    original = load_original_data()
    print(f"Loaded {len(original)} original seeds: {[d['seed'] for d in original]}")

    # Load extra seeds
    extra = []
    for seed in SEEDS_EXTRA:
        try:
            extra.append(load_extra_seed(seed))
            print(f"Loaded extra seed {seed}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    all_data = original + extra
    n = len(all_data)
    print(f"\nTotal seeds: {n}")

    if n < 3:
        print("ERROR: Need at least 3 seeds for analysis")
        return

    # Overall eval-set average
    anchor_avgs = [d["anchor_avg"] for d in all_data]
    baseline_avgs = [d["baseline_avg"] for d in all_data]
    overall = paired_test(anchor_avgs, baseline_avgs, "eval_set_average")

    print(f"\n{'='*60}")
    print(f"anchor_p05_equal vs baseline (n={n} seeds)")
    print(f"{'='*60}")
    print(f"\n--- Eval-set Average ---")
    for d in all_data:
        delta = d["anchor_avg"] - d["baseline_avg"]
        print(f"  Seed {d['seed']:>4}: anchor={d['anchor_avg']:.2f}, "
              f"baseline={d['baseline_avg']:.2f}, delta={delta:+.2f}")
    print(f"\n  Mean delta: {overall['mean_delta']:.2f} +/- {overall['sample_std']:.2f}")
    print(f"  Paired t-test: t={overall['t_stat']:.3f}, p={overall['p_value']:.4f}")
    print(f"  95% CI: [{overall['ci_95_lower']:.2f}, {overall['ci_95_upper']:.2f}]")
    print(f"  Significant at alpha=0.05: {overall['significant_at_05']}")
    print(f"  Significant at alpha=0.10: {overall['significant_at_10']}")

    # Per-opponent breakdown
    results = {"overall": overall, "per_opponent": {}}
    for opp_name, key_a, key_b in [
        ("B-line", "anchor_bline", "baseline_bline"),
        ("Meander", "anchor_meander", "baseline_meander"),
    ]:
        a_vals = [d[key_a] for d in all_data if d[key_a] is not None]
        b_vals = [d[key_b] for d in all_data if d[key_b] is not None]
        if len(a_vals) >= 2:
            opp_result = paired_test(a_vals, b_vals, opp_name)
            results["per_opponent"][opp_name] = opp_result
            print(f"\n--- {opp_name} ---")
            for d in all_data:
                if d[key_a] is not None:
                    delta = d[key_a] - d[key_b]
                    print(f"  Seed {d['seed']:>4}: anchor={d[key_a]:.2f}, "
                          f"baseline={d[key_b]:.2f}, delta={delta:+.2f}")
            print(f"  Mean delta: {opp_result['mean_delta']:.1f} +/- {opp_result['sample_std']:.1f}")
            print(f"  p={opp_result['p_value']:.4f}, "
                  f"CI=[{opp_result['ci_95_lower']:.1f}, {opp_result['ci_95_upper']:.1f}]")
            print(f"  Significant at alpha=0.05: {opp_result['significant_at_05']}")

    # Scripted-only average (primary recommended metric)
    scripted_anchor = []
    scripted_baseline = []
    for d in all_data:
        if d["anchor_bline"] is not None and d["anchor_meander"] is not None:
            scripted_anchor.append((d["anchor_bline"] + d["anchor_meander"]) / 2)
            scripted_baseline.append((d["baseline_bline"] + d["baseline_meander"]) / 2)
    if len(scripted_anchor) >= 2:
        scripted_result = paired_test(scripted_anchor, scripted_baseline, "scripted_only_avg")
        results["scripted_only"] = scripted_result
        print(f"\n--- Scripted-only Average (recommended primary metric) ---")
        for i, d in enumerate(all_data):
            print(f"  Seed {d['seed']:>4}: anchor={scripted_anchor[i]:.2f}, "
                  f"baseline={scripted_baseline[i]:.2f}, "
                  f"delta={scripted_anchor[i]-scripted_baseline[i]:+.2f}")
        print(f"  Mean delta: {scripted_result['mean_delta']:.1f} +/- {scripted_result['sample_std']:.1f}")
        print(f"  p={scripted_result['p_value']:.4f}, "
              f"CI=[{scripted_result['ci_95_lower']:.1f}, {scripted_result['ci_95_upper']:.1f}]")
        print(f"  Significant at alpha=0.05: {scripted_result['significant_at_05']}")

    # Save results
    out_path = Path("docs/experiments/anchor/data/significance_5seeds.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON serialization
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    out_path.write_text(json.dumps(clean(results), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
