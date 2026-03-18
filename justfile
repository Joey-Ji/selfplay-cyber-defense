# Self-play experiment runner.
#
# Override the config for any recipe:
#   just config=experiments/quick.toml train-all
#
# Pass extra CLI flags after the recipe name:
#   just train-all --parallel

set shell := ["bash", "-euo", "pipefail", "-c"]

# Experiment config used by batch recipes (change to run a different set).
config := "experiments/paper.toml"


# List available recipes
default:
    @just --list --unsorted


# Train one self-play agent pair
train *ARGS:
    uv run python scripts/train.py {{ARGS}}

# Evaluate Red vs Blue agents
evaluate *ARGS:
    uv run python scripts/evaluate.py {{ARGS}}


# Train all strategies × seeds
train-all *ARGS:
    uv run python scripts/train_all.py --config {{config}} {{ARGS}}

# Cross-play evaluation
crossplay *ARGS:
    uv run python scripts/eval_crossplay.py --config {{config}} {{ARGS}}

# Exploitability curves
exploitability *ARGS:
    uv run python scripts/eval_exploitability.py --config {{config}} {{ARGS}}

# Comparative reward plots
plot *ARGS:
    uv run python scripts/plot_results.py --config {{config}} {{ARGS}}

# Summary statistics
analyze *ARGS:
    uv run python scripts/analyze.py --config {{config}} {{ARGS}}


# Collect B-line demonstrations
collect-demos *ARGS:
    uv run python scripts/collect_demos.py {{ARGS}}

# Pre-train Red via behavioral cloning
pretrain-red *ARGS:
    uv run python scripts/pretrain_red.py {{ARGS}}


# Full paper-reproduction pipeline (parallel)
pipeline:
    uv run python scripts/train_all.py  --config {{config}} --parallel
    uv run python scripts/eval_crossplay.py --config {{config}}
    uv run python scripts/eval_exploitability.py --config {{config}}
    uv run python scripts/plot_results.py --config {{config}}
    uv run python scripts/analyze.py --config {{config}}

# Quick smoke test (minutes, not hours)
smoke:
    uv run python scripts/train_all.py  --config experiments/quick.toml
    uv run python scripts/eval_crossplay.py --config experiments/quick.toml
    uv run python scripts/plot_results.py --config experiments/quick.toml


# --- Anchor self-play sweep ---

# Train one strategy/seed from a batch config
train-one config seed strategy_index="0":
    uv run python scripts/run_one_from_batch.py --config {{config}} --seed {{seed}} --strategy_index {{strategy_index}}

# Run the full anchor research pipeline (train + eval + summarize)
anchor-sweep:
    bash scripts/run_anchor_research.sh

# Generate anchor figures (ICML format, SVG + PNG)
anchor-plot:
    uv run python scripts/plot_anchor_results_with_errorbars.py

# Run 5-seed significance analysis for anchor_p05_equal
anchor-significance:
    uv run python scripts/analyze_anchor_5seeds.py


# --- Poster ---

# Compile the poster (requires typst)
poster:
    typst compile docs/poster/poster.typ docs/poster/poster.pdf

# Watch poster for live preview
poster-watch:
    typst watch docs/poster/poster.typ docs/poster/poster.pdf


# --- Final Report ---

# Compile the final report (requires latexmk + pdflatex)
report:
    cd docs/final_report && latexmk -pdf -interaction=nonstopmode final.tex

# Watch final report for live recompilation
report-watch:
    cd docs/final_report && latexmk -pdf -pvc -interaction=nonstopmode final.tex

# Clean LaTeX build artifacts
report-clean:
    cd docs/final_report && latexmk -C final.tex


# --- Tests ---

# Run test suite
test:
    uv run pytest tests/ -q
