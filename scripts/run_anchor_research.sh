#!/usr/bin/env bash
set -euo pipefail

cd /Users/xiyan/Development/CybORG_plus_plus

configs=(
  anchor_p05_meander
  anchor_p07_meander
  anchor_p05_equal
)
seeds=(42 123 456)
baseline_dir="out/results_anchor_baseline_delta07"
log_dir="docs/experiments/anchor/logs"
lock_dir="out/locks/anchor_research"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

mkdir -p out/locks
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "$(timestamp) LOCK_HELD exiting lock_dir=${lock_dir}"
  exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

heldout_reds_for_seed() {
  local seed="$1"
  local -a reds=()
  local candidates=(
    "out/models/vanilla_seed${seed}/vanilla/red/latest.zip"
    "out/models/pfsp_seed${seed}/pfsp/red/latest.zip"
    "out/models/delta_uniform_d0.3_seed${seed}/delta_uniform/red/latest.zip"
    "out/models/delta_uniform_d0.5_seed${seed}/delta_uniform/red/latest.zip"
    "out/models/delta_uniform_d0.7_seed${seed}/delta_uniform/red/latest.zip"
  )
  for red in "${candidates[@]}"; do
    if [[ -f "$red" ]]; then
      reds+=(--heldout_red "$red")
    fi
  done
  printf '%s\0' "${reds[@]}"
}

read_heldout_args() {
  local seed="$1"
  HELDOUT_ARGS=()
  while IFS= read -r -d '' arg; do
    HELDOUT_ARGS+=("$arg")
  done < <(heldout_reds_for_seed "$seed")
}

config_rounds() {
  local config="$1"
  python - "$config" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
print(data["selfplay"]["total_rounds"])
PY
}

run_complete() {
  local model_dir="$1"
  local expected_rounds="$2"
  python - "$model_dir" "$expected_rounds" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
metrics = model_dir / "metrics.json"
blue = model_dir / "blue" / "latest.zip"
red = model_dir / "red" / "latest.zip"
if not (metrics.is_file() and blue.is_file() and red.is_file()):
    raise SystemExit(1)
rows = json.loads(metrics.read_text())
raise SystemExit(0 if len(rows) == expected else 1)
PY
}

ensure_baseline_eval() {
  local seed="$1"
  local output="${baseline_dir}/eval_seed${seed}.json"
  local log="${log_dir}/baseline_delta07_seed${seed}_eval.log"

  if [[ -f "$output" ]]; then
    echo "$(timestamp) SKIP baseline eval seed=${seed} output=${output}"
    return
  fi

  read_heldout_args "$seed"
  echo "$(timestamp) START baseline eval seed=${seed} output=${output}"
  uv run python scripts/eval_anchor_pilot.py \
    --blue "out/models/delta_uniform_d0.7_seed${seed}/delta_uniform/blue/latest.zip" \
    "${HELDOUT_ARGS[@]}" \
    --output "$output" \
    > "$log" 2>&1
  echo "$(timestamp) DONE baseline eval seed=${seed} output=${output}"
}

mkdir -p "$baseline_dir" "$log_dir"

for seed in "${seeds[@]}"; do
  ensure_baseline_eval "$seed"
done

for name in "${configs[@]}"; do
  config="experiments/${name}.toml"
  results_dir="out/results_${name}"
  models_dir="out/models_${name}"
  expected_rounds="$(config_rounds "$config")"
  mkdir -p "$results_dir"

  echo "$(timestamp) START train ${name}"
  pids=()
  for seed in "${seeds[@]}"; do
    run_dir="${models_dir}/delta_uniform_d0.7_seed${seed}/delta_uniform"
    log="${log_dir}/${name}_seed${seed}_train.log"
    if run_complete "$run_dir" "$expected_rounds"; then
      echo "$(timestamp) SKIP train ${name} seed=${seed} run_dir=${run_dir}"
      continue
    fi
    echo "$(timestamp) LAUNCH train ${name} seed=${seed} log=${log}"
    uv run python scripts/run_one_from_batch.py --config "$config" --seed "$seed" > "$log" 2>&1 &
    pids+=("$!")
    sleep 1
  done

  for pid in "${pids[@]:-}"; do
    wait "$pid"
  done
  echo "$(timestamp) DONE train ${name}"

  echo "$(timestamp) START analyze ${name}"
  uv run python scripts/analyze.py --config "$config" --output "${results_dir}/analysis.json" \
    > "${log_dir}/${name}_analyze.log" 2>&1
  echo "$(timestamp) DONE analyze ${name}"

  for seed in "${seeds[@]}"; do
    blue="${models_dir}/delta_uniform_d0.7_seed${seed}/delta_uniform/blue/latest.zip"
    output="${results_dir}/anchor_eval_seed${seed}.json"
    log="${log_dir}/${name}_seed${seed}_eval.log"
    if [[ -f "$output" ]]; then
      echo "$(timestamp) SKIP eval ${name} seed=${seed} output=${output}"
      continue
    fi
    read_heldout_args "$seed"
    echo "$(timestamp) START eval ${name} seed=${seed}"
    uv run python scripts/eval_anchor_pilot.py \
      --blue "$blue" \
      "${HELDOUT_ARGS[@]}" \
      --output "$output" \
      > "$log" 2>&1
    echo "$(timestamp) DONE eval ${name} seed=${seed}"
  done

  echo "$(timestamp) START summarize ${name}"
  uv run python scripts/summarize_anchor_results.py \
    --anchor-dir "$results_dir" \
    --baseline-dir "$baseline_dir" \
    --anchor-analysis "${results_dir}/analysis.json" \
    --baseline-analysis "out/results/analysis.json" \
    --output "${results_dir}/summary.json" \
    > "${log_dir}/${name}_summary.log" 2>&1
  echo "$(timestamp) DONE summarize ${name}"
done

echo "$(timestamp) ALL_DONE"
