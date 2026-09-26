#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SS_ICRL_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${SS_ICRL_CONFIG:-$ROOT/configs/ss_icrl.yaml}"
RESULTS_ROOT="${SS_ICRL_RESULTS_DIR:-$ROOT/results/ss_icrl}"
CSV_PATH="${SS_ICRL_CSV_PATH:-$RESULTS_ROOT/metrics.csv}"

export HF_HOME="${HF_HOME:-$ROOT/models}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export SS_ICRL_CONFIG="$CONFIG"

read_cfg () {
  "$PYTHON_BIN" - "$CONFIG" "$1" <<'PY'
import sys
import yaml

config_path, key = sys.argv[1], sys.argv[2]
with open(config_path, encoding="utf-8") as fp:
    config = yaml.safe_load(fp) or {}
value = config.get(key)
if isinstance(value, list):
    print(" ".join(str(item) for item in value))
elif value is not None:
    print(value)
PY
}

single_run () {
  local MODEL="$1"
  local TASK="$2"
  shift 2

  local MODEL_PATH="$MODEL"
  local TASK_PATH="$TASK"
  local MODEL_TAG="${MODEL##*/}"
  local TASK_TAG="${TASK##*/}"
  local OUTDIR="$RESULTS_ROOT/${TASK_TAG}_${MODEL_TAG}"

  if [[ -d "$ROOT/models/$MODEL" ]]; then
    MODEL_PATH="$ROOT/models/$MODEL"
  fi
  if [[ -d "$ROOT/data/$TASK" ]]; then
    TASK_PATH="$ROOT/data/$TASK"
  fi

  echo "[RUN] SS-ICRL | model=$MODEL | task=$TASK"
  echo "[CFG] $CONFIG"
  "$PYTHON_BIN" -m icrl.ss_icrl_runner \
    --model_path "$MODEL_PATH" \
    --task_dir "$TASK_PATH" \
    --output_dir "$OUTDIR" \
    --csv_path "$CSV_PATH" \
    "$@"
}

if [[ "${1:-}" == "all" ]]; then
  read -r -a MODELS <<< "$(read_cfg models)"
  read -r -a DATASETS <<< "$(read_cfg datasets)"
  shift
  [[ ${#MODELS[@]} -gt 0 ]] || { echo "No models configured in $CONFIG"; exit 1; }
  [[ ${#DATASETS[@]} -gt 0 ]] || { echo "No datasets configured in $CONFIG"; exit 1; }
  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      single_run "$model" "$dataset" "$@"
    done
  done
else
  [[ $# -ge 2 ]] || {
    echo "Usage: $0 MODEL DATASET [runner args...]"
    echo "   or: $0 all [runner args...]"
    exit 1
  }
  MODEL="$1"
  TASK="$2"
  shift 2
  single_run "$MODEL" "$TASK" "$@"
fi
