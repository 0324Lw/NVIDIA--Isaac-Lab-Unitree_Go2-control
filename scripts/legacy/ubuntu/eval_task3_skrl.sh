#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ubuntu/eval_task3_skrl.sh /path/to/go2_task3_model.pt [start_k]"
  echo "Example:"
  echo "  bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/go2_task3_model.pt 0.12"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-1.0}"

python src/go2_rl/tasks/task3/task3_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 16 \
  --steps 3000 \
  --start-k "${START_K}" \
  --print-interval 100 \
  --headless \
  --device cuda:0
