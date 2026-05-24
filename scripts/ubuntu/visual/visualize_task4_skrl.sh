#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ubuntu/visual/visualize_task4_skrl.sh /path/to/task4_model.pt [start_k]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export DISPLAY="${DISPLAY:-:0}"

CKPT="$1"
START_K="${2:-1.0}"

python src/go2_rl/tasks/task4/task4_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 1 \
  --steps 2000 \
  --start-k "${START_K}" \
  --print-interval 50 \
  --visualize \
  --device cuda:0
