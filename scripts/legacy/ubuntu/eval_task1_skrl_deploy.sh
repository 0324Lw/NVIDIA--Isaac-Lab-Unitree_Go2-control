#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/ubuntu/eval_task1_skrl_deploy.sh /path/to/go2_task1_skrl_deploy.pt"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python src/go2_rl/tasks/task1/task1_model_test.py \
  --checkpoint "$1" \
  --checkpoint-type deploy \
  --num-envs 16 \
  --steps 2000 \
  --print-interval 100 \
  --headless \
  --device cuda:0
