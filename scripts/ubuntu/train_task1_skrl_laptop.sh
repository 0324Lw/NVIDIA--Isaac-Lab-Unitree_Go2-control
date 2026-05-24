#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python src/go2_rl/tasks/task1/task1_train.py \
  --num-envs 512 \
  --total-env-steps 350000000 \
  --rollouts 64 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 1e-4 \
  --min-lr 2e-5 \
  --max-lr 3e-4 \
  --summary-interval 10 \
  --tb-log-interval-steps 50 \
  --skrl-write-interval 1000000 \
  --skrl-checkpoint-interval 0 \
  --save-freq-env-steps 20000000 \
  --headless \
  --device cuda:0
