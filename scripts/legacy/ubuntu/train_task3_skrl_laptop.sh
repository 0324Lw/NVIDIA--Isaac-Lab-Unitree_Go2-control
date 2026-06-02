#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

TASK2_CKPT="${1:-}"
TASK1_CKPT="${2:-}"

EXTRA_ARGS=()

if [ -n "${TASK2_CKPT}" ]; then
  EXTRA_ARGS+=(--pretrained-task2 "${TASK2_CKPT}")
fi

if [ -n "${TASK1_CKPT}" ]; then
  EXTRA_ARGS+=(--pretrained-task1 "${TASK1_CKPT}")
fi

python src/go2_rl/tasks/task3/task3_train.py \
  --num-envs 512 \
  --total-env-steps 800000000 \
  --rollouts 64 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 5e-5 \
  --min-lr 2e-5 \
  --max-lr 1.2e-4 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --kl-threshold 0.015 \
  --entropy-coef 0.004 \
  --value-coef 2.0 \
  --init-log-std -1.35 \
  --pretrained-log-std -1.75 \
  --summary-interval 10 \
  --tb-log-interval-steps 50 \
  --skrl-write-interval 1000000 \
  --skrl-checkpoint-interval 0 \
  --save-freq-env-steps 20000000 \
  --headless \
  --device cuda:0 \
  "${EXTRA_ARGS[@]}"
