#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "Go2 Task3 skrl PPO smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)
import torch
print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda:", torch.cuda.is_available())
import isaaclab
print("[CHECK] isaaclab: ok")
import skrl
print("[CHECK] skrl:", getattr(skrl, "__version__", "unknown"))
PY

python src/go2_rl/tasks/task3/task3_train.py \
  --num-envs 32 \
  --total-env-steps 65536 \
  --rollouts 32 \
  --learning-epochs 3 \
  --mini-batches 4 \
  --lr 5e-5 \
  --min-lr 2e-5 \
  --max-lr 1.2e-4 \
  --summary-interval 1 \
  --tb-log-interval-steps 20 \
  --skrl-write-interval 1000000 \
  --skrl-checkpoint-interval 0 \
  --save-freq-env-steps 65536 \
  --headless \
  --device cuda:0
