#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task2 正式训练入口。
#
# 本文件用于启动 Task2 多地形运动任务的 Ubuntu 正式训练流程。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和运行环境；
#   2. 调用 Task2 的 Python 训练入口 task2_train.py；
#   3. 保留 Task1 checkpoint warm-start 的可选入口；
#   4. 使用适合单机 Ubuntu GPU 的默认训练参数。
#
# 参数:
#   $1 可选，Task1 checkpoint 路径，用于 --pretrained-task1。
#
# 本脚本调用:
#   src/go2_rl/tasks/task2/task2_train.py
#
# 使用方式:
#   bash scripts/ubuntu/train_task2.sh
#   bash scripts/ubuntu/train_task2.sh /path/to/go2_task1_model.pt
#
# Unitree Go2 Scripts: Ubuntu Task2 formal training entry.
#
# This file launches the Ubuntu formal training pipeline for Task2 multi-terrain locomotion.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and runtime environment;
#   2. Call the Task2 Python training entry task2_train.py;
#   3. Keep an optional Task1 checkpoint warm-start entry;
#   4. Use default training parameters suitable for a single Ubuntu GPU machine.
#
# Arguments:
#   $1 optional, Task1 checkpoint path for --pretrained-task1.
#
# This script calls:
#   src/go2_rl/tasks/task2/task2_train.py
#
# Usage:
#   bash scripts/ubuntu/train_task2.sh
#   bash scripts/ubuntu/train_task2.sh /path/to/go2_task1_model.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task2 formal training"

go2_check_python_stack --isaaclab --skrl

TASK1_CKPT="${1:-}"
EXTRA_ARGS=()
if [ -n "${TASK1_CKPT}" ]; then
    EXTRA_ARGS+=(--pretrained-task1 "${TASK1_CKPT}")
fi

python src/go2_rl/tasks/task2/task2_train.py \
    --num-envs 512 \
    --total-env-steps 600000000 \
    --rollouts 64 \
    --learning-epochs 5 \
    --mini-batches 8 \
    --lr 5e-5 \
    --min-lr 2e-5 \
    --max-lr 1.2e-4 \
    --gamma 0.995 \
    --gae-lambda 0.95 \
    --kl-threshold 0.015 \
    --entropy-coef 0.003 \
    --value-coef 2.0 \
    --init-log-std -1.25 \
    --pretrained-log-std -1.65 \
    --summary-interval 10 \
    --tb-log-interval-steps 50 \
    --skrl-write-interval 1000000 \
    --skrl-checkpoint-interval 0 \
    --save-freq-env-steps 20000000 \
    --headless \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}"
