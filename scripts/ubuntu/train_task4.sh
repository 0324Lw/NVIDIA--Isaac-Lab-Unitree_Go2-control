#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task4 正式训练入口。
#
# 本文件用于启动 Task4 Sim2Real / RMA teacher 任务的 Ubuntu 正式训练流程。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和运行环境；
#   2. 调用 Task4 的 Python 训练入口 task4_train.py；
#   3. 保留 Task2 / Task1 / Task3 checkpoint warm-start 的可选入口；
#   4. 使用当前稳定 Task4 维度: actor history = 240，privileged obs = 25，teacher obs = 265。
#
# 参数:
#   $1 可选，Task2 checkpoint 路径，用于 --pretrained-task2；
#   $2 可选，Task1 checkpoint 路径，用于 --pretrained-task1；
#   $3 可选，Task3 checkpoint 路径，用于 --pretrained-task3。
#
# 本脚本调用:
#   src/go2_rl/tasks/task4/task4_train.py
#
# 使用方式:
#   bash scripts/ubuntu/train_task4.sh
#   bash scripts/ubuntu/train_task4.sh /path/to/go2_task2_model.pt /path/to/go2_task1_model.pt /path/to/go2_task3_model.pt
#
# Unitree Go2 Scripts: Ubuntu Task4 formal training entry.
#
# This file launches the Ubuntu formal training pipeline for the Task4 Sim2Real / RMA teacher task.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and runtime environment;
#   2. Call the Task4 Python training entry task4_train.py;
#   3. Keep optional Task2 / Task1 / Task3 checkpoint warm-start entries;
#   4. Use the current stable Task4 dimensions: actor history = 240, privileged obs = 25, teacher obs = 265.
#
# Arguments:
#   $1 optional, Task2 checkpoint path for --pretrained-task2;
#   $2 optional, Task1 checkpoint path for --pretrained-task1;
#   $3 optional, Task3 checkpoint path for --pretrained-task3.
#
# This script calls:
#   src/go2_rl/tasks/task4/task4_train.py
#
# Usage:
#   bash scripts/ubuntu/train_task4.sh
#   bash scripts/ubuntu/train_task4.sh /path/to/go2_task2_model.pt /path/to/go2_task1_model.pt /path/to/go2_task3_model.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task4 formal training"

go2_check_python_stack --isaaclab --skrl

TASK2_CKPT="${1:-}"
TASK1_CKPT="${2:-}"
TASK3_CKPT="${3:-}"
EXTRA_ARGS=()

if [ -n "${TASK2_CKPT}" ]; then
    EXTRA_ARGS+=(--pretrained-task2 "${TASK2_CKPT}")
fi

if [ -n "${TASK1_CKPT}" ]; then
    EXTRA_ARGS+=(--pretrained-task1 "${TASK1_CKPT}")
fi

if [ -n "${TASK3_CKPT}" ]; then
    EXTRA_ARGS+=(--pretrained-task3 "${TASK3_CKPT}")
fi

python src/go2_rl/tasks/task4/task4_train.py \
    --num-envs 512 \
    --total-env-steps 400000000 \
    --rollouts 64 \
    --learning-epochs 5 \
    --mini-batches 8 \
    --lr 3e-5 \
    --min-lr 2e-5 \
    --max-lr 7e-5 \
    --gamma 0.995 \
    --gae-lambda 0.95 \
    --kl-threshold 0.015 \
    --entropy-coef 0.0025 \
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
