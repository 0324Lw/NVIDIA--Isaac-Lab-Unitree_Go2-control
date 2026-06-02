#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task1 正式训练入口。
#
# 本文件用于启动 Task1 平地运动任务的 Ubuntu 正式训练流程。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和运行环境；
#   2. 调用 Task1 的 Python 训练入口 task1_train.py；
#   3. 使用适合单机 Ubuntu GPU 的默认训练参数；
#   4. 允许用户通过修改脚本参数或直接调用 Python 入口进行更细粒度配置。
#
# 本脚本调用:
#   src/go2_rl/tasks/task1/task1_train.py
#
# 使用方式:
#   bash scripts/ubuntu/train_task1.sh
#
# Unitree Go2 Scripts: Ubuntu Task1 formal training entry.
#
# This file launches the Ubuntu formal training pipeline for Task1 flat locomotion.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and runtime environment;
#   2. Call the Task1 Python training entry task1_train.py;
#   3. Use default training parameters suitable for a single Ubuntu GPU machine;
#   4. Allow users to edit script arguments or call the Python entry directly for finer configuration.
#
# This script calls:
#   src/go2_rl/tasks/task1/task1_train.py
#
# Usage:
#   bash scripts/ubuntu/train_task1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task1 formal training"

go2_check_python_stack --isaaclab --skrl

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
