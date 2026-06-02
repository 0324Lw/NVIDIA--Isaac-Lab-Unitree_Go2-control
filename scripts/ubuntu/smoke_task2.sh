#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task2 快速训练检查入口。
#
# 本文件用于启动 Task2 多地形运动任务的最小规模 smoke 训练。
# 主要用途:
#   1. 检查 Task2 terrain、privileged obs、skrl PPO 训练链路是否可以正常启动；
#   2. 使用较小 num-envs 和 total-env-steps，避免误启动大规模训练；
#   3. Task1 checkpoint 为可选项，便于开源用户先验证基础运行环境。
#
# 本脚本调用:
#   src/go2_rl/tasks/task2/task2_train.py
#
# 使用方式:
#   bash scripts/ubuntu/smoke_task2.sh
#
# Unitree Go2 Scripts: Ubuntu Task2 smoke training entry.
#
# This file launches a minimal smoke training run for Task2 multi-terrain locomotion.
# Main purposes:
#   1. Check whether the Task2 terrain, privileged obs, and skrl PPO training pipeline can start correctly;
#   2. Use small num-envs and total-env-steps to avoid accidentally launching a large training run;
#   3. Treat the Task1 checkpoint as optional so open-source users can validate the basic runtime first.
#
# This script calls:
#   src/go2_rl/tasks/task2/task2_train.py
#
# Usage:
#   bash scripts/ubuntu/smoke_task2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task2 smoke training"

go2_check_python_stack --isaaclab --skrl

python src/go2_rl/tasks/task2/task2_train.py \
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
