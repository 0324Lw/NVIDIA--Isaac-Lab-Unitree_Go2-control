#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task3 快速训练检查入口。
#
# 本文件用于启动 Task3 导航避障任务的最小规模 smoke 训练。
# 主要用途:
#   1. 检查 Task3 解析世界、lidar、privileged obs 和 skrl PPO 训练链路是否可以正常启动；
#   2. 使用当前稳定维度: actor obs = 208，privileged obs = 276，lidar rays = 60；
#   3. 使用较小 num-envs 和 total-env-steps，避免误启动大规模训练。
#
# 本脚本调用:
#   src/go2_rl/tasks/task3/task3_train.py
#
# 使用方式:
#   bash scripts/ubuntu/smoke_task3.sh
#
# Unitree Go2 Scripts: Ubuntu Task3 smoke training entry.
#
# This file launches a minimal smoke training run for Task3 navigation and obstacle avoidance.
# Main purposes:
#   1. Check whether the Task3 analytical world, lidar, privileged obs, and skrl PPO training pipeline can start correctly;
#   2. Use the current stable dimensions: actor obs = 208, privileged obs = 276, lidar rays = 60;
#   3. Use small num-envs and total-env-steps to avoid accidentally launching a large training run.
#
# This script calls:
#   src/go2_rl/tasks/task3/task3_train.py
#
# Usage:
#   bash scripts/ubuntu/smoke_task3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task3 smoke training"

go2_check_python_stack --isaaclab --skrl

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
