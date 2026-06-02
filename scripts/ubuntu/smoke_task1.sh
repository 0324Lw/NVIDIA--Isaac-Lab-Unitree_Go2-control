#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task1 快速训练检查入口。
#
# 本文件用于启动 Task1 平地运动任务的最小规模 smoke 训练。
# 主要用途:
#   1. 检查 IsaacLab、skrl、Go2 环境和 PPO 训练链路是否可以正常启动；
#   2. 使用较小 num-envs 和 total-env-steps，避免误启动大规模训练；
#   3. 将日志写入 task1_train.py 默认日志目录或用户通过参数覆盖的日志目录。
#
# 本脚本调用:
#   src/go2_rl/tasks/task1/task1_train.py
#
# 使用方式:
#   bash scripts/ubuntu/smoke_task1.sh
#
# Unitree Go2 Scripts: Ubuntu Task1 smoke training entry.
#
# This file launches a minimal smoke training run for Task1 flat locomotion.
# Main purposes:
#   1. Check whether IsaacLab, skrl, the Go2 environment, and the PPO training pipeline can start correctly;
#   2. Use small num-envs and total-env-steps to avoid accidentally launching a large training run;
#   3. Write logs to the default task1_train.py log directory or to a user-provided log directory.
#
# This script calls:
#   src/go2_rl/tasks/task1/task1_train.py
#
# Usage:
#   bash scripts/ubuntu/smoke_task1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task1 smoke training"

go2_check_python_stack --isaaclab --skrl

python src/go2_rl/tasks/task1/task1_train.py \
    --num-envs 64 \
    --total-env-steps 65536 \
    --rollouts 32 \
    --learning-epochs 3 \
    --mini-batches 4 \
    --summary-interval 1 \
    --tb-log-interval-steps 20 \
    --skrl-write-interval 1000000 \
    --skrl-checkpoint-interval 0 \
    --save-freq-env-steps 65536 \
    --headless \
    --device cuda:0
