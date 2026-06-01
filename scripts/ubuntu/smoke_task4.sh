#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task4 快速训练检查入口。
#
# 本文件用于启动 Task4 Sim2Real / RMA teacher 任务的最小规模 smoke 训练。
# 主要用途:
#   1. 检查 Task4 teacher obs、domain randomization、外部扰动和 skrl PPO 训练链路是否可以正常启动；
#   2. 使用当前稳定维度: actor history = 240，privileged obs = 25，teacher obs = 265；
#   3. 使用较小 num-envs 和 total-env-steps，避免误启动大规模训练。
#
# 本脚本调用:
#   src/go2_rl/tasks/task4/task4_train.py
#
# 使用方式:
#   bash scripts/ubuntu/smoke_task4.sh
#
# Unitree Go2 Scripts: Ubuntu Task4 smoke training entry.
#
# This file launches a minimal smoke training run for the Task4 Sim2Real / RMA teacher task.
# Main purposes:
#   1. Check whether Task4 teacher obs, domain randomization, external disturbances, and the skrl PPO training pipeline can start correctly;
#   2. Use the current stable dimensions: actor history = 240, privileged obs = 25, teacher obs = 265;
#   3. Use small num-envs and total-env-steps to avoid accidentally launching a large training run.
#
# This script calls:
#   src/go2_rl/tasks/task4/task4_train.py
#
# Usage:
#   bash scripts/ubuntu/smoke_task4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task4 smoke training"

go2_check_python_stack --isaaclab --skrl

python src/go2_rl/tasks/task4/task4_train.py \
    --num-envs 32 \
    --total-env-steps 65536 \
    --rollouts 32 \
    --learning-epochs 3 \
    --mini-batches 4 \
    --lr 3e-5 \
    --min-lr 2e-5 \
    --max-lr 7e-5 \
    --summary-interval 1 \
    --tb-log-interval-steps 20 \
    --skrl-write-interval 1000000 \
    --skrl-checkpoint-interval 0 \
    --save-freq-env-steps 65536 \
    --headless \
    --device cuda:0
