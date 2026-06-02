#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task1 模型评估入口。
#
# 本文件用于在 Ubuntu 下评估 Task1 平地运动模型。
# 主要职责:
#   1. 检查 checkpoint 参数；
#   2. 复用 scripts/ubuntu/_common.sh 设置项目路径和运行环境；
#   3. 调用 Task1 的 Python 评估入口 task1_model_test.py；
#   4. 默认使用 --headless-eval，适合服务器或终端环境。
#
# 参数:
#   $1 必填，Task1 checkpoint 路径。
#
# 本脚本调用:
#   src/go2_rl/tasks/task1/task1_model_test.py
#
# 使用方式:
#   bash scripts/ubuntu/eval_task1.sh /path/to/go2_task1_model.pt
#
# Unitree Go2 Scripts: Ubuntu Task1 model evaluation entry.
#
# This file evaluates a Task1 flat locomotion model on Ubuntu.
# Main responsibilities:
#   1. Check the checkpoint argument;
#   2. Reuse scripts/ubuntu/_common.sh to set the project path and runtime environment;
#   3. Call the Task1 Python evaluation entry task1_model_test.py;
#   4. Use --headless-eval by default for server or terminal environments.
#
# Arguments:
#   $1 required, Task1 checkpoint path.
#
# This script calls:
#   src/go2_rl/tasks/task1/task1_model_test.py
#
# Usage:
#   bash scripts/ubuntu/eval_task1.sh /path/to/go2_task1_model.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

CKPT="${1:-}"
go2_require_checkpoint_arg "${CKPT}" "Usage: bash scripts/ubuntu/eval_task1.sh /path/to/go2_task1_model.pt"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task1 model evaluation"

go2_check_python_stack --isaaclab --skrl

python src/go2_rl/tasks/task1/task1_model_test.py \
    --checkpoint "${CKPT}" \
    --num-envs 16 \
    --steps 2000 \
    --print-interval 100 \
    --headless-eval \
    --device cuda:0
