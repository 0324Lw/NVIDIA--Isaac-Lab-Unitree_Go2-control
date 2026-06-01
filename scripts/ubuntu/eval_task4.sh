#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task4 模型评估入口。
#
# 本文件用于在 Ubuntu 下评估 Task4 Sim2Real / RMA teacher 模型。
# 主要职责:
#   1. 检查 checkpoint 参数；
#   2. 复用 scripts/ubuntu/_common.sh 设置项目路径和运行环境；
#   3. 调用 Task4 的 Python 评估入口 task4_model_test.py；
#   4. 使用当前稳定 Task4 维度: actor history = 240，privileged obs = 25，teacher obs = 265；
#   5. 支持通过 start_k 指定课程进度评估点。
#
# 参数:
#   $1 必填，Task4 teacher checkpoint 路径；
#   $2 可选，start_k，默认 1.0。
#
# 本脚本调用:
#   src/go2_rl/tasks/task4/task4_model_test.py
#
# 使用方式:
#   bash scripts/ubuntu/eval_task4.sh /path/to/go2_task4_teacher_model.pt
#   bash scripts/ubuntu/eval_task4.sh /path/to/go2_task4_teacher_model.pt 0.30
#
# Unitree Go2 Scripts: Ubuntu Task4 model evaluation entry.
#
# This file evaluates a Task4 Sim2Real / RMA teacher model on Ubuntu.
# Main responsibilities:
#   1. Check the checkpoint argument;
#   2. Reuse scripts/ubuntu/_common.sh to set the project path and runtime environment;
#   3. Call the Task4 Python evaluation entry task4_model_test.py;
#   4. Use the current stable Task4 dimensions: actor history = 240, privileged obs = 25, teacher obs = 265;
#   5. Support start_k for evaluating a specific curriculum progress point.
#
# Arguments:
#   $1 required, Task4 teacher checkpoint path;
#   $2 optional, start_k, default 1.0.
#
# This script calls:
#   src/go2_rl/tasks/task4/task4_model_test.py
#
# Usage:
#   bash scripts/ubuntu/eval_task4.sh /path/to/go2_task4_teacher_model.pt
#   bash scripts/ubuntu/eval_task4.sh /path/to/go2_task4_teacher_model.pt 0.30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

CKPT="${1:-}"
START_K="${2:-1.0}"
go2_require_checkpoint_arg "${CKPT}" "Usage: bash scripts/ubuntu/eval_task4.sh /path/to/go2_task4_teacher_model.pt [start_k]"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task4 model evaluation"

go2_check_python_stack --isaaclab --skrl

python src/go2_rl/tasks/task4/task4_model_test.py \
    --checkpoint "${CKPT}" \
    --num-envs 16 \
    --steps 3000 \
    --start-k "${START_K}" \
    --print-interval 100 \
    --headless-eval \
    --device cuda:0
