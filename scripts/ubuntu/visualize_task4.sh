#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task4 GUI 可视化入口。
#
# 本文件用于在 Ubuntu 图形界面下可视化 Task4 Sim2Real / RMA teacher 模型。
# 主要职责:
#   1. 检查 checkpoint 参数；
#   2. 设置 DISPLAY 和项目 PYTHONPATH；
#   3. 调用 Task4 的 Python 评估入口 task4_model_test.py；
#   4. 使用当前稳定 Task4 维度: actor history = 240，privileged obs = 25，teacher obs = 265；
#   5. 默认使用 GUI，不传入 --headless-eval。
#
# 参数:
#   $1 必填，Task4 teacher checkpoint 路径；
#   $2 可选，start_k，默认 1.0。
#
# 本脚本调用:
#   src/go2_rl/tasks/task4/task4_model_test.py
#
# 使用方式:
#   bash scripts/ubuntu/visualize_task4.sh /path/to/go2_task4_teacher_model.pt
#   bash scripts/ubuntu/visualize_task4.sh /path/to/go2_task4_teacher_model.pt 0.30
#
# Unitree Go2 Scripts: Ubuntu Task4 GUI visualization entry.
#
# This file visualizes a Task4 Sim2Real / RMA teacher model with the Ubuntu GUI.
# Main responsibilities:
#   1. Check the checkpoint argument;
#   2. Set DISPLAY and the project PYTHONPATH;
#   3. Call the Task4 Python evaluation entry task4_model_test.py;
#   4. Use the current stable Task4 dimensions: actor history = 240, privileged obs = 25, teacher obs = 265;
#   5. Use GUI by default; --headless-eval is reserved for terminal-only evaluation.
#
# Arguments:
#   $1 required, Task4 teacher checkpoint path;
#   $2 optional, start_k, default 1.0.
#
# This script calls:
#   src/go2_rl/tasks/task4/task4_model_test.py
#
# Usage:
#   bash scripts/ubuntu/visualize_task4.sh /path/to/go2_task4_teacher_model.pt
#   bash scripts/ubuntu/visualize_task4.sh /path/to/go2_task4_teacher_model.pt 0.30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

CKPT="${1:-}"
START_K="${2:-1.0}"
go2_require_checkpoint_arg "${CKPT}" "Usage: bash scripts/ubuntu/visualize_task4.sh /path/to/go2_task4_teacher_model.pt [start_k]"

go2_prepare_runtime
export DISPLAY="${DISPLAY:-:0}"

go2_print_header "Unitree Go2 Task4 GUI visualization"

go2_check_python_stack --isaaclab --skrl

python src/go2_rl/tasks/task4/task4_model_test.py \
    --checkpoint "${CKPT}" \
    --num-envs 1 \
    --steps 2000 \
    --start-k "${START_K}" \
    --print-interval 50 \
    --visualize \
    --device cuda:0
