#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task2 环境测试入口。
#
# 本文件用于运行 Task2 多地形运动环境测试。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和 PYTHONPATH；
#   2. 检查 Python、torch、IsaacLab 运行环境；
#   3. 调用 tests/task2/task2_env_test.py；
#   4. 验证 Task2 环境、terrain privileged obs、接触、课程和 rollout 链路。
#
# 本脚本调用:
#   tests/task2/task2_env_test.py
#
# 使用方式:
#   bash scripts/ubuntu/test_task2_env.sh
#
# Unitree Go2 Scripts: Ubuntu Task2 environment test entry.
#
# This file runs the Task2 multi-terrain locomotion environment test.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and PYTHONPATH;
#   2. Check the Python, torch, and IsaacLab runtime environment;
#   3. Call tests/task2/task2_env_test.py;
#   4. Validate the Task2 environment, terrain privileged obs, contacts, curriculum, and rollout pipeline.
#
# This script calls:
#   tests/task2/task2_env_test.py
#
# Usage:
#   bash scripts/ubuntu/test_task2_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task2 environment test"

go2_check_python_stack --isaaclab

python tests/task2/task2_env_test.py \
    --num-envs 32 \
    --steps 240 \
    --collect-interval 40 \
    --headless \
    --device cuda:0
