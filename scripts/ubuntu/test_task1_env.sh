#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task1 环境测试入口。
#
# 本文件用于运行 Task1 平地运动环境测试。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和 PYTHONPATH；
#   2. 检查 Python、torch、IsaacLab 运行环境；
#   3. 调用 tests/task1/task1_env_test.py；
#   4. 验证 Task1 环境初始化、观测、接触、课程、终止条件和 rollout 链路。
#
# 本脚本调用:
#   tests/task1/task1_env_test.py
#
# 使用方式:
#   bash scripts/ubuntu/test_task1_env.sh
#
# Unitree Go2 Scripts: Ubuntu Task1 environment test entry.
#
# This file runs the Task1 flat locomotion environment test.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and PYTHONPATH;
#   2. Check the Python, torch, and IsaacLab runtime environment;
#   3. Call tests/task1/task1_env_test.py;
#   4. Validate Task1 environment initialization, observations, contacts, curriculum, termination, and rollout pipeline.
#
# This script calls:
#   tests/task1/task1_env_test.py
#
# Usage:
#   bash scripts/ubuntu/test_task1_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task1 environment test"

go2_check_python_stack --isaaclab

python tests/task1/task1_env_test.py \
    --num-envs 64 \
    --steps 300 \
    --collect-interval 50 \
    --headless \
    --device cuda:0
