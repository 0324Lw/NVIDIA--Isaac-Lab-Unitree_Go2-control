#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task3 环境测试入口。
#
# 本文件用于运行 Task3 导航避障环境测试。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和 PYTHONPATH；
#   2. 检查 Python、torch、IsaacLab 运行环境；
#   3. 调用 tests/task3/task3_env_test.py；
#   4. 保护当前稳定 Task3 维度: actor obs = 208，privileged obs = 276，lidar rays = 60。
#
# 本脚本调用:
#   tests/task3/task3_env_test.py
#
# 使用方式:
#   bash scripts/ubuntu/test_task3_env.sh
#
# Unitree Go2 Scripts: Ubuntu Task3 environment test entry.
#
# This file runs the Task3 navigation and obstacle avoidance environment test.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and PYTHONPATH;
#   2. Check the Python, torch, and IsaacLab runtime environment;
#   3. Call tests/task3/task3_env_test.py;
#   4. Protect the current stable Task3 dimensions: actor obs = 208, privileged obs = 276, lidar rays = 60.
#
# This script calls:
#   tests/task3/task3_env_test.py
#
# Usage:
#   bash scripts/ubuntu/test_task3_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task3 environment test"

go2_check_python_stack --isaaclab

python tests/task3/task3_env_test.py \
    --num-envs 32 \
    --steps 240 \
    --collect-interval 40 \
    --rollout-k 0.12 \
    --headless \
    --test-device cuda:0
