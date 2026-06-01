#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task3 解析世界测试入口。
#
# 本文件用于运行 Task3 纯 torch 解析世界测试，不启动 IsaacLab AppLauncher。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和 PYTHONPATH；
#   2. 检查 Python 和 torch 运行环境；
#   3. 调用 tests/task3/task3_world_test.py；
#   4. 保护 Task3 当前稳定维度: lidar rays = 60，world privileged dim = 68。
#
# 本脚本调用:
#   tests/task3/task3_world_test.py
#
# 使用方式:
#   bash scripts/ubuntu/test_task3_world.sh
#
# Unitree Go2 Scripts: Ubuntu Task3 analytical world test entry.
#
# This file runs the pure-torch Task3 analytical world test without launching
# IsaacLab AppLauncher.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and PYTHONPATH;
#   2. Check the Python and torch runtime environment;
#   3. Call tests/task3/task3_world_test.py;
#   4. Protect the current stable Task3 dimensions: lidar rays = 60, world privileged dim = 68.
#
# This script calls:
#   tests/task3/task3_world_test.py
#
# Usage:
#   bash scripts/ubuntu/test_task3_world.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task3 analytical world test"

go2_check_python_stack

python tests/task3/task3_world_test.py \
    --num-envs 2048 \
    --steps 200 \
    --test-device cuda:0
