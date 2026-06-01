#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu Task4 环境测试入口。
#
# 本文件用于运行 Task4 Sim2Real / RMA teacher 环境测试。
# 主要职责:
#   1. 复用 scripts/ubuntu/_common.sh 解析项目根目录和 PYTHONPATH；
#   2. 检查 Python、torch、IsaacLab 运行环境；
#   3. 调用 tests/task4/task4_env_test.py；
#   4. 验证 actor history、privileged obs、teacher obs、domain randomization、外部扰动和 rollout 链路。
#
# 本脚本调用:
#   tests/task4/task4_env_test.py
#
# 使用方式:
#   bash scripts/ubuntu/test_task4_env.sh
#
# Unitree Go2 Scripts: Ubuntu Task4 environment test entry.
#
# This file runs the Task4 Sim2Real / RMA teacher environment test.
# Main responsibilities:
#   1. Reuse scripts/ubuntu/_common.sh to resolve the project root and PYTHONPATH;
#   2. Check the Python, torch, and IsaacLab runtime environment;
#   3. Call tests/task4/task4_env_test.py;
#   4. Validate actor history, privileged obs, teacher obs, domain randomization, external disturbances, and rollout pipeline.
#
# This script calls:
#   tests/task4/task4_env_test.py
#
# Usage:
#   bash scripts/ubuntu/test_task4_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Task4 environment test"

go2_check_python_stack --isaaclab

python tests/task4/task4_env_test.py \
    --num-envs 32 \
    --steps 240 \
    --collect-interval 40 \
    --rollout-k 0.30 \
    --headless \
    --test-device cuda:0
