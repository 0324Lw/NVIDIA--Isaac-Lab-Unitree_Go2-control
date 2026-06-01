#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu 环境检查入口。
#
# 本文件用于检查 Ubuntu 运行环境和项目基础结构，不启动训练、测试或模型评估。
# 检查内容:
#   1. 项目根目录是否可以从脚本位置正确解析；
#   2. src/go2_rl、scripts、configs、tests 等基础目录是否存在；
#   3. Python、torch、IsaacLab、skrl 是否可以在当前环境中导入；
#   4. PYTHONPATH 是否能够覆盖项目 src 目录。
#
# 使用方式:
#   bash scripts/ubuntu/check_env.sh
#
# Unitree Go2 Scripts: Ubuntu environment check entry.
#
# This file checks the Ubuntu runtime environment and the basic project
# structure. It does not launch training, testing, or model evaluation.
# Check items:
#   1. Whether the project root can be resolved from the script location;
#   2. Whether base directories such as src/go2_rl, scripts, configs, and tests exist;
#   3. Whether Python, torch, IsaacLab, and skrl can be imported in the active environment;
#   4. Whether PYTHONPATH covers the project src directory.
#
# Usage:
#   bash scripts/ubuntu/check_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

go2_prepare_runtime
go2_print_header "Unitree Go2 Ubuntu environment check"

go2_require_dir "${GO2_PROJECT_ROOT}/src/go2_rl"
go2_require_dir "${GO2_PROJECT_ROOT}/scripts"
go2_require_dir "${GO2_PROJECT_ROOT}/configs"
go2_require_dir "${GO2_PROJECT_ROOT}/tests"

go2_require_file "${GO2_PROJECT_ROOT}/src/go2_rl/tasks/task1/task1_config.py"
go2_require_file "${GO2_PROJECT_ROOT}/src/go2_rl/tasks/task2/task2_config.py"
go2_require_file "${GO2_PROJECT_ROOT}/src/go2_rl/tasks/task3/task3_config.py"
go2_require_file "${GO2_PROJECT_ROOT}/src/go2_rl/tasks/task4/task4_config.py"

go2_check_python_stack --isaaclab --skrl

python - <<'PY'
import go2_rl
print("[CHECK] go2_rl import: ok")
PY

go2_ok "Ubuntu environment check passed."
