#!/usr/bin/env bash
# Copyright (c) 2026
# Unitree Go2 Scripts: Ubuntu 公共脚本工具。
#
# 本文件为 Ubuntu 运行脚本提供公共函数，不直接启动训练、测试或评估。
# 主要职责:
#   1. 根据脚本位置解析项目根目录；
#   2. 设置 PYTHONPATH，使 src/go2_rl 可以被直接导入；
#   3. 提供统一的控制台输出格式；
#   4. 检查 Python / torch / IsaacLab / skrl 运行环境；
#   5. 检查必要文件、目录和 checkpoint 参数。
#
# 路径设计:
#   项目根目录由 scripts/ubuntu/_common.sh 的相对位置推导；
#   不写入个人绝对路径，不绑定具体硬件型号。
#
# 使用方式:
#   source scripts/ubuntu/_common.sh
#   go2_prepare_runtime
#
# Unitree Go2 Scripts: Ubuntu common script utilities.
#
# This file provides shared functions for Ubuntu runtime scripts. It does not
# launch training, testing, or evaluation directly.
# Main responsibilities:
#   1. Resolve the project root from the script location;
#   2. Set PYTHONPATH so src/go2_rl can be imported directly;
#   3. Provide a unified console output format;
#   4. Check the Python / torch / IsaacLab / skrl runtime environment;
#   5. Validate required files, directories, and checkpoint arguments.
#
# Path design:
#   The project root is resolved from the relative location of this file;
#   no personal absolute path or hardware-specific name is stored here.
#
# Usage:
#   source scripts/ubuntu/_common.sh
#   go2_prepare_runtime

set -euo pipefail

GO2_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO2_PROJECT_ROOT="$(cd "${GO2_SCRIPT_DIR}/../.." && pwd)"

go2_info() {
    echo "[INFO] $*"
}

go2_ok() {
    echo "[OK] $*"
}

go2_warn() {
    echo "[WARN] $*" >&2
}

go2_fail() {
    echo "[FAIL] $*" >&2
}

go2_project_root() {
    echo "${GO2_PROJECT_ROOT}"
}

go2_cd_project_root() {
    cd "${GO2_PROJECT_ROOT}"
}

go2_setup_pythonpath() {
    export PYTHONPATH="${GO2_PROJECT_ROOT}/src:${PYTHONPATH:-}"
}

go2_prepare_runtime() {
    go2_cd_project_root
    go2_setup_pythonpath
}

go2_print_header() {
    local title="$1"
    echo "============================================================"
    echo "${title}"
    echo "PROJECT_ROOT=${GO2_PROJECT_ROOT}"
    echo "PYTHON=$(command -v python || true)"
    echo "============================================================"
}

go2_require_file() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        go2_fail "Required file not found: ${path}"
        return 1
    fi
}

go2_require_dir() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        go2_fail "Required directory not found: ${path}"
        return 1
    fi
}

go2_require_checkpoint_arg() {
    local checkpoint="${1:-}"
    local usage="${2:-Usage: provide checkpoint path}"
    if [ -z "${checkpoint}" ]; then
        go2_fail "${usage}"
        return 1
    fi
}

go2_check_python_stack() {
    local require_isaaclab=0
    local require_skrl=0

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --isaaclab)
                require_isaaclab=1
                ;;
            --skrl)
                require_skrl=1
                ;;
            *)
                go2_fail "Unknown go2_check_python_stack option: $1"
                return 1
                ;;
        esac
        shift
    done

    REQUIRE_ISAACLAB="${require_isaaclab}" REQUIRE_SKRL="${require_skrl}" python - <<'PY'
import os
import sys

print("[CHECK] Python:", sys.executable)

try:
    import torch
    print("[CHECK] torch:", torch.__version__)
    print("[CHECK] cuda available:", torch.cuda.is_available())
except Exception as exc:
    raise RuntimeError(
        "Current Python cannot import torch. Please activate the IsaacLab environment."
    ) from exc

if os.environ.get("REQUIRE_ISAACLAB") == "1":
    try:
        import isaaclab  # noqa: F401
        print("[CHECK] isaaclab: ok")
    except Exception as exc:
        raise RuntimeError(
            "Current Python cannot import isaaclab. Please activate the IsaacLab environment."
        ) from exc

if os.environ.get("REQUIRE_SKRL") == "1":
    try:
        import skrl
        print("[CHECK] skrl:", getattr(skrl, "__version__", "unknown"))
    except Exception as exc:
        raise RuntimeError(
            "Current Python cannot import skrl. Please install skrl in the active environment."
        ) from exc
PY
}
