# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task3 工程检查入口。
#
# 本文件用于检查 Task3 导航避障任务的 Windows 工程结构和运行环境，不启动训练或评估。
# 检查内容:
#   1. ProjectRoot、IsaacLabRoot、python.bat 是否可以解析；
#   2. Task3 config、world、env、train、model_test 文件是否存在；
#   3. Python、torch、IsaacLab、skrl 是否可以导入；
#   4. 当前 Task3 稳定维度由 Python 测试保护: actor obs = 208，privileged obs = 276，lidar rays = 60。
#
# 使用方式:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/check_task3.ps1 -IsaacLabRoot <path-to-IsaacLab>
#
# Unitree Go2 Scripts: Windows Task3 project check entry.
#
# This file checks the Windows project structure and runtime environment for
# Task3 navigation and obstacle avoidance. It does not launch training or evaluation.
# Check items:
#   1. Whether ProjectRoot, IsaacLabRoot, and python.bat can be resolved;
#   2. Whether Task3 config, world, env, train, and model_test files exist;
#   3. Whether Python, torch, IsaacLab, and skrl can be imported;
#   4. The current stable Task3 dimensions are protected by Python tests: actor obs = 208, privileged obs = 276, lidar rays = 60.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/check_task3.ps1 -IsaacLabRoot <path-to-IsaacLab>

param(
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = ""
)

. "$PSScriptRoot\_common.ps1"

Set-Go2WindowsRuntime

$ProjectRoot = Resolve-Go2ProjectRoot -ProjectRoot $ProjectRoot -ScriptRoot $PSScriptRoot
$IsaacLabRoot = Resolve-Go2IsaacLabRoot -IsaacLabRoot $IsaacLabRoot
$PythonBat = Resolve-Go2PythonBat -IsaacLabRoot $IsaacLabRoot

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task3 project check"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "PythonBat = $PythonBat"
Write-Host "============================================================"

Test-Go2RequiredPath (Join-Path $ProjectRoot "src\go2_rl\tasks\task3\task3_config.py")
Test-Go2RequiredPath (Join-Path $ProjectRoot "src\go2_rl\tasks\task3\task3_world.py")
Test-Go2RequiredPath (Join-Path $ProjectRoot "src\go2_rl\tasks\task3\task3_env.py")
Test-Go2RequiredPath (Join-Path $ProjectRoot "src\go2_rl\tasks\task3\task3_train.py")
Test-Go2RequiredPath (Join-Path $ProjectRoot "src\go2_rl\tasks\task3\task3_model_test.py")
Test-Go2RequiredPath (Join-Path $ProjectRoot "tests\task3\task3_world_test.py")
Test-Go2RequiredPath (Join-Path $ProjectRoot "tests\task3\task3_env_test.py")

Set-Go2PythonPath -ProjectRoot $ProjectRoot
Test-Go2PythonStack -PythonBat $PythonBat -RequireIsaacLab -RequireSkrl

Write-Go2Ok "Windows Task3 project check passed."
