# Copyright (c) 2026
# Unitree Go2 Scripts: Windows 环境检查入口。
#
# 本文件用于检查 Windows 运行环境和项目基础结构，不启动训练、测试或模型评估。
# 检查内容:
#   1. ProjectRoot、IsaacLabRoot、python.bat 是否可以解析；
#   2. src/go2_rl、scripts、configs、tests 等基础目录是否存在；
#   3. Python、torch、IsaacLab、skrl 是否可以在 IsaacLab Python 环境中导入；
#   4. PYTHONPATH 是否能够覆盖项目 src 目录。
#
# 使用方式:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/check_env.ps1 -IsaacLabRoot <path-to-IsaacLab>
#
# Unitree Go2 Scripts: Windows environment check entry.
#
# This file checks the Windows runtime environment and the basic project
# structure. It does not launch training, testing, or model evaluation.
# Check items:
#   1. Whether ProjectRoot, IsaacLabRoot, and python.bat can be resolved;
#   2. Whether base directories such as src/go2_rl, scripts, configs, and tests exist;
#   3. Whether Python, torch, IsaacLab, and skrl can be imported in the IsaacLab Python environment;
#   4. Whether PYTHONPATH covers the project src directory.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/check_env.ps1 -IsaacLabRoot <path-to-IsaacLab>

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
Write-Host "Unitree Go2 Windows environment check"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "PythonBat = $PythonBat"
Write-Host "============================================================"

Test-Go2RequiredPath (Join-Path $ProjectRoot "src\go2_rl")
Test-Go2RequiredPath (Join-Path $ProjectRoot "scripts")
Test-Go2RequiredPath (Join-Path $ProjectRoot "configs")
Test-Go2RequiredPath (Join-Path $ProjectRoot "tests")

Set-Go2PythonPath -ProjectRoot $ProjectRoot
Test-Go2PythonStack -PythonBat $PythonBat -RequireIsaacLab -RequireSkrl

& $PythonBat -c "import go2_rl; print('[CHECK] go2_rl import: ok')"
if ($LASTEXITCODE -ne 0) {
    throw "go2_rl import check failed."
}

Write-Go2Ok "Windows environment check passed."
