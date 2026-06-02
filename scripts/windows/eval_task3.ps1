# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task3 模型评估入口。
#
# 本文件用于在 Windows 下评估 Task3 导航避障模型。
# 主要职责:
#   1. 复用 scripts/windows/_common.ps1 解析 ProjectRoot、IsaacLabRoot、PythonBat 和 LogRoot；
#   2. 支持用户传入 checkpoint，或者从日志目录自动查找最新 final_checkpoint；
#   3. 调用 Task3 的 Python 评估入口 task3_model_test.py；
#   4. 使用当前稳定维度: actor obs = 208，privileged obs = 276，lidar rays = 60；
#   5. 默认使用 GUI，只有传入 -HeadlessEval 时才切换无头评估。
#
# 使用方式:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/eval_task3.ps1 -IsaacLabRoot <path-to-IsaacLab> -Checkpoint <checkpoint>
#
# Unitree Go2 Scripts: Windows Task3 model evaluation entry.
#
# This file evaluates a Task3 navigation and obstacle avoidance model on Windows.
# Main responsibilities:
#   1. Reuse scripts/windows/_common.ps1 to resolve ProjectRoot, IsaacLabRoot, PythonBat, and LogRoot;
#   2. Support a user-provided checkpoint or automatically find the latest final_checkpoint from the log directory;
#   3. Call the Task3 Python evaluation entry task3_model_test.py;
#   4. Use the current stable dimensions: actor obs = 208, privileged obs = 276, lidar rays = 60;
#   5. Use GUI by default, and switch to headless evaluation only when -HeadlessEval is provided.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/eval_task3.ps1 -IsaacLabRoot <path-to-IsaacLab> -Checkpoint <checkpoint>

param(
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = "",
    [string]$LogRoot = "",
    [string]$Checkpoint = "",
    [int]$NumEnvs = 1,
    [int]$Steps = 3000,
    [double]$StartK = 1.0,
    [int]$PrintInterval = 100,
    [string]$Device = "cuda:0",
    [switch]$HeadlessEval,
    [switch]$NoCleanup
)

. "$PSScriptRoot\_common.ps1"

Set-Go2WindowsRuntime

$TaskName = "task3"
$ProjectRoot = Resolve-Go2ProjectRoot -ProjectRoot $ProjectRoot -ScriptRoot $PSScriptRoot
$IsaacLabRoot = Resolve-Go2IsaacLabRoot -IsaacLabRoot $IsaacLabRoot
$PythonBat = Resolve-Go2PythonBat -IsaacLabRoot $IsaacLabRoot
$LogRoot = Resolve-Go2LogRoot -ProjectRoot $ProjectRoot -TaskName $TaskName -LogRoot $LogRoot
$EvalPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task3\task3_model_test.py"
$CheckpointModel = Resolve-Go2Checkpoint -Checkpoint $Checkpoint -LogRoot $LogRoot -ModelFileName "go2_task3_model.pt"

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task3 model evaluation"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "EvalPy = $EvalPy"
Write-Host "CheckpointModel = $CheckpointModel"
Write-Host "NumEnvs = $NumEnvs"
Write-Host "Steps = $Steps"
Write-Host "StartK = $StartK"
Write-Host "HeadlessEval = $HeadlessEval"
Write-Host "Device = $Device"
Write-Host "============================================================"

Test-Go2RequiredPath $EvalPy
Set-Go2PythonPath -ProjectRoot $ProjectRoot
Test-Go2PythonStack -PythonBat $PythonBat -RequireIsaacLab -RequireSkrl

Set-Location $ProjectRoot

$ArgsList = @(
    $EvalPy,
    "--checkpoint", "$CheckpointModel",
    "--num-envs", "$NumEnvs",
    "--steps", "$Steps",
    "--start-k", "$StartK",
    "--print-interval", "$PrintInterval",
    "--device", "$Device"
)

if ($HeadlessEval) {
    $ArgsList += "--headless-eval"
}

try {
    & $PythonBat @ArgsList
    $ExitCode = $LASTEXITCODE
} finally {
    if (-not $NoCleanup) {
        Stop-Go2IsaacProcesses -ProjectRoot $ProjectRoot -IsaacLabRoot $IsaacLabRoot -PythonFileName "task3_model_test.py" -Reason "after task3 eval"
    }
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task3 eval finished with exit code: $ExitCode"
Write-Host "============================================================"
exit $ExitCode
