# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task2 快速训练检查入口。
#
# 本文件用于启动 Task2 多地形运动任务的 Windows 最小规模 smoke 训练。
# 主要职责:
#   1. 复用 scripts/windows/_common.ps1 解析 ProjectRoot、IsaacLabRoot、PythonBat 和 LogRoot；
#   2. 调用 Task2 的 Python 训练入口 task2_train.py；
#   3. 验证 Task2 terrain、privileged obs、Go2 环境和 skrl PPO 训练链路；
#   4. 通过 approval gate 避免误启动训练。
#
# 使用方式:
#   $env:GO2_TASK2_WINDOWS_SMOKE_APPROVED = "1"
#   powershell -ExecutionPolicy Bypass -File scripts/windows/smoke_task2.ps1 -IsaacLabRoot <path-to-IsaacLab>
#
# Unitree Go2 Scripts: Windows Task2 smoke training entry.
#
# This file launches a minimal Windows smoke training run for Task2 multi-terrain locomotion.
# Main responsibilities:
#   1. Reuse scripts/windows/_common.ps1 to resolve ProjectRoot, IsaacLabRoot, PythonBat, and LogRoot;
#   2. Call the Task2 Python training entry task2_train.py;
#   3. Validate the Task2 terrain, privileged obs, Go2 environment, and skrl PPO training pipeline;
#   4. Use an approval gate to avoid accidentally launching training.
#
# Usage:
#   $env:GO2_TASK2_WINDOWS_SMOKE_APPROVED = "1"
#   powershell -ExecutionPolicy Bypass -File scripts/windows/smoke_task2.ps1 -IsaacLabRoot <path-to-IsaacLab>

param(
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = "",
    [string]$LogRoot = "",
    [int]$NumEnvs = 32,
    [Int64]$TotalEnvSteps = 65536,
    [int]$Rollouts = 32,
    [int]$LearningEpochs = 3,
    [int]$MiniBatches = 4,
    [string]$Device = "cuda:0",
    [string]$RunName = "",
    [string]$PretrainedTask1 = ""
)

. "$PSScriptRoot\_common.ps1"

Set-Go2WindowsRuntime

$TaskName = "task2"
$ProjectRoot = Resolve-Go2ProjectRoot -ProjectRoot $ProjectRoot -ScriptRoot $PSScriptRoot
$IsaacLabRoot = Resolve-Go2IsaacLabRoot -IsaacLabRoot $IsaacLabRoot
$PythonBat = Resolve-Go2PythonBat -IsaacLabRoot $IsaacLabRoot
$LogRoot = Resolve-Go2LogRoot -ProjectRoot $ProjectRoot -TaskName $TaskName -LogRoot $LogRoot
$TrainPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task2\task2_train.py"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "win_task2_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task2 smoke training"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "TrainPy = $TrainPy"
Write-Host "LogRoot = $LogRoot"
Write-Host "NumEnvs = $NumEnvs"
Write-Host "TotalEnvSteps = $TotalEnvSteps"
Write-Host "Device = $Device"
Write-Host "RunName = $RunName"
Write-Host "PretrainedTask1 = $PretrainedTask1"
Write-Host "============================================================"

Confirm-Go2RunApproved -Kind "smoke" -TaskName $TaskName
Test-Go2RequiredPath $TrainPy
Set-Go2PythonPath -ProjectRoot $ProjectRoot
Test-Go2PythonStack -PythonBat $PythonBat -RequireIsaacLab -RequireSkrl

Set-Location $ProjectRoot

$ArgsList = @(
    $TrainPy,
    "--num-envs", "$NumEnvs",
    "--total-env-steps", "$TotalEnvSteps",
    "--rollouts", "$Rollouts",
    "--learning-epochs", "$LearningEpochs",
    "--mini-batches", "$MiniBatches",
    "--lr", "5e-5",
    "--min-lr", "2e-5",
    "--max-lr", "1.2e-4",
    "--summary-interval", "1",
    "--tb-log-interval-steps", "20",
    "--skrl-write-interval", "1000000",
    "--skrl-checkpoint-interval", "0",
    "--save-freq-env-steps", "$TotalEnvSteps",
    "--log-root", "$LogRoot",
    "--run-name", "$RunName",
    "--headless",
    "--device", "$Device"
)

if (-not [string]::IsNullOrWhiteSpace($PretrainedTask1)) {
    $ArgsList += @("--pretrained-task1", "$PretrainedTask1")
}

$Transcript = Start-Go2DriverTranscript -LogRoot $LogRoot -RunName $RunName
try {
    & $PythonBat @ArgsList
    $ExitCode = $LASTEXITCODE
} finally {
    Stop-Go2DriverTranscript -Started ([bool]$Transcript.Started)
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task2 smoke finished with exit code: $ExitCode"
Write-Host "Driver log: $($Transcript.Path)"
Write-Host "============================================================"
exit $ExitCode
