# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task4 快速训练检查入口。
#
# 本文件用于启动 Task4 Sim2Real / RMA teacher 任务的 Windows 最小规模 smoke 训练。
# 主要职责:
#   1. 复用 scripts/windows/_common.ps1 解析 ProjectRoot、IsaacLabRoot、PythonBat 和 LogRoot；
#   2. 调用 Task4 的 Python 训练入口 task4_train.py；
#   3. 使用当前稳定维度: actor history = 240，privileged obs = 25，teacher obs = 265；
#   4. 通过 approval gate 避免误启动训练。
#
# 使用方式:
#   $env:GO2_TASK4_WINDOWS_SMOKE_APPROVED = "1"
#   powershell -ExecutionPolicy Bypass -File scripts/windows/smoke_task4.ps1 -IsaacLabRoot <path-to-IsaacLab>
#
# Unitree Go2 Scripts: Windows Task4 smoke training entry.
#
# This file launches a minimal Windows smoke training run for the Task4 Sim2Real / RMA teacher task.
# Main responsibilities:
#   1. Reuse scripts/windows/_common.ps1 to resolve ProjectRoot, IsaacLabRoot, PythonBat, and LogRoot;
#   2. Call the Task4 Python training entry task4_train.py;
#   3. Use the current stable dimensions: actor history = 240, privileged obs = 25, teacher obs = 265;
#   4. Use an approval gate to avoid accidentally launching training.
#
# Usage:
#   $env:GO2_TASK4_WINDOWS_SMOKE_APPROVED = "1"
#   powershell -ExecutionPolicy Bypass -File scripts/windows/smoke_task4.ps1 -IsaacLabRoot <path-to-IsaacLab>

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
    [string]$PretrainedTask2 = "",
    [string]$PretrainedTask1 = "",
    [string]$PretrainedTask3 = ""
)

. "$PSScriptRoot\_common.ps1"

Set-Go2WindowsRuntime

$TaskName = "task4"
$ProjectRoot = Resolve-Go2ProjectRoot -ProjectRoot $ProjectRoot -ScriptRoot $PSScriptRoot
$IsaacLabRoot = Resolve-Go2IsaacLabRoot -IsaacLabRoot $IsaacLabRoot
$PythonBat = Resolve-Go2PythonBat -IsaacLabRoot $IsaacLabRoot
$LogRoot = Resolve-Go2LogRoot -ProjectRoot $ProjectRoot -TaskName $TaskName -LogRoot $LogRoot
$TrainPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task4\task4_train.py"

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "win_task4_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task4 smoke training"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "TrainPy = $TrainPy"
Write-Host "LogRoot = $LogRoot"
Write-Host "NumEnvs = $NumEnvs"
Write-Host "TotalEnvSteps = $TotalEnvSteps"
Write-Host "Device = $Device"
Write-Host "RunName = $RunName"
Write-Host "PretrainedTask2 = $PretrainedTask2"
Write-Host "PretrainedTask1 = $PretrainedTask1"
Write-Host "PretrainedTask3 = $PretrainedTask3"
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
    "--lr", "3e-5",
    "--min-lr", "2e-5",
    "--max-lr", "7e-5",
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

if (-not [string]::IsNullOrWhiteSpace($PretrainedTask2)) {
    $ArgsList += @("--pretrained-task2", "$PretrainedTask2")
}

if (-not [string]::IsNullOrWhiteSpace($PretrainedTask1)) {
    $ArgsList += @("--pretrained-task1", "$PretrainedTask1")
}

if (-not [string]::IsNullOrWhiteSpace($PretrainedTask3)) {
    $ArgsList += @("--pretrained-task3", "$PretrainedTask3")
}

$Transcript = Start-Go2DriverTranscript -LogRoot $LogRoot -RunName $RunName
try {
    & $PythonBat @ArgsList
    $ExitCode = $LASTEXITCODE
} finally {
    Stop-Go2DriverTranscript -Started ([bool]$Transcript.Started)
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task4 smoke finished with exit code: $ExitCode"
Write-Host "Driver log: $($Transcript.Path)"
Write-Host "============================================================"
exit $ExitCode
