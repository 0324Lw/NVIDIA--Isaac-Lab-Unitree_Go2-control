# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task2 正式训练入口。
#
# 本文件用于启动 Task2 多地形运动任务的 Windows 正式训练流程。
# 主要职责:
#   1. 复用 scripts/windows/_common.ps1 解析 ProjectRoot、IsaacLabRoot、PythonBat 和 LogRoot；
#   2. 调用 Task2 的 Python 训练入口 task2_train.py；
#   3. 保留 Task1 checkpoint warm-start 和 resume 的可选入口；
#   4. 通过 approval gate 避免误启动长时间训练。
#
# 使用方式:
#   $env:GO2_TASK2_WINDOWS_TRAIN_APPROVED = "1"
#   powershell -ExecutionPolicy Bypass -File scripts/windows/train_task2.ps1 -IsaacLabRoot <path-to-IsaacLab> -PretrainedTask1 <checkpoint>
#
# Unitree Go2 Scripts: Windows Task2 formal training entry.
#
# This file launches the Windows formal training pipeline for Task2 multi-terrain locomotion.
# Main responsibilities:
#   1. Reuse scripts/windows/_common.ps1 to resolve ProjectRoot, IsaacLabRoot, PythonBat, and LogRoot;
#   2. Call the Task2 Python training entry task2_train.py;
#   3. Keep optional Task1 checkpoint warm-start and resume entries;
#   4. Use an approval gate to avoid accidentally launching a long training run.
#
# Usage:
#   $env:GO2_TASK2_WINDOWS_TRAIN_APPROVED = "1"
#   powershell -ExecutionPolicy Bypass -File scripts/windows/train_task2.ps1 -IsaacLabRoot <path-to-IsaacLab> -PretrainedTask1 <checkpoint>

param(
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = "",
    [string]$LogRoot = "",
    [int]$NumEnvs = 1024,
    [Int64]$TotalEnvSteps = 600000000,
    [int]$Rollouts = 64,
    [int]$LearningEpochs = 5,
    [int]$MiniBatches = 8,
    [double]$Lr = 0.00005,
    [double]$MinLr = 0.00002,
    [double]$MaxLr = 0.00012,
    [string]$Device = "cuda:0",
    [string]$RunName = "",
    [string]$PretrainedTask1 = "",
    [string]$Resume = ""
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
    $RunName = "win_task2_train_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task2 formal training"
Write-Host "============================================================"
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "TrainPy = $TrainPy"
Write-Host "LogRoot = $LogRoot"
Write-Host "NumEnvs = $NumEnvs"
Write-Host "TotalEnvSteps = $TotalEnvSteps"
Write-Host "Rollouts = $Rollouts"
Write-Host "LearningEpochs = $LearningEpochs"
Write-Host "MiniBatches = $MiniBatches"
Write-Host "LR = $Lr / $MinLr / $MaxLr"
Write-Host "Device = $Device"
Write-Host "RunName = $RunName"
Write-Host "PretrainedTask1 = $PretrainedTask1"
Write-Host "Resume = $Resume"
Write-Host "============================================================"

Confirm-Go2RunApproved -Kind "train" -TaskName $TaskName
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
    "--lr", "$Lr",
    "--min-lr", "$MinLr",
    "--max-lr", "$MaxLr",
    "--gamma", "0.995",
    "--gae-lambda", "0.95",
    "--kl-threshold", "0.015",
    "--entropy-coef", "0.003",
    "--value-coef", "2.0",
    "--init-log-std", "-1.25",
    "--pretrained-log-std", "-1.65",
    "--summary-interval", "20",
    "--tb-log-interval-steps", "100",
    "--skrl-write-interval", "1000000",
    "--skrl-checkpoint-interval", "0",
    "--save-freq-env-steps", "20000000",
    "--log-root", "$LogRoot",
    "--run-name", "$RunName",
    "--headless",
    "--device", "$Device"
)

if (-not [string]::IsNullOrWhiteSpace($PretrainedTask1)) {
    $ArgsList += @("--pretrained-task1", "$PretrainedTask1")
}

if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $ArgsList += @("--resume", "$Resume")
}

$Transcript = Start-Go2DriverTranscript -LogRoot $LogRoot -RunName $RunName
try {
    & $PythonBat @ArgsList
    $ExitCode = $LASTEXITCODE
} finally {
    Stop-Go2DriverTranscript -Started ([bool]$Transcript.Started)
}

Write-Host "============================================================"
Write-Host "Unitree Go2 Windows Task2 training finished with exit code: $ExitCode"
Write-Host "Driver log: $($Transcript.Path)"
Write-Host "============================================================"
exit $ExitCode
