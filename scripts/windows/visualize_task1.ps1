# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task1 GUI 可视化入口。
#
# 本文件用于在 Windows 下以 GUI 方式可视化 Task1 平地运动模型。
# 主要职责:
#   1. 接收用户传入的 checkpoint 或使用 eval_task1.ps1 的自动 checkpoint 查找逻辑；
#   2. 调用 scripts/windows/eval_task1.ps1；
#   3. 默认不传入 -HeadlessEval，使 Isaac Sim 以 GUI 模式运行。
#
# 使用方式:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/visualize_task1.ps1 -IsaacLabRoot <path-to-IsaacLab> -Checkpoint <checkpoint>
#
# Unitree Go2 Scripts: Windows Task1 GUI visualization entry.
#
# This file visualizes a Task1 flat locomotion model with the Windows GUI.
# Main responsibilities:
#   1. Accept a user-provided checkpoint or reuse the automatic checkpoint discovery in eval_task1.ps1;
#   2. Call scripts/windows/eval_task1.ps1;
#   3. Keep -HeadlessEval off by default so Isaac Sim runs in GUI mode.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/visualize_task1.ps1 -IsaacLabRoot <path-to-IsaacLab> -Checkpoint <checkpoint>

param(
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = "",
    [string]$LogRoot = "",
    [string]$Checkpoint = "",
    [int]$Steps = 2000,
    [int]$PrintInterval = 50,
    [string]$Device = "cuda:0",
    [switch]$NoCleanup
)

& "$PSScriptRoot\eval_task1.ps1" `
    -ProjectRoot $ProjectRoot `
    -IsaacLabRoot $IsaacLabRoot `
    -LogRoot $LogRoot `
    -Checkpoint $Checkpoint `
    -NumEnvs 1 `
    -Steps $Steps `
    -PrintInterval $PrintInterval `
    -Device $Device `
    -NoCleanup:$NoCleanup
exit $LASTEXITCODE
