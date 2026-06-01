# Copyright (c) 2026
# Unitree Go2 Scripts: Windows Task4 GUI 可视化入口。
#
# 本文件用于在 Windows 下以 GUI 方式可视化 Task4 Sim2Real / RMA teacher 模型。
# 主要职责:
#   1. 接收用户传入的 checkpoint 或使用 eval_task4.ps1 的自动 checkpoint 查找逻辑；
#   2. 调用 scripts/windows/eval_task4.ps1；
#   3. 支持通过 StartK 指定课程进度评估点；
#   4. 使用当前稳定 Task4 维度: actor history = 240，privileged obs = 25，teacher obs = 265；
#   5. 默认不传入 -HeadlessEval，使 Isaac Sim 以 GUI 模式运行。
#
# 使用方式:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/visualize_task4.ps1 -IsaacLabRoot <path-to-IsaacLab> -Checkpoint <checkpoint>
#
# Unitree Go2 Scripts: Windows Task4 GUI visualization entry.
#
# This file visualizes a Task4 Sim2Real / RMA teacher model with the Windows GUI.
# Main responsibilities:
#   1. Accept a user-provided checkpoint or reuse the automatic checkpoint discovery in eval_task4.ps1;
#   2. Call scripts/windows/eval_task4.ps1;
#   3. Support StartK for evaluating a specific curriculum progress point;
#   4. Use the current stable Task4 dimensions: actor history = 240, privileged obs = 25, teacher obs = 265;
#   5. Do not pass -HeadlessEval by default, so Isaac Sim runs in GUI mode.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/windows/visualize_task4.ps1 -IsaacLabRoot <path-to-IsaacLab> -Checkpoint <checkpoint>

param(
    [string]$ProjectRoot = "",
    [string]$IsaacLabRoot = "",
    [string]$LogRoot = "",
    [string]$Checkpoint = "",
    [double]$StartK = 1.0,
    [int]$Steps = 3000,
    [int]$PrintInterval = 50,
    [string]$Device = "cuda:0",
    [switch]$NoCleanup
)

& "$PSScriptRoot\eval_task4.ps1" `
    -ProjectRoot $ProjectRoot `
    -IsaacLabRoot $IsaacLabRoot `
    -LogRoot $LogRoot `
    -Checkpoint $Checkpoint `
    -NumEnvs 1 `
    -Steps $Steps `
    -StartK $StartK `
    -PrintInterval $PrintInterval `
    -Device $Device `
    -NoCleanup:$NoCleanup
exit $LASTEXITCODE
