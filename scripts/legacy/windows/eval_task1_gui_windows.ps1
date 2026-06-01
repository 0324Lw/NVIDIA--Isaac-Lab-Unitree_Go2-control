param(
    [string]$Checkpoint = "",
    [int]$NumEnvs = 1,
    [int]$Steps = 20000,
    [int]$PrintInterval = 200,
    [string]$Device = "cuda:0",
    [switch]$HeadlessEval,
    [switch]$NoCleanup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    chcp 65001 | Out-Null
    $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [Console]::InputEncoding = $Utf8NoBom
    [Console]::OutputEncoding = $Utf8NoBom
    $OutputEncoding = $Utf8NoBom
} catch {}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"

$ProjectRoot = "G:\rt_isaaclab_ws\repos\NVIDIA--Isaac-Lab-Unitree_Go2-control"
$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2"
$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$EvalPy = "G:\rt_isaaclab_ws\repos\NVIDIA--Isaac-Lab-Unitree_Go2-control\src\go2_rl\tasks\task1\task1_model_test.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\task1"
$ModelName = "go2_task1_model.pt"

function Find-LatestModelPath {
    param([string]$Root, [string]$ModelFileName)
    if (-not (Test-Path $Root)) { throw "log root not found: $Root" }
    $candidates = Get-ChildItem $Root -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $final = Join-Path $_.FullName "final_checkpoint"
            $model = Join-Path $final $ModelFileName
            if (Test-Path $model) {
                [PSCustomObject]@{
                    ModelPath = $model
                    LastWriteTime = (Get-Item $model).LastWriteTime
                }
            }
        } | Sort-Object LastWriteTime -Descending
    if ($null -eq $candidates -or $candidates.Count -eq 0) {
        throw "No final_checkpoint with $ModelFileName found under $Root"
    }
    return $candidates[0].ModelPath
}

function Resolve-CheckpointModelPath {
    param([string]$InputPath, [string]$Root, [string]$ModelFileName)
    if ([string]::IsNullOrWhiteSpace($InputPath)) { return Find-LatestModelPath -Root $Root -ModelFileName $ModelFileName }
    if (-not (Test-Path $InputPath)) { throw "checkpoint path not found: $InputPath" }
    $item = Get-Item $InputPath
    if ($item.PSIsContainer) {
        $model = Join-Path $item.FullName $ModelFileName
        if (-not (Test-Path $model)) { throw "model not found: $model" }
        return $model
    }
    return $item.FullName
}

function Stop-Go2EvalProcesses {
    param([string]$Reason = "eval cleanup")
    Write-Host "[CLEANUP] $Reason"
    $patterns = @(
        [regex]::Escape($ProjectRoot), [regex]::Escape($IsaacLabRoot),
        "isaaclab.python", "_isaac_sim", "Isaac-Sim", "task1_model_test.py"
    )
    $names = @("kit.exe", "python.exe", "pythonw.exe", "cmd.exe")
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $cmd = [string]$_.CommandLine
        if ([string]::IsNullOrWhiteSpace($cmd)) { return $false }
        if ($names -notcontains $_.Name) { return $false }
        foreach ($pat in $patterns) { if ($cmd -match $pat) { return $true } }
        return $false
    }
    foreach ($p in $targets) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }
        catch { Write-Host "[WARN] Cannot stop PID $($p.ProcessId)" }
    }
}

$CheckpointModel = Resolve-CheckpointModelPath -InputPath $Checkpoint -Root $LogRoot -ModelFileName $ModelName

Write-Host "`n============================================================"
Write-Host "Go2 Task1 Windows GUI model evaluation runner"
Write-Host "============================================================"
Write-Host "EvalPy          = $EvalPy"
Write-Host "CheckpointModel = $CheckpointModel"
Write-Host "NumEnvs         = $NumEnvs"
Write-Host "Steps           = $Steps"
Write-Host "HeadlessEval    = $HeadlessEval"
Write-Host "===========================================================`n"

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
Set-Location $ProjectRoot

$ArgsList = @(
    $EvalPy,
    "--checkpoint", "$CheckpointModel",
    "--num-envs", "$NumEnvs",
    "--steps", "$Steps",
    "--print-interval", "$PrintInterval",
    "--device", "$Device"
)
if ($HeadlessEval) { $ArgsList += "--headless-eval" }

try {
    & $PythonBat @ArgsList
    $exitCode = $LASTEXITCODE
} finally {
    if (-not $NoCleanup) { Stop-Go2EvalProcesses -Reason "after task1 eval" }
}

Write-Host "`n============================================================"
Write-Host "Go2 Task1 eval finished with exit code: $exitCode"
Write-Host "===========================================================`n"
exit $exitCode
