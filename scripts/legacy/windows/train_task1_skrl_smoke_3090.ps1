param(
    [int]$NumEnvs = 64,
    [Int64]$TotalEnvSteps = 65536,
    [int]$Rollouts = 32,
    [int]$LearningEpochs = 3,
    [int]$MiniBatches = 4,
    [string]$Device = "cuda:0",
    [string]$RunName = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"



# Windows runtime compatibility.
# Keep Python logs UTF-8 and unbuffered. This does not change training logic.
try {
    chcp 65001 | Out-Null
    $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [Console]::InputEncoding = $Utf8NoBom
    [Console]::OutputEncoding = $Utf8NoBom
    $OutputEncoding = $Utf8NoBom
} catch {
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"
$ProjectRoot = "G:\rt_isaaclab_ws\repos\NVIDIA--Isaac-Lab-Unitree_Go2-control"
$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2"
$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$TrainPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task1\task1_train.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\task1"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "win_task1_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task1 Windows skrl PPO SMOKE runner"
Write-Host "============================================================"
Write-Host "This script WILL launch a tiny training run when approved."
Write-Host "ProjectRoot     = $ProjectRoot"
Write-Host "TrainPy         = $TrainPy"
Write-Host "NumEnvs         = $NumEnvs"
Write-Host "TotalEnvSteps   = $TotalEnvSteps"
Write-Host "Rollouts        = $Rollouts"
Write-Host "Device          = $Device"
Write-Host "RunName         = $RunName"
Write-Host "DriverLog       = $DriverLog"
Write-Host "============================================================"

if ($env:GO2_TASK1_WINDOWS_SMOKE_APPROVED -ne "1") {
    Write-Host ""
    Write-Host "[STOP] To actually start smoke training, run first:" -ForegroundColor Yellow
    Write-Host '  $env:GO2_TASK1_WINDOWS_SMOKE_APPROVED = "1"'
    Write-Host ""
    exit 0
}

if (-not (Test-Path $PythonBat)) {
    throw "python.bat not found: $PythonBat"
}
if (-not (Test-Path $TrainPy)) {
    throw "train file not found: $TrainPy"
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_GO2_TASK1_LOG_ROOT = $LogRoot

Set-Location $ProjectRoot

$ArgsList = @(
    $TrainPy,
    "--num-envs", "$NumEnvs",
    "--total-env-steps", "$TotalEnvSteps",
    "--rollouts", "$Rollouts",
    "--learning-epochs", "$LearningEpochs",
    "--mini-batches", "$MiniBatches",
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
$TranscriptStarted = $false
try {
    Start-Transcript -Path $DriverLog -Force | Out-Null
    $TranscriptStarted = $true
} catch {
    Write-Host "[WARN] Start-Transcript failed: $($_.Exception.Message)"
}

& $PythonBat @ArgsList
$exitCode = $LASTEXITCODE

if ($TranscriptStarted) {
    try {
        Stop-Transcript | Out-Null
    } catch {
    }
}
Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task1 Windows smoke finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
