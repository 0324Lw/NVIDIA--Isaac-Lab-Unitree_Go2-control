param(
    [int]$NumEnvs = 1024,
    [Int64]$TotalEnvSteps = 350000000,
    [int]$Rollouts = 64,
    [int]$LearningEpochs = 5,
    [int]$MiniBatches = 8,
    [double]$Lr = 0.0001,
    [double]$MinLr = 0.00002,
    [double]$MaxLr = 0.0003,
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
    $RunName = "win_task1_3090_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task1 Windows RTX 3090 skrl PPO training runner"
Write-Host "============================================================"
Write-Host "ProjectRoot     = $ProjectRoot"
Write-Host "TrainPy         = $TrainPy"
Write-Host "NumEnvs         = $NumEnvs"
Write-Host "TotalEnvSteps   = $TotalEnvSteps"
Write-Host "Rollouts        = $Rollouts"
Write-Host "LearningEpochs  = $LearningEpochs"
Write-Host "MiniBatches     = $MiniBatches"
Write-Host "LR              = $Lr / $MinLr / $MaxLr"
Write-Host "Device          = $Device"
Write-Host "RunName         = $RunName"
Write-Host "DriverLog       = $DriverLog"
Write-Host "============================================================"

if ($env:GO2_TASK1_WINDOWS_TRAIN_APPROVED -ne "1") {
    Write-Host ""
    Write-Host "[STOP] To actually start Windows 3090 training, run first:" -ForegroundColor Yellow
    Write-Host '  $env:GO2_TASK1_WINDOWS_TRAIN_APPROVED = "1"'
    Write-Host ""
    Write-Host "Suggested first formal run:"
    Write-Host "  .\scripts\windows\train_task1_skrl_3090.ps1 -NumEnvs 512 -TotalEnvSteps 20000000"
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
    "--lr", "$Lr",
    "--min-lr", "$MinLr",
    "--max-lr", "$MaxLr",
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
Write-Host "Go2 Task1 Windows RTX 3090 training finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
