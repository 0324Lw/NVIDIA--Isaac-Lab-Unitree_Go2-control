param(
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

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_go2_isaaclab_rl"
$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2"
$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$TrainPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task2\task2_train.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\task2"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "win_task2_3090_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task2 Windows RTX 3090 skrl PPO training runner"
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
Write-Host "PretrainedTask1 = $PretrainedTask1"
Write-Host "Resume          = $Resume"
Write-Host "DriverLog       = $DriverLog"
Write-Host "============================================================"

if ($env:GO2_TASK2_WINDOWS_TRAIN_APPROVED -ne "1") {
    Write-Host ""
    Write-Host "[STOP] To actually start Windows Task2 training, run first:" -ForegroundColor Yellow
    Write-Host '  $env:GO2_TASK2_WINDOWS_TRAIN_APPROVED = "1"'
    Write-Host ""
    Write-Host "Suggested first formal run:"
    Write-Host "  .\scripts\windows\train_task2_skrl_3090.ps1 -NumEnvs 512 -TotalEnvSteps 20000000 -PretrainedTask1 <task1_ckpt>"
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
$env:RT_GO2_TASK2_LOG_ROOT = $LogRoot

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

& $PythonBat @ArgsList 2>&1 | Tee-Object -FilePath $DriverLog

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task2 Windows RTX 3090 training finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
