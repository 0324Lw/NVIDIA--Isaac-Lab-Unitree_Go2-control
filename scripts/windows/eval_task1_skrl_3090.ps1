param(
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,

    [int]$NumEnvs = 16,
    [int]$Steps = 2000,
    [string]$Device = "cuda:0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_go2_isaaclab_rl"
$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2"
$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$EvalPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task1\task1_model_test.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\task1"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

$RunName = "win_task1_eval_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task1 Windows model evaluation runner"
Write-Host "============================================================"
Write-Host "Checkpoint = $Checkpoint"
Write-Host "NumEnvs    = $NumEnvs"
Write-Host "Steps      = $Steps"
Write-Host "Device     = $Device"
Write-Host "DriverLog  = $DriverLog"
Write-Host "============================================================"

if (-not (Test-Path $PythonBat)) {
    throw "python.bat not found: $PythonBat"
}
if (-not (Test-Path $EvalPy)) {
    throw "eval file not found: $EvalPy"
}
if (-not (Test-Path $Checkpoint)) {
    throw "checkpoint not found: $Checkpoint"
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_GO2_TASK1_LOG_ROOT = $LogRoot

Set-Location $ProjectRoot

$ArgsList = @(
    $EvalPy,
    "--checkpoint", "$Checkpoint",
    "--num-envs", "$NumEnvs",
    "--steps", "$Steps",
    "--print-interval", "100",
    "--headless",
    "--device", "$Device"
)

& $PythonBat @ArgsList 2>&1 | Tee-Object -FilePath $DriverLog

$exitCode = $LASTEXITCODE
Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task1 Windows model eval finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
