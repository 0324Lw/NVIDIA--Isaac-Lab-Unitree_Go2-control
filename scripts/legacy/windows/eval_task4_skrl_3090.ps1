param(
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,

    [int]$NumEnvs = 16,
    [int]$Steps = 3000,
    [double]$StartK = 0.30,
    [string]$Device = "cuda:0"
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
$EvalPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task4\task4_model_test.py"
$LogRoot = "G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\task4"
$DriverLogRoot = Join-Path $LogRoot "windows_driver"

New-Item -ItemType Directory -Force -Path $DriverLogRoot | Out-Null

$RunName = "win_task4_eval_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$DriverLog = Join-Path $DriverLogRoot ($RunName + "_driver.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task4 Windows Teacher model evaluation runner"
Write-Host "============================================================"
Write-Host "Checkpoint = $Checkpoint"
Write-Host "NumEnvs    = $NumEnvs"
Write-Host "Steps      = $Steps"
Write-Host "StartK     = $StartK"
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
$env:RT_GO2_TASK4_LOG_ROOT = $LogRoot

Set-Location $ProjectRoot

$ArgsList = @(
    $EvalPy,
    "--checkpoint", "$Checkpoint",
    "--num-envs", "$NumEnvs",
    "--steps", "$Steps",
    "--start-k", "$StartK",
    "--print-interval", "100",
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
Write-Host "Go2 Task4 Windows Teacher model eval finished with exit code: $exitCode"
Write-Host "Driver log: $DriverLog"
Write-Host "============================================================"

exit $exitCode
