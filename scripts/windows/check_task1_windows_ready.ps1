param(
    [string]$ProjectRoot = "G:\rt_isaaclab_ws\projects\unitree_go2_isaaclab_rl",
    [string]$IsaacLabRoot = "G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
$TrainPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task1\task1_train.py"
$EnvPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task1\task1_env.py"
$ConfigPy = Join-Path $ProjectRoot "src\go2_rl\tasks\task1\task1_config.py"

Write-Host ""
Write-Host "============================================================"
Write-Host "Go2 Task1 Windows non-training readiness check"
Write-Host "============================================================"
Write-Host "ProjectRoot  = $ProjectRoot"
Write-Host "IsaacLabRoot = $IsaacLabRoot"
Write-Host "PythonBat    = $PythonBat"
Write-Host "TrainPy      = $TrainPy"
Write-Host "============================================================"

$required = @(
    $ProjectRoot,
    $IsaacLabRoot,
    $PythonBat,
    $TrainPy,
    $EnvPy,
    $ConfigPy
)

$missing = @()
foreach ($p in $required) {
    if (-not (Test-Path $p)) {
        $missing += $p
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "[WARN] Missing required paths:" -ForegroundColor Yellow
    foreach ($m in $missing) {
        Write-Host "  - $m" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Please copy Ubuntu project to:"
    Write-Host "  $ProjectRoot"
    exit 1
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
$env:RT_GO2_TASK1_LOG_ROOT = "G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\task1"

Write-Host ""
Write-Host "[CHECK] Python / torch / isaaclab / skrl import check..."
& $PythonBat -c "import sys; print('[PYTHON]', sys.executable); import torch; print('[TORCH]', torch.__version__, 'cuda=', torch.cuda.is_available()); import isaaclab; print('[ISAACLAB] ok'); import skrl; print('[SKRL]', getattr(skrl, '__version__', 'unknown'))"

if ($LASTEXITCODE -ne 0) {
    throw "Python environment check failed."
}

Write-Host ""
Write-Host "[OK] Windows non-training readiness check passed."
Write-Host "No training has been launched."
