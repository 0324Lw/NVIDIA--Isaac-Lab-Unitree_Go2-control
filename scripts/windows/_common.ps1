# Copyright (c) 2026
# Unitree Go2 Scripts: Windows 公共脚本工具。
#
# 本文件为 Windows PowerShell 运行脚本提供公共函数，不直接启动训练、测试或评估。
# 主要职责:
#   1. 设置 Windows 控制台 UTF-8、Python UTF-8、无缓冲输出和 faulthandler；
#   2. 根据脚本位置、参数或环境变量解析 ProjectRoot、IsaacLabRoot、PythonBat；
#   3. 设置 PYTHONPATH，使 src/go2_rl 可以被直接导入；
#   4. 解析任务日志目录；
#   5. 检查 Python / torch / IsaacLab / skrl 运行环境；
#   6. 解析 checkpoint 文件或从日志目录查找最新 checkpoint；
#   7. 提供统一的控制台输出、driver log 和可选进程清理函数。
#
# 路径设计:
#   ProjectRoot 优先来自脚本参数，其次由脚本位置向上推导；
#   IsaacLabRoot 优先来自脚本参数，其次来自 ISAACLAB_ROOT 环境变量；
#   LogRoot 优先来自脚本参数，其次来自 RT_GO2_TASK*_LOG_ROOT，最后落到项目内 logs/task*。
#   本文件不写入个人绝对路径，不绑定具体硬件型号。
#
# 使用方式:
#   . "$PSScriptRoot\_common.ps1"
#   Set-Go2WindowsRuntime
#
# Unitree Go2 Scripts: Windows common script utilities.
#
# This file provides shared functions for Windows PowerShell runtime scripts. It
# does not launch training, testing, or evaluation directly.
# Main responsibilities:
#   1. Configure UTF-8 console output, Python UTF-8 mode, unbuffered output, and faulthandler;
#   2. Resolve ProjectRoot, IsaacLabRoot, and PythonBat from script location, parameters, or environment variables;
#   3. Set PYTHONPATH so src/go2_rl can be imported directly;
#   4. Resolve task log directories;
#   5. Check the Python / torch / IsaacLab / skrl runtime environment;
#   6. Resolve checkpoint files or find the latest checkpoint from log directories;
#   7. Provide unified console output, driver logging, and optional process cleanup helpers.
#
# Path design:
#   ProjectRoot is resolved from an explicit parameter first, then from the script location;
#   IsaacLabRoot is resolved from an explicit parameter first, then from ISAACLAB_ROOT;
#   LogRoot is resolved from an explicit parameter first, then RT_GO2_TASK*_LOG_ROOT, then logs/task* under the project root.
#   No personal absolute path or hardware-specific name is stored here.
#
# Usage:
#   . "$PSScriptRoot\_common.ps1"
#   Set-Go2WindowsRuntime

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Go2Info {
    param([string]$Message)
    Write-Host "[INFO] $Message"
}

function Write-Go2Ok {
    param([string]$Message)
    Write-Host "[OK] $Message"
}

function Write-Go2Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Go2Fail {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

function Set-Go2WindowsRuntime {
    try {
        chcp 65001 | Out-Null
        $Utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [Console]::InputEncoding = $Utf8NoBom
        [Console]::OutputEncoding = $Utf8NoBom
        $script:OutputEncoding = $Utf8NoBom
    } catch {
        Write-Go2Warn "Failed to configure UTF-8 console encoding: $($_.Exception.Message)"
    }

    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONFAULTHANDLER = "1"
}

function Resolve-Go2ProjectRoot {
    param(
        [string]$ProjectRoot = "",
        [string]$ScriptRoot = $PSScriptRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($ProjectRoot)) {
        return (Resolve-Path $ProjectRoot).Path
    }

    $candidate = Resolve-Path (Join-Path $ScriptRoot "..\..")
    return $candidate.Path
}

function Resolve-Go2IsaacLabRoot {
    param([string]$IsaacLabRoot = "")

    if (-not [string]::IsNullOrWhiteSpace($IsaacLabRoot)) {
        if (-not (Test-Path $IsaacLabRoot)) {
            throw "IsaacLabRoot not found: $IsaacLabRoot"
        }
        return (Resolve-Path $IsaacLabRoot).Path
    }

    if (-not [string]::IsNullOrWhiteSpace($env:ISAACLAB_ROOT)) {
        if (-not (Test-Path $env:ISAACLAB_ROOT)) {
            throw "ISAACLAB_ROOT is set but not found: $env:ISAACLAB_ROOT"
        }
        return (Resolve-Path $env:ISAACLAB_ROOT).Path
    }

    throw "IsaacLabRoot is required. Pass -IsaacLabRoot or set ISAACLAB_ROOT."
}

function Resolve-Go2PythonBat {
    param([string]$IsaacLabRoot)

    $pythonBat = Join-Path $IsaacLabRoot "_isaac_sim\python.bat"
    if (-not (Test-Path $pythonBat)) {
        throw "python.bat not found: $pythonBat"
    }
    return $pythonBat
}

function Set-Go2PythonPath {
    param([string]$ProjectRoot)

    $src = Join-Path $ProjectRoot "src"
    if (-not (Test-Path $src)) {
        throw "src directory not found: $src"
    }

    $env:PYTHONPATH = "$src;$env:PYTHONPATH"
}

function Resolve-Go2LogRoot {
    param(
        [string]$ProjectRoot,
        [string]$TaskName,
        [string]$LogRoot = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($LogRoot)) {
        New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
        return (Resolve-Path $LogRoot).Path
    }

    $envName = "RT_GO2_$($TaskName.ToUpper())_LOG_ROOT"
    $envValue = [Environment]::GetEnvironmentVariable($envName)

    if (-not [string]::IsNullOrWhiteSpace($envValue)) {
        New-Item -ItemType Directory -Force -Path $envValue | Out-Null
        return (Resolve-Path $envValue).Path
    }

    $defaultRoot = Join-Path $ProjectRoot ("logs\" + $TaskName.ToLower())
    New-Item -ItemType Directory -Force -Path $defaultRoot | Out-Null
    return (Resolve-Path $defaultRoot).Path
}

function Test-Go2RequiredPath {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

function Test-Go2PythonStack {
    param(
        [string]$PythonBat,
        [switch]$RequireIsaacLab,
        [switch]$RequireSkrl
    )

    $lines = @(
        "import os, sys",
        "print('[CHECK] Python:', sys.executable)",
        "import torch",
        "print('[CHECK] torch:', torch.__version__, 'cuda=', torch.cuda.is_available())"
    )

    if ($RequireIsaacLab) {
        $lines += "import isaaclab; print('[CHECK] isaaclab: ok')"
    }

    if ($RequireSkrl) {
        $lines += "import skrl; print('[CHECK] skrl:', getattr(skrl, '__version__', 'unknown'))"
    }

    & $PythonBat -c ($lines -join "; ")
    if ($LASTEXITCODE -ne 0) {
        throw "Python runtime stack check failed."
    }
}

function Find-LatestGo2Checkpoint {
    param(
        [string]$LogRoot,
        [string]$ModelFileName
    )

    if (-not (Test-Path $LogRoot)) {
        throw "Log root not found: $LogRoot"
    }

    $candidates = Get-ChildItem $LogRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $final = Join-Path $_.FullName "final_checkpoint"
            $model = Join-Path $final $ModelFileName
            if (Test-Path $model) {
                [PSCustomObject]@{
                    ModelPath = $model
                    LastWriteTime = (Get-Item $model).LastWriteTime
                }
            }
        } |
        Sort-Object LastWriteTime -Descending

    if ($null -eq $candidates -or $candidates.Count -eq 0) {
        throw "No final_checkpoint with $ModelFileName found under $LogRoot"
    }

    return $candidates[0].ModelPath
}

function Resolve-Go2Checkpoint {
    param(
        [string]$Checkpoint = "",
        [string]$LogRoot,
        [string]$ModelFileName
    )

    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        return Find-LatestGo2Checkpoint -LogRoot $LogRoot -ModelFileName $ModelFileName
    }

    if (-not (Test-Path $Checkpoint)) {
        throw "Checkpoint path not found: $Checkpoint"
    }

    $item = Get-Item $Checkpoint
    if ($item.PSIsContainer) {
        $model = Join-Path $item.FullName $ModelFileName
        if (-not (Test-Path $model)) {
            throw "Model file not found under checkpoint directory: $model"
        }
        return $model
    }

    return $item.FullName
}

function Start-Go2DriverTranscript {
    param(
        [string]$LogRoot,
        [string]$RunName
    )

    $driverLogRoot = Join-Path $LogRoot "windows_driver"
    New-Item -ItemType Directory -Force -Path $driverLogRoot | Out-Null

    $driverLog = Join-Path $driverLogRoot ($RunName + "_driver.log")

    try {
        Start-Transcript -Path $driverLog -Force | Out-Null
        return @{
            Started = $true
            Path = $driverLog
        }
    } catch {
        Write-Go2Warn "Start-Transcript failed: $($_.Exception.Message)"
        return @{
            Started = $false
            Path = $driverLog
        }
    }
}

function Stop-Go2DriverTranscript {
    param([bool]$Started)

    if ($Started) {
        try {
            Stop-Transcript | Out-Null
        } catch {
        }
    }
}

function Confirm-Go2RunApproved {
    param(
        [string]$Kind,
        [string]$TaskName
    )

    $taskUpper = $TaskName.ToUpper()
    $kindUpper = $Kind.ToUpper()
    $specificName = "GO2_${taskUpper}_WINDOWS_${kindUpper}_APPROVED"
    $genericName = "GO2_WINDOWS_${kindUpper}_APPROVED"

    $specificValue = [Environment]::GetEnvironmentVariable($specificName)
    $genericValue = [Environment]::GetEnvironmentVariable($genericName)

    if ($specificValue -eq "1" -or $genericValue -eq "1") {
        return
    }

    Write-Go2Warn "This script will start Windows $TaskName $Kind run."
    Write-Go2Warn "Set one of the following environment variables before running:"
    Write-Host "  `$env:$specificName = `"1`""
    Write-Host "  `$env:$genericName = `"1`""
    exit 0
}

function Stop-Go2IsaacProcesses {
    param(
        [string]$ProjectRoot,
        [string]$IsaacLabRoot,
        [string]$PythonFileName,
        [string]$Reason = "cleanup"
    )

    Write-Go2Info "Process cleanup: $Reason"

    $patterns = @(
        [regex]::Escape($ProjectRoot),
        [regex]::Escape($IsaacLabRoot),
        "isaaclab.python",
        "_isaac_sim",
        "Isaac-Sim",
        [regex]::Escape($PythonFileName)
    )

    $names = @("kit.exe", "python.exe", "pythonw.exe", "cmd.exe")
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $cmd = [string]$_.CommandLine
        if ([string]::IsNullOrWhiteSpace($cmd)) {
            return $false
        }
        if ($names -notcontains $_.Name) {
            return $false
        }
        foreach ($pat in $patterns) {
            if ($cmd -match $pat) {
                return $true
            }
        }
        return $false
    }

    foreach ($p in $targets) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Go2Warn "Cannot stop PID $($p.ProcessId)"
        }
    }
}
