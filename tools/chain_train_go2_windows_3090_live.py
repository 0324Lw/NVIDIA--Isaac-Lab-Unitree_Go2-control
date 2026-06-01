from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(r"G:\rt_isaaclab_ws\repos\NVIDIA--Isaac-Lab-Unitree_Go2-control")
ISAACLAB_ROOT = Path(r"G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2")
PYTHON_BAT = ISAACLAB_ROOT / "_isaac_sim" / "python.bat"

LOG_ROOT = Path(r"G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl")
CHAIN_LOG_ROOT = LOG_ROOT / "windows_chain"

TASK1_TRAIN_PY = PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task1" / "task1_train.py"
TASK2_TRAIN_PY = PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task2" / "task2_train.py"
TASK3_TRAIN_PY = PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task3" / "task3_train.py"
TASK4_TRAIN_PY = PROJECT_ROOT / "src" / "go2_rl" / "tasks" / "task4" / "task4_train.py"

TASK1_LOG_ROOT = LOG_ROOT / "task1"
TASK2_LOG_ROOT = LOG_ROOT / "task2"
TASK3_LOG_ROOT = LOG_ROOT / "task3"
TASK4_LOG_ROOT = LOG_ROOT / "task4"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ChainLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def build_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"

    src = str(PROJECT_ROOT / "src")
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + old_pythonpath if old_pythonpath else "")

    return env


def cleanup_runtime_processes(logger: ChainLogger, reason: str) -> None:
    logger.write(f"清理进程: {reason}")

    self_pid = os.getpid()

    ps_script = rf"""
$projectRoot = "{PROJECT_ROOT}"
$isaacRoot = "{ISAACLAB_ROOT}"
$selfPid = {self_pid}

$patterns = @(
    [regex]::Escape($projectRoot),
    [regex]::Escape($isaacRoot),
    "isaaclab.python.headless.kit",
    "_isaac_sim",
    "Isaac-Sim",
    "task1_train.py",
    "task2_train.py",
    "task3_train.py",
    "task4_train.py"
)

$names = @("kit.exe", "python.exe", "pythonw.exe", "cmd.exe")

$targets = Get-CimInstance Win32_Process | Where-Object {{
    if ($_.ProcessId -eq $selfPid) {{
        return $false
    }}

    $cmd = [string]$_.CommandLine
    if ([string]::IsNullOrWhiteSpace($cmd)) {{
        return $false
    }}

    if ($cmd -match "chain_train_go2_windows_3090_live.py") {{
        return $false
    }}

    if ($names -notcontains $_.Name) {{
        return $false
    }}

    foreach ($pat in $patterns) {{
        if ($cmd -match $pat) {{
            return $true
        }}
    }}

    return $false
}}

foreach ($p in $targets) {{
    Write-Host ("[CLEANUP] Stopping PID={{0}}, Name={{1}}" -f $p.ProcessId, $p.Name)
    try {{
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
    }} catch {{
        Write-Host ("[WARN] Failed to stop PID={{0}}: {{1}}" -f $p.ProcessId, $_.Exception.Message)
    }}
}}
"""

    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=False,
    )

    time.sleep(5)


def wait_between_tasks(logger: ChainLogger, seconds: int, next_task_name: str) -> None:
    logger.write(f"等待 {seconds} 秒后启动 {next_task_name} ...")
    time.sleep(seconds)


def expected_final_dir(task_log_root: Path, run_name: str) -> Path:
    return task_log_root / run_name / "final_checkpoint"


def expected_final_model_path(task_name: str, task_log_root: Path, run_name: str) -> Path:
    task_key = task_name.lower()
    return task_log_root / run_name / "final_checkpoint" / f"go2_{task_key}_model.pt"


def find_task_final_checkpoint(
    logger: ChainLogger,
    task_name: str,
    task_log_root: Path,
    expected_run_name: str = "",
    preferred_path: str = "",
) -> Path:
    task_key = task_name.lower()
    model_name = f"go2_{task_key}_model.pt"

    if preferred_path.strip():
        p = Path(preferred_path)
        if not p.exists():
            raise FileNotFoundError(f"指定的 {task_name} final_checkpoint 不存在: {p}")
        model = p / model_name
        if not model.exists():
            raise FileNotFoundError(f"{model_name} 不存在: {model}")
        return p.resolve()

    if expected_run_name.strip():
        expected = task_log_root / expected_run_name / "final_checkpoint"
        expected_model = expected / model_name
        if expected_model.exists():
            return expected.resolve()

    candidates = []
    if task_log_root.exists():
        for run_dir in task_log_root.iterdir():
            final_dir = run_dir / "final_checkpoint"
            model = final_dir / model_name
            if model.exists():
                candidates.append((model.stat().st_mtime, final_dir))

    if not candidates:
        raise FileNotFoundError(
            f"没有在 {task_log_root} 下找到 {task_name} final_checkpoint/{model_name}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    found = candidates[0][1].resolve()
    logger.write(f"自动选择最新 {task_name} final checkpoint: {found}")
    return found


def run_training_live(
    logger: ChainLogger,
    task_name: str,
    train_py: Path,
    task_log_root: Path,
    run_name: str,
    total_env_steps: int,
    num_envs: int,
    rollouts: int,
    learning_epochs: int,
    mini_batches: int,
    summary_interval: int,
    device: str,
    lr: float,
    min_lr: float,
    max_lr: float,
    entropy_coef: float,
    init_log_std: float,
    final_exit_grace_seconds: int,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    require_path(PYTHON_BAT, "Isaac Sim python.bat")
    require_path(train_py, f"{task_name} train script")

    task_log_root.mkdir(parents=True, exist_ok=True)

    final_dir = expected_final_dir(task_log_root, run_name)
    final_model = expected_final_model_path(task_name, task_log_root, run_name)

    args: List[str] = [
        str(PYTHON_BAT),
        "-u",
        str(train_py),
        "--num-envs",
        str(num_envs),
        "--total-env-steps",
        str(total_env_steps),
        "--rollouts",
        str(rollouts),
        "--learning-epochs",
        str(learning_epochs),
        "--mini-batches",
        str(mini_batches),
        "--lr",
        str(lr),
        "--min-lr",
        str(min_lr),
        "--max-lr",
        str(max_lr),
        "--summary-interval",
        str(summary_interval),
        "--tb-log-interval-steps",
        "500",
        "--skrl-write-interval",
        "5000000",
        "--skrl-checkpoint-interval",
        "0",
        "--save-freq-env-steps",
        "50000000",
        "--log-root",
        str(task_log_root),
        "--run-name",
        run_name,
        "--headless",
        "--device",
        device,
        "--entropy-coef",
        str(entropy_coef),
        "--init-log-std",
        str(init_log_std),
    ]

    if extra_args:
        args.extend(str(x) for x in extra_args)

    logger.write("=" * 80)
    logger.write(f"START {task_name}")
    logger.write(f"TrainPy        = {train_py}")
    logger.write(f"RunName        = {run_name}")
    logger.write(f"NumEnvs        = {num_envs}")
    logger.write(f"TotalEnvSteps  = {total_env_steps}")
    logger.write(f"Rollouts       = {rollouts}")
    logger.write(f"LearningEpochs = {learning_epochs}")
    logger.write(f"MiniBatches    = {mini_batches}")
    logger.write(f"LR             = {lr} / {min_lr} / {max_lr}")
    logger.write(f"EntropyCoef    = {entropy_coef}")
    logger.write(f"InitLogStd     = {init_log_std}")
    logger.write(f"SummaryEvery   = {summary_interval} PPO updates")
    logger.write(f"FinalDir       = {final_dir}")
    logger.write(f"FinalModel     = {final_model}")
    logger.write("=" * 80)

    print("", flush=True)
    print("[LIVE COMMAND]", flush=True)
    print(" ".join(f'"{x}"' if " " in x else x for x in args), flush=True)
    print("", flush=True)

    process = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        env=build_env(),
        stdin=None,
        stdout=None,
        stderr=None,
        shell=False,
    )

    final_seen_at: Optional[float] = None

    try:
        while True:
            return_code = process.poll()

            if return_code is not None:
                logger.write(f"FINISH {task_name} with exit code: {return_code}")

                if return_code != 0:
                    cleanup_runtime_processes(logger, f"{task_name} 非零退出")
                    raise RuntimeError(f"{task_name} failed with exit code {return_code}")

                return

            # Windows 下 Isaac Sim / kit.exe 训练结束后可能不自动退出。
            # 一旦 final_checkpoint 已保存，就等待 grace 秒，然后强制清理残留进程并视为成功。
            if final_model.exists():
                if final_seen_at is None:
                    final_seen_at = time.time()
                    logger.write(f"{task_name} final checkpoint detected: {final_model}")
                    logger.write(
                        f"等待 {final_exit_grace_seconds} 秒让 Isaac/kit 自行退出；如果不退出，将强制清理并进入下一任务。"
                    )

                elapsed_after_final = time.time() - final_seen_at
                if elapsed_after_final >= final_exit_grace_seconds:
                    logger.write(
                        f"{task_name} final checkpoint 已保存，但进程仍未退出。现在强制终止残留进程，并判定该任务完成。"
                    )
                    cleanup_runtime_processes(logger, f"{task_name} final checkpoint saved")
                    return

            time.sleep(2)

    except KeyboardInterrupt:
        logger.write(f"[WARN] 收到 Ctrl+C，正在清理 {task_name} 运行进程。")
        cleanup_runtime_processes(logger, f"{task_name} keyboard interrupt")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live continuous training chain for Unitree Go2 on Windows RTX 3090 Ti"
    )

    parser.add_argument("--num-envs", type=int, default=4096)

    parser.add_argument("--task1-steps", type=int, default=600_000_000)
    parser.add_argument("--task2-steps", type=int, default=1_500_000_000)
    parser.add_argument("--task3-steps", type=int, default=1_500_000_000)
    parser.add_argument("--task4-steps", type=int, default=1_200_000_000)

    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--learning-epochs", type=int, default=5)
    parser.add_argument("--mini-batches", type=int, default=16)

    parser.add_argument("--summary-interval", type=int, default=5)

    parser.add_argument("--sleep-between-tasks", type=int, default=60)
    parser.add_argument("--final-exit-grace-seconds", type=int, default=45)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run-prefix", type=str, default="")

    parser.add_argument("--skip-task1", action="store_true")
    parser.add_argument("--skip-task2", action="store_true")
    parser.add_argument("--skip-task3", action="store_true")
    parser.add_argument("--skip-task4", action="store_true")

    parser.add_argument("--task1-final-checkpoint", type=str, default="")
    parser.add_argument("--cleanup-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for p, name in [
        (PROJECT_ROOT, "PROJECT_ROOT"),
        (ISAACLAB_ROOT, "ISAACLAB_ROOT"),
        (PYTHON_BAT, "PYTHON_BAT"),
        (TASK1_TRAIN_PY, "TASK1_TRAIN_PY"),
        (TASK2_TRAIN_PY, "TASK2_TRAIN_PY"),
        (TASK3_TRAIN_PY, "TASK3_TRAIN_PY"),
        (TASK4_TRAIN_PY, "TASK4_TRAIN_PY"),
    ]:
        require_path(p, name)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    CHAIN_LOG_ROOT.mkdir(parents=True, exist_ok=True)

    run_prefix = args.run_prefix.strip()
    if not run_prefix:
        run_prefix = f"win3090_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    master_log = CHAIN_LOG_ROOT / f"{run_prefix}_master.log"
    logger = ChainLogger(master_log)

    if args.cleanup_only:
        cleanup_runtime_processes(logger, "cleanup-only")
        return

    task1_run_name = f"{run_prefix}_task1_4096_600m"
    task2_run_name = f"{run_prefix}_task2_from_task1_4096_1500m"
    task3_run_name = f"{run_prefix}_task3_4096_1500m"
    task4_run_name = f"{run_prefix}_task4_from_task1_4096_1200m"

    logger.write("=" * 80)
    logger.write("Go2 Windows 3090 continuous LIVE training chain")
    logger.write("=" * 80)
    logger.write(f"ProjectRoot = {PROJECT_ROOT}")
    logger.write(f"IsaacLabRoot = {ISAACLAB_ROOT}")
    logger.write(f"PythonBat = {PYTHON_BAT}")
    logger.write("GPU target = RTX 3090 Ti / cuda:0")
    logger.write(f"NumEnvs = {args.num_envs}")
    logger.write(f"Task1TotalEnvSteps = {args.task1_steps}")
    logger.write(f"Task2TotalEnvSteps = {args.task2_steps}")
    logger.write(f"Task3TotalEnvSteps = {args.task3_steps}")
    logger.write(f"Task4TotalEnvSteps = {args.task4_steps}")
    logger.write(f"SummaryInterval = {args.summary_interval}")
    logger.write(f"FinalExitGraceSeconds = {args.final_exit_grace_seconds}")
    logger.write(f"MasterLog = {master_log}")
    logger.write("=" * 80)

    cleanup_runtime_processes(logger, "before live training chain")

    task1_final = ""

    try:
        # ====================================================
        # Task1: no pretrained model, 600M
        # ====================================================
        if not args.skip_task1:
            run_training_live(
                logger=logger,
                task_name="Task1",
                train_py=TASK1_TRAIN_PY,
                task_log_root=TASK1_LOG_ROOT,
                run_name=task1_run_name,
                total_env_steps=args.task1_steps,
                num_envs=args.num_envs,
                rollouts=args.rollouts,
                learning_epochs=args.learning_epochs,
                mini_batches=args.mini_batches,
                summary_interval=args.summary_interval,
                device=args.device,
                lr=1.0e-4,
                min_lr=2.0e-5,
                max_lr=3.0e-4,
                entropy_coef=0.003,
                init_log_std=-1.0,
                final_exit_grace_seconds=args.final_exit_grace_seconds,
            )

            cleanup_runtime_processes(logger, "after Task1")
            wait_between_tasks(logger, args.sleep_between_tasks, "Task2")
        else:
            logger.write("[SKIP] Task1 skipped by user.")

        # Task1 final checkpoint is needed by Task2 and Task4.
        task1_final_path = find_task_final_checkpoint(
            logger=logger,
            task_name="Task1",
            task_log_root=TASK1_LOG_ROOT,
            expected_run_name=task1_run_name,
            preferred_path=args.task1_final_checkpoint,
        )
        task1_final = str(task1_final_path)
        logger.write(f"Task1 final checkpoint selected: {task1_final}")

        # ====================================================
        # Task2: pretrained from Task1, 1500M
        # ====================================================
        if not args.skip_task2:
            run_training_live(
                logger=logger,
                task_name="Task2",
                train_py=TASK2_TRAIN_PY,
                task_log_root=TASK2_LOG_ROOT,
                run_name=task2_run_name,
                total_env_steps=args.task2_steps,
                num_envs=args.num_envs,
                rollouts=args.rollouts,
                learning_epochs=args.learning_epochs,
                mini_batches=args.mini_batches,
                summary_interval=args.summary_interval,
                device=args.device,
                lr=5.0e-5,
                min_lr=2.0e-5,
                max_lr=1.2e-4,
                entropy_coef=0.003,
                init_log_std=-1.25,
                final_exit_grace_seconds=args.final_exit_grace_seconds,
                extra_args=[
                    "--pretrained-task1",
                    task1_final,
                    "--pretrained-log-std",
                    "-1.65",
                    "--start-k",
                    "0.0",
                ],
            )

            cleanup_runtime_processes(logger, "after Task2")
            wait_between_tasks(logger, args.sleep_between_tasks, "Task3")
        else:
            logger.write("[SKIP] Task2 skipped by user.")

        # ====================================================
        # Task3: no pretrained model, 1500M
        # ====================================================
        if not args.skip_task3:
            run_training_live(
                logger=logger,
                task_name="Task3",
                train_py=TASK3_TRAIN_PY,
                task_log_root=TASK3_LOG_ROOT,
                run_name=task3_run_name,
                total_env_steps=args.task3_steps,
                num_envs=args.num_envs,
                rollouts=args.rollouts,
                learning_epochs=args.learning_epochs,
                mini_batches=args.mini_batches,
                summary_interval=args.summary_interval,
                device=args.device,
                lr=3.0e-5,
                min_lr=1.5e-5,
                max_lr=7.0e-5,
                entropy_coef=0.004,
                init_log_std=-1.35,
                final_exit_grace_seconds=args.final_exit_grace_seconds,
                extra_args=["--start-k", "0.0"],
            )

            cleanup_runtime_processes(logger, "after Task3")
            wait_between_tasks(logger, args.sleep_between_tasks, "Task4")
        else:
            logger.write("[SKIP] Task3 skipped by user.")

        # ====================================================
        # Task4: pretrained from Task1, 1200M
        # ====================================================
        if not args.skip_task4:
            logger.write(f"Task4 will use Task1 final checkpoint: {task1_final}")

            run_training_live(
                logger=logger,
                task_name="Task4",
                train_py=TASK4_TRAIN_PY,
                task_log_root=TASK4_LOG_ROOT,
                run_name=task4_run_name,
                total_env_steps=args.task4_steps,
                num_envs=args.num_envs,
                rollouts=args.rollouts,
                learning_epochs=args.learning_epochs,
                mini_batches=args.mini_batches,
                summary_interval=args.summary_interval,
                device=args.device,
                lr=5.0e-5,
                min_lr=2.0e-5,
                max_lr=1.2e-4,
                entropy_coef=0.003,
                init_log_std=-1.35,
                final_exit_grace_seconds=args.final_exit_grace_seconds,
                extra_args=[
                    "--pretrained-task1",
                    task1_final,
                    "--start-k",
                    "0.0",
                ],
            )

            cleanup_runtime_processes(logger, "after Task4")
        else:
            logger.write("[SKIP] Task4 skipped by user.")

        logger.write("=" * 80)
        logger.write("[DONE] Continuous live training chain finished successfully.")
        logger.write(f"Task1Run = {task1_run_name}")
        logger.write(f"Task2Run = {task2_run_name}")
        logger.write(f"Task3Run = {task3_run_name}")
        logger.write(f"Task4Run = {task4_run_name}")
        logger.write(f"Task1FinalCheckpoint = {task1_final}")
        logger.write("=" * 80)

    except KeyboardInterrupt:
        logger.write("[WARN] KeyboardInterrupt received. Cleaning runtime processes.")
        cleanup_runtime_processes(logger, "keyboard interrupt")
        raise
    except Exception as exc:
        logger.write(f"[FATAL] Continuous live training chain aborted: {type(exc).__name__}: {exc}")
        cleanup_runtime_processes(logger, "fatal abort")
        raise


if __name__ == "__main__":
    main()