from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import ImageGrab


PROJECT_ROOT = Path(r"G:\rt_isaaclab_ws\repos\NVIDIA--Isaac-Lab-Unitree_Go2-control")
ISAACLAB_ROOT = Path(r"G:\rt_isaaclab_ws\repos\IsaacLab_v2.3.2")
LOG_ROOT = Path(r"G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl")

DEFAULT_CROP = "50,125,1430,840"

TASKS: Dict[str, Dict[str, str]] = {
    "task1": {
        "model_name": "go2_task1_model.pt",
        "eval_script": str(PROJECT_ROOT / "scripts" / "windows" / "eval_task1_gui_windows.ps1"),
        "process_key": "task1_model_test.py",
    },
    "task2": {
        "model_name": "go2_task2_model.pt",
        "eval_script": str(PROJECT_ROOT / "scripts" / "windows" / "eval_task2_gui_windows.ps1"),
        "process_key": "task2_model_test.py",
    },
    "task3": {
        "model_name": "go2_task3_model.pt",
        "eval_script": str(PROJECT_ROOT / "scripts" / "windows" / "eval_task3_gui_windows.ps1"),
        "process_key": "task3_model_test.py",
    },
    "task4": {
        "model_name": "go2_task4_model.pt",
        "eval_script": str(PROJECT_ROOT / "scripts" / "windows" / "eval_task4_gui_windows.ps1"),
        "process_key": "task4_model_test.py",
    },
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_crop(crop_text: str) -> Optional[tuple[int, int, int, int]]:
    text = crop_text.strip()
    if not text:
        return None

    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be 'left,top,right,bottom'")

    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop box: {parts}")

    return left, top, right, bottom


def find_latest_final_checkpoint(task_name: str) -> Path:
    cfg = TASKS[task_name]
    task_log_root = LOG_ROOT / task_name
    model_name = cfg["model_name"]

    if not task_log_root.exists():
        raise FileNotFoundError(f"{task_name} log root not found: {task_log_root}")

    candidates = []
    for run_dir in task_log_root.iterdir():
        final_dir = run_dir / "final_checkpoint"
        model = final_dir / model_name
        if model.exists():
            candidates.append((model.stat().st_mtime, final_dir))

    if not candidates:
        raise FileNotFoundError(f"No {task_name} final_checkpoint found under {task_log_root}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1].resolve()


def cleanup_eval_processes(task_names: Optional[Iterable[str]] = None) -> None:
    keys = []
    if task_names is None:
        keys = [cfg["process_key"] for cfg in TASKS.values()]
    else:
        keys = [TASKS[t]["process_key"] for t in task_names]

    joined_keys = '","'.join(keys)

    ps_script = rf"""
$projectRoot = "{PROJECT_ROOT}"
$isaacRoot = "{ISAACLAB_ROOT}"

$patterns = @(
    [regex]::Escape($projectRoot),
    [regex]::Escape($isaacRoot),
    "isaaclab.python",
    "_isaac_sim",
    "Isaac-Sim",
    "{joined_keys}"
)

$names = @("kit.exe", "python.exe", "pythonw.exe", "cmd.exe")

$targets = Get-CimInstance Win32_Process | Where-Object {{
    $cmd = [string]$_.CommandLine

    if ([string]::IsNullOrWhiteSpace($cmd)) {{
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
    try {{
        Write-Host ("[CLEANUP] Stopping PID={{0}}, Name={{1}}" -f $p.ProcessId, $p.Name)
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


def start_eval_process(task_name: str, checkpoint_dir: Path, steps: int) -> subprocess.Popen:
    cfg = TASKS[task_name]
    eval_script = Path(cfg["eval_script"])

    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(eval_script),
        "-Checkpoint",
        str(checkpoint_dir),
        "-NumEnvs",
        "1",
        "-Steps",
        str(steps),
        "-PrintInterval",
        "500",
        "-NoCleanup",
    ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdin=None,
        stdout=None,
        stderr=None,
        shell=False,
    )


def grab_frames(
    duration_s: float,
    capture_fps: int,
    warmup_s: float,
    crop: Optional[tuple[int, int, int, int]],
    max_width: int,
) -> list:
    print(f"[INFO] waiting GUI warmup: {warmup_s:.1f}s", flush=True)
    time.sleep(warmup_s)

    total_frames = int(duration_s * capture_fps)
    interval = 1.0 / max(capture_fps, 1)

    frames = []
    print(
        f"[INFO] start capture: duration={duration_s}s, fps={capture_fps}, "
        f"frames={total_frames}, crop={crop}",
        flush=True,
    )

    next_t = time.time()
    for i in range(total_frames):
        img = ImageGrab.grab(bbox=crop)

        if max_width > 0 and img.width > max_width:
            new_height = int(img.height * (max_width / img.width))
            img = img.resize((max_width, new_height))

        frames.append(img.convert("P", palette=1))

        if (i + 1) % max(capture_fps, 1) == 0:
            print(f"[CAPTURE] {i + 1}/{total_frames}", flush=True)

        next_t += interval
        sleep_time = next_t - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    return frames


def save_gif(frames: list, out_path: Path, capture_fps: int, speed: float) -> None:
    if not frames:
        raise RuntimeError("no frames captured")

    duration_ms = int(1000 / max(capture_fps, 1) / max(speed, 1e-6))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def record_one_gif(
    task_name: str,
    checkpoint_dir: Path,
    index: int,
    duration: float,
    fps: int,
    speed: float,
    warmup: float,
    eval_steps: int,
    crop: Optional[tuple[int, int, int, int]],
    max_width: int,
    output_dir: Path,
) -> Path:
    out_path = output_dir / f"{task_name}_eval_gif_{index:02d}_20s_speed{speed:g}_{now_stamp()}.gif"

    print("")
    print("=" * 100)
    print(f"[START] {task_name} GIF {index:02d}")
    print("=" * 100)
    print(f"checkpoint_dir = {checkpoint_dir}")
    print(f"output_gif     = {out_path}")
    print(f"duration       = {duration}s real capture")
    print(f"fps            = {fps}")
    print(f"speed          = {speed}x")
    print(f"gif_play_time  = {duration / max(speed, 1e-6):.2f}s approx")
    print(f"warmup         = {warmup}s")
    print(f"crop           = {crop if crop else '<full screen>'}")
    print(f"max_width      = {max_width}")
    print("=" * 100)

    cleanup_eval_processes([task_name])
    process = start_eval_process(task_name=task_name, checkpoint_dir=checkpoint_dir, steps=eval_steps)

    try:
        frames = grab_frames(
            duration_s=duration,
            capture_fps=fps,
            warmup_s=warmup,
            crop=crop,
            max_width=max_width,
        )

        save_gif(
            frames=frames,
            out_path=out_path,
            capture_fps=fps,
            speed=speed,
        )

        print("")
        print("=" * 100)
        print(f"[OK] {task_name} GIF saved")
        print(f"path = {out_path}")
        print("=" * 100)

        return out_path

    finally:
        cleanup_eval_processes([task_name])
        try:
            process.terminate()
        except Exception:
            pass

        time.sleep(5)


def parse_tasks(tasks_text: str) -> list[str]:
    raw = tasks_text.strip().lower()

    if raw in ("all", "*"):
        return ["task1", "task2", "task3", "task4"]

    tasks = [x.strip().lower() for x in raw.split(",") if x.strip()]
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}; valid tasks: {list(TASKS)}")

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Go2 Task1/2/3/4 GUI evaluation GIFs on Windows")
    parser.add_argument("--tasks", type=str, default="all", help="all or comma list, e.g. task1,task4")
    parser.add_argument("--count", type=int, default=2, help="GIF count per task")
    parser.add_argument("--duration", type=float, default=12.0, help="real capture duration in seconds")
    parser.add_argument("--fps", type=int, default=10, help="capture fps before speed-up")
    parser.add_argument("--speed", type=float, default=1.5, help="GIF playback speed multiplier")
    parser.add_argument("--warmup", type=float, default=30.0, help="wait seconds before capture")
    parser.add_argument("--eval-steps", type=int, default=30000)
    parser.add_argument("--crop", type=str, default=DEFAULT_CROP, help="crop box: left,top,right,bottom")
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--output-subdir", type=str, default="", help="optional subdir under final_checkpoint")
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()

    if args.cleanup_only:
        cleanup_eval_processes()
        return

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(PROJECT_ROOT)

    tasks = parse_tasks(args.tasks)
    crop = parse_crop(args.crop)

    print("=" * 100)
    print("Go2 universal GIF recorder")
    print("=" * 100)
    print(f"tasks          = {tasks}")
    print(f"count per task = {args.count}")
    print(f"duration       = {args.duration}s real capture")
    print(f"fps            = {args.fps}")
    print(f"speed          = {args.speed}x")
    print(f"warmup         = {args.warmup}s")
    print(f"crop           = {crop if crop else '<full screen>'}")
    print(f"max_width      = {args.max_width}")
    print("=" * 100)

    saved_paths: list[Path] = []

    try:
        for task_name in tasks:
            checkpoint_dir = find_latest_final_checkpoint(task_name)

            if args.output_subdir.strip():
                output_dir = checkpoint_dir / args.output_subdir.strip()
            else:
                output_dir = checkpoint_dir

            for index in range(1, int(args.count) + 1):
                out_path = record_one_gif(
                    task_name=task_name,
                    checkpoint_dir=checkpoint_dir,
                    index=index,
                    duration=float(args.duration),
                    fps=int(args.fps),
                    speed=float(args.speed),
                    warmup=float(args.warmup),
                    eval_steps=int(args.eval_steps),
                    crop=crop,
                    max_width=int(args.max_width),
                    output_dir=output_dir,
                )
                saved_paths.append(out_path)

        print("")
        print("=" * 100)
        print("[DONE] All GIF recordings finished")
        print("=" * 100)
        for p in saved_paths:
            print(p)

    finally:
        cleanup_eval_processes()


if __name__ == "__main__":
    main()