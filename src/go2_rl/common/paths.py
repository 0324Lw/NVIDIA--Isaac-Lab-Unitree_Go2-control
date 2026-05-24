from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_log_root(task_name: str) -> str:
    env_key = f"RT_GO2_{task_name.upper()}_LOG_ROOT"
    if env_key in os.environ:
        return os.environ[env_key]

    if os.name == "nt":
        return rf"G:\rt_isaaclab_ws\logs\unitree_go2_isaaclab_rl\{task_name.lower()}"

    return str(project_root() / "logs" / task_name.lower())


def ensure_dir(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
