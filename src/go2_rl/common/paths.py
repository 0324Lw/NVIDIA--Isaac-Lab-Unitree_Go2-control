# Copyright (c) 2026
# Unitree Go2 Common: 路径解析工具。
#
# 本文件提供项目根目录、本地路径配置和日志目录解析函数。
# 本文件不依赖 IsaacLab，不启动 AppLauncher，也不创建训练环境。
#
# 路径优先级:
#   1. 显式参数；
#   2. 任务专用环境变量 RT_GO2_TASK*_LOG_ROOT；
#   3. 通用环境变量 RT_GO2_LOG_ROOT；
#   4. local_paths.yaml 中的 log_roots / task_log_roots / log_root；
#   5. 项目内 logs/<task_name>。
#
# 本文件不写入个人绝对路径，不绑定具体硬件型号。
#
# Unitree Go2 Common: path resolution utilities.
#
# This file provides project-root, local path configuration, and log-root
# resolution helpers. It does not depend on IsaacLab, launch AppLauncher, or
# create training environments.
#
# Path priority:
#   1. Explicit argument;
#   2. Task-specific environment variable RT_GO2_TASK*_LOG_ROOT;
#   3. Generic environment variable RT_GO2_LOG_ROOT;
#   4. log_roots / task_log_roots / log_root from local_paths.yaml;
#   5. logs/<task_name> under the project root.
#
# No personal absolute path or hardware-specific name is stored here.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def project_root() -> Path:
    """Return the repository root inferred from src/go2_rl/common/paths.py."""

    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> str:
    """Create a directory when needed and return its resolved string path."""

    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _read_yaml_if_available(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml
    except Exception:
        return {}

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def load_local_paths(explicit: str | Path | None = None) -> Dict[str, Any]:
    """Load optional local path settings.

    The real local file should not be committed. The example file remains only
    documentation for users.
    """

    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())

    candidates.extend(
        [
            project_root() / "local_paths.yaml",
            project_root() / "configs" / "local_paths.yaml",
        ]
    )

    for path in candidates:
        data = _read_yaml_if_available(path)
        if data:
            return data

    return {}


def _task_env_key(task_name: str) -> str:
    task = str(task_name).strip().upper().replace("-", "_")
    return f"RT_GO2_{task}_LOG_ROOT"


def _lookup_task_log_root(local_paths: Mapping[str, Any], task_name: str) -> Optional[str]:
    task = str(task_name).strip().lower()

    for section_name in ("log_roots", "task_log_roots", "logs"):
        section = local_paths.get(section_name)
        if isinstance(section, Mapping):
            value = section.get(task)
            if value:
                return str(value)

    return None


def resolve_log_root(
    task_name: str,
    explicit: str | Path | None = None,
    local_paths_file: str | Path | None = None,
) -> str:
    """Resolve a task log root without using personal absolute paths."""

    task = str(task_name).strip().lower()
    if not task:
        raise ValueError("task_name must be a non-empty string")

    if explicit is not None and str(explicit).strip():
        return ensure_dir(explicit)

    task_env = os.environ.get(_task_env_key(task))
    if task_env:
        return ensure_dir(task_env)

    generic_env = os.environ.get("RT_GO2_LOG_ROOT")
    if generic_env:
        return ensure_dir(Path(generic_env) / task)

    local_paths = load_local_paths(local_paths_file)

    task_local = _lookup_task_log_root(local_paths, task)
    if task_local:
        return ensure_dir(task_local)

    generic_local = local_paths.get("log_root")
    if generic_local:
        return ensure_dir(Path(str(generic_local)) / task)

    return ensure_dir(project_root() / "logs" / task)


def default_log_root(task_name: str) -> str:
    """Backward-compatible alias used by existing task training scripts."""

    return resolve_log_root(task_name)
