# Copyright (c) 2026
# Unitree Go2 Common: checkpoint 路径解析与状态提取工具。
#
# 本文件提供 checkpoint 文件查找、路径解析、torch 加载和 policy state 提取函数。
# 本文件不依赖 IsaacLab，不启动 AppLauncher，也不创建训练环境。
#
# 工程说明:
#   训练脚本和评估脚本中经常需要同时支持“文件路径”和“final_checkpoint 目录”。
#   这里把通用路径解析逻辑独立出来，后续再逐步替换各任务脚本中的重复代码。
#
# Unitree Go2 Common: checkpoint path resolution and state extraction utilities.
#
# This file provides checkpoint discovery, path resolution, torch loading, and
# policy state extraction helpers. It does not depend on IsaacLab, launch
# AppLauncher, or create training environments.
#
# Engineering notes:
#   Training and evaluation scripts often need to support both direct checkpoint
#   files and final_checkpoint directories. This module isolates the common
#   path-resolution logic so task scripts can be migrated gradually.

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def find_latest_checkpoint(
    log_root: str | Path,
    model_file_name: str,
    checkpoint_dir_name: str = "final_checkpoint",
) -> str:
    """Find the newest model file under log_root/*/final_checkpoint."""

    root = Path(log_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"log_root not found: {root}")

    candidates = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        model_path = run_dir / checkpoint_dir_name / model_file_name
        if model_path.exists():
            candidates.append(model_path)

    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint file named {model_file_name!r} found under {root}/*/{checkpoint_dir_name}"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def resolve_checkpoint_path(
    checkpoint: str | Path | None = None,
    *,
    log_root: str | Path | None = None,
    model_file_name: str | None = None,
) -> str:
    """Resolve a checkpoint file from file, directory, or latest log root."""

    if checkpoint is not None and str(checkpoint).strip():
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint path not found: {path}")

        if path.is_dir():
            if not model_file_name:
                raise ValueError("model_file_name is required when checkpoint is a directory")
            model_path = path / model_file_name
            if not model_path.exists():
                raise FileNotFoundError(f"model file not found under checkpoint directory: {model_path}")
            return str(model_path.resolve())

        return str(path)

    if log_root is None or not model_file_name:
        raise ValueError("log_root and model_file_name are required when checkpoint is empty")

    return find_latest_checkpoint(log_root, model_file_name)


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> Any:
    """Load a torch checkpoint with a consistent map_location default."""

    import torch

    return torch.load(Path(path).expanduser().resolve(), map_location=map_location)


def extract_policy_state(checkpoint: Any) -> Mapping[str, Any]:
    """Extract a policy state dict from common skrl checkpoint layouts."""

    if isinstance(checkpoint, Mapping):
        for key in (
            "policy",
            "policy_state_dict",
            "model",
            "model_state_dict",
            "state_dict",
        ):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value

        models = checkpoint.get("models")
        if isinstance(models, Mapping):
            for key in ("policy", "actor"):
                value = models.get(key)
                if isinstance(value, Mapping):
                    return value

    if isinstance(checkpoint, Mapping):
        return checkpoint

    raise TypeError(f"cannot extract policy state from checkpoint type={type(checkpoint)!r}")
