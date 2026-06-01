# Copyright (c) 2026
# Unitree Go2 Common: 训练元数据写入工具。
#
# 本文件提供训练参数和补充信息的 JSON 保存函数。
# 本文件不依赖 IsaacLab，不启动 AppLauncher，也不创建训练环境。
#
# 工程说明:
#   训练脚本需要记录命令行参数、环境维度、checkpoint 来源和运行配置。
#   这里提供轻量 JSON 写入工具，避免各任务训练脚本重复实现。
#
# Unitree Go2 Common: training metadata writing utilities.
#
# This file provides JSON helpers for saving training arguments and auxiliary
# metadata. It does not depend on IsaacLab, launch AppLauncher, or create
# training environments.
#
# Engineering notes:
#   Training scripts need to record command-line arguments, environment
#   dimensions, checkpoint sources, and runtime configuration. This module
#   provides a lightweight JSON writer to avoid duplicate implementations.

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))

    if isinstance(value, argparse.Namespace):
        return _to_jsonable(vars(value))

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: str | Path, data: Mapping[str, Any]) -> str:
    """Write a JSON file and return its resolved path."""

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)


def write_train_metadata(log_dir: str | Path, *, args: Any = None, extra: Mapping[str, Any] | None = None) -> str:
    """Write train_metadata.json under a run log directory."""

    data = {}
    if args is not None:
        data["args"] = _to_jsonable(args)
    if extra:
        data["extra"] = _to_jsonable(extra)

    return write_json(Path(log_dir) / "train_metadata.json", data)
