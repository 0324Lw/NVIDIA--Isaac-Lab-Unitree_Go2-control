# Copyright (c) 2026
# Unitree Go2 Common: skrl normalizer 保存与加载工具。
#
# 本文件提供 skrl preprocessor / normalizer 的保存和加载函数。
# 本文件不依赖 IsaacLab，不启动 AppLauncher，也不创建训练环境。
#
# 工程说明:
#   skrl 在不同版本中可能使用公开属性或下划线属性保存 preprocessors。
#   因此这里同时检查 observation_preprocessor、state_preprocessor、
#   value_preprocessor 以及对应的下划线属性，便于兼容不同版本。
#
# Unitree Go2 Common: skrl normalizer save/load utilities.
#
# This file provides helper functions for saving and loading skrl
# preprocessors / normalizers. It does not depend on IsaacLab, launch
# AppLauncher, or create training environments.
#
# Engineering notes:
#   Different skrl versions may store preprocessors in public attributes or
#   underscore-prefixed attributes. Therefore this module checks both forms for
#   observation, state, and value preprocessors.

from __future__ import annotations

from pathlib import Path
from typing import List


NORMALIZER_ATTRS = (
    "observation_preprocessor",
    "state_preprocessor",
    "value_preprocessor",
    "_observation_preprocessor",
    "_state_preprocessor",
    "_value_preprocessor",
)


def save_normalizers(agent, save_dir: str | Path) -> List[str]:
    """Save available skrl normalizer state_dicts and return saved names."""

    import torch

    path = Path(save_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for name in NORMALIZER_ATTRS:
        obj = getattr(agent, name, None)
        if obj is None:
            continue

        try:
            torch.save(obj.state_dict(), path / f"{name}.pt")
            saved.append(name)
        except Exception:
            pass

    return saved


def load_normalizers(agent, load_dir: str | Path) -> List[str]:
    """Load available skrl normalizer state_dicts and return loaded names."""

    import torch

    path = Path(load_dir).expanduser().resolve()
    loaded: List[str] = []

    for name in NORMALIZER_ATTRS:
        obj = getattr(agent, name, None)
        if obj is None:
            continue

        normalizer_path = path / f"{name}.pt"
        if not normalizer_path.exists():
            continue

        try:
            obj.load_state_dict(torch.load(normalizer_path, map_location=getattr(obj, "device", "cpu")))
            loaded.append(name)
        except Exception:
            pass

    return loaded
