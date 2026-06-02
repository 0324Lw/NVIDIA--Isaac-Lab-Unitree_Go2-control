# Copyright (c) 2026
# Unitree Go2 Common: 训练信息、日志标量和表格输出工具。
#
# 本文件提供轻量级训练日志辅助函数，不启动 IsaacLab AppLauncher，也不创建训练环境。
# 主要职责:
#   1. 将 tensor / numpy / Python 数值尽量转换为日志标量；
#   2. 展平嵌套 info 字典，便于控制台输出和 TensorBoard 写入；
#   3. 从 skrl agent 中提取 tracking_data 和当前学习率；
#   4. 写入 TensorBoard scalar；
#   5. 生成固定宽度控制台表格；
#   6. 保留 save_normalizers / load_normalizers 兼容入口，并转发到 normalizer_utils.py。
#
# 工程说明:
#   info 中通常保留 GPU tensor，以减少环境 step 内的 CPU 同步。
#   本文件只在低频日志、summary 和调试输出中做 best-effort 标量转换。
#
# Unitree Go2 Common: training info, scalar logging, and table utilities.
#
# This file provides lightweight training logging helpers. It does not launch
# IsaacLab AppLauncher or create training environments.
# Main responsibilities:
#   1. Convert tensor / numpy / Python values to logging scalars on a best-effort basis;
#   2. Flatten nested info dictionaries for console output and TensorBoard writing;
#   3. Extract tracking_data and the current learning rate from a skrl agent;
#   4. Write TensorBoard scalars;
#   5. Generate fixed-width console tables;
#   6. Keep save_normalizers / load_normalizers compatibility entries and forward them to normalizer_utils.py.
#
# Engineering notes:
#   info dictionaries often keep GPU tensors to reduce CPU synchronization in
#   environment step. This file performs best-effort scalar conversion only for
#   low-frequency logging, summaries, and debugging output.

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from go2_rl.common.normalizer_utils import load_normalizers, save_normalizers


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch

        return torch.is_tensor(x)
    except Exception:
        return False


def to_float(x: Any):
    """Best-effort scalar conversion for low-frequency logging."""

    try:
        if _is_torch_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (list, tuple)):
            return float(np.mean(x)) if len(x) else None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None

    return None


def flat_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested info dictionaries into scalar metrics."""

    out: Dict[str, float] = {}

    for k, v in (d or {}).items():
        if k == "terminal_observation":
            continue

        name = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat_dict(v, name))
        else:
            val = to_float(v)
            if val is not None and np.isfinite(val):
                out[name] = val

    return out


def tracking_mean(agent) -> Dict[str, float]:
    """Return mean/min/max summaries from skrl tracking_data."""

    out: Dict[str, float] = {}

    for k, v in getattr(agent, "tracking_data", {}).items():
        if v is None or len(v) == 0:
            continue

        try:
            arr = np.asarray(v, dtype=np.float64)
            if k.endswith("(min)"):
                out[k] = float(np.min(arr))
            elif k.endswith("(max)"):
                out[k] = float(np.max(arr))
            else:
                out[k] = float(np.mean(arr))
        except Exception:
            val = to_float(v)
            if val is not None:
                out[k] = val

    return out


def current_lr(agent) -> float:
    """Read the current optimizer learning rate from a skrl agent."""

    for obj in [
        getattr(agent, "optimizer", None),
        getattr(getattr(agent, "scheduler", None), "optimizer", None),
    ]:
        try:
            if obj is not None:
                return float(obj.param_groups[0]["lr"])
        except Exception:
            pass

    return float("nan")


def write_scalars(writer, data, step: int, prefix: str) -> None:
    """Write scalar-convertible values to TensorBoard."""

    if writer is None:
        return

    for k, v in (data or {}).items():
        val = to_float(v)
        if val is None:
            continue

        try:
            writer.add_scalar(f"{prefix}/{k}".replace("//", "/"), val, step)
        except Exception:
            pass


def make_table(title: str, data: Dict[str, Any], width: int = 112) -> str:
    """Build a fixed-width text table for console summaries."""

    lines = ["-" * width, f"| {title:<{width - 4}} |", "-" * width]

    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)

    for k in sorted(data.keys()):
        v = data[k]
        ks = (k[:68] + "...") if len(k) > 71 else k

        if isinstance(v, float):
            if abs(v) > 1e4 or 0 < abs(v) < 1e-3:
                vs = f"{v:.6e}"
            else:
                vs = f"{v:.6f}"
        else:
            val = to_float(v)
            if val is not None:
                vs = f"{val:.6f}"
            else:
                vs = str(v)

        vs = (vs[:36] + "...") if len(vs) > 39 else vs
        lines.append(f"| {ks:<71} | {vs:>{width - 78}} |")

    lines.append("-" * width)
    return "\n".join(lines)
