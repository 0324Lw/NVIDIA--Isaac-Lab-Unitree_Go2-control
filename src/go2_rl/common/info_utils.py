from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def to_float(x: Any):
    """Best-effort scalar conversion.

    This is used only for low-frequency logging / progress display.
    Training code should avoid calling it inside heavy per-env reward computation.
    """

    try:
        if torch.is_tensor(x):
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
    if writer is None:
        return

    for k, v in (data or {}).items():
        val = to_float(v)
        if val is not None:
            try:
                writer.add_scalar(f"{prefix}/{k}".replace("//", "/"), val, step)
            except Exception:
                pass


def save_normalizers(agent, save_dir: str) -> None:
    import os

    for name in [
        "observation_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
        "_observation_preprocessor",
        "_state_preprocessor",
        "_value_preprocessor",
    ]:
        obj = getattr(agent, name, None)
        if obj is None:
            continue
        try:
            torch.save(obj.state_dict(), os.path.join(save_dir, f"{name}.pt"))
        except Exception:
            pass


def load_normalizers(agent, load_dir: str):
    import os

    loaded = []
    for name in [
        "observation_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
        "_observation_preprocessor",
        "_state_preprocessor",
        "_value_preprocessor",
    ]:
        obj = getattr(agent, name, None)
        if obj is None:
            continue

        path = os.path.join(load_dir, f"{name}.pt")
        if not os.path.exists(path):
            continue

        try:
            obj.load_state_dict(torch.load(path, map_location=getattr(obj, "device", "cpu")))
            loaded.append(name)
        except Exception:
            pass
    return loaded


def make_table(title: str, data: Dict[str, Any], width: int = 112) -> str:
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
