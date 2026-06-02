# Copyright (c) 2026
# Unitree Go2 Common: 模型评估课程进度工具。
#
# 本文件提供模型评估阶段的课程进度强制设置函数。
# 本文件不启动 IsaacLab AppLauncher，也不创建训练环境。
#
# 主要职责:
#   1. 遍历 skrl wrapper、Gymnasium wrapper 和原始环境组成的 env 链；
#   2. 从 env 或 cfg 中推断 curriculum_total_steps；
#   3. 根据 start_k 设置评估时的 global_steps / curriculum_steps；
#   4. 保证最终 checkpoint 可以在指定课程阶段进行评估。
#
# 工程说明:
#   评估脚本经常会在设置 start_k 后再次 reset 环境。
#   因此需要在 wrapper 链中找到真实环境对象，并设置影响课程阶段计算的步数字段。
#   本函数不直接改写 command buffer，而是让环境在 reset / step 中按自己的逻辑重新计算命令。
#
# Unitree Go2 Common: model-evaluation curriculum utilities.
#
# This file provides helper functions for forcing curriculum progress during
# model evaluation. It does not launch IsaacLab AppLauncher or create training
# environments.
#
# Main responsibilities:
#   1. Traverse the env chain formed by skrl wrappers, Gymnasium wrappers, and the raw environment;
#   2. Infer curriculum_total_steps from the env or cfg objects;
#   3. Set global_steps / curriculum_steps according to start_k for evaluation;
#   4. Ensure final checkpoints can be evaluated at a specified curriculum stage.
#
# Engineering notes:
#   Evaluation scripts may reset the environment after start_k has been set.
#   Therefore this helper searches the wrapper chain for the real environment
#   objects and updates the step fields used by curriculum-stage computation.
#   It does not directly overwrite command buffers; the environment recomputes
#   commands through its own reset / step logic.

from __future__ import annotations

from typing import Any, List, Set


def _unwrap_candidates(obj: Any) -> List[Any]:
    out = []

    for attr in [
        "env",
        "_env",
        "unwrapped",
        "venv",
        "gym_env",
        "raw_env",
        "base_env",
        "wrapped_env",
    ]:
        try:
            value = getattr(obj, attr, None)
        except Exception:
            value = None

        if value is not None and value is not obj:
            out.append(value)

    return out


def _collect_env_chain(root: Any) -> List[Any]:
    seen: Set[int] = set()
    stack = [root]
    out = []

    while stack:
        obj = stack.pop(0)

        if obj is None:
            continue

        obj_id = id(obj)
        if obj_id in seen:
            continue

        seen.add(obj_id)
        out.append(obj)

        for child in _unwrap_candidates(obj):
            stack.append(child)

    return out


def _find_total_curriculum_steps(env_chain: List[Any]) -> int:
    candidates = []

    for env in env_chain:
        cfg = getattr(env, "cfg", None)

        for obj in [env, cfg]:
            if obj is None:
                continue

            for attr in [
                "curriculum_total_steps",
                "total_env_steps",
                "max_curriculum_steps",
                "training_total_steps",
            ]:
                if hasattr(obj, attr):
                    try:
                        value = int(getattr(obj, attr))
                        if value > 0:
                            candidates.append(value)
                    except Exception:
                        pass

    if candidates:
        return max(candidates)

    return 600_000_000


def force_eval_curriculum(env_like: Any, start_k: float = 1.0, label: str = "") -> int:
    """Set evaluation curriculum progress and return the target step count."""

    try:
        k = float(start_k)
    except Exception:
        k = 1.0

    k = max(0.0, min(1.0, k))

    chain = _collect_env_chain(env_like)
    total_steps = _find_total_curriculum_steps(chain)
    target_steps = int(k * total_steps)

    changed = 0

    for env in chain:
        for attr in [
            "global_steps",
            "global_env_steps",
            "curriculum_steps",
            "curriculum_step",
        ]:
            if hasattr(env, attr):
                try:
                    setattr(env, attr, target_steps)
                    changed += 1
                except Exception:
                    pass

    prefix = f"[CURRICULUM][{label}]" if label else "[CURRICULUM]"
    print(
        f"{prefix} forced start_k={k:.4f}, target_steps={target_steps:,}, "
        f"total_steps={total_steps:,}, updated_fields={changed}",
        flush=True,
    )

    return target_steps
