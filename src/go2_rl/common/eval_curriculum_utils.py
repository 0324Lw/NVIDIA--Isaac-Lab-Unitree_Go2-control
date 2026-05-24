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

    # Fallback only used if old env has no explicit curriculum_total_steps.
    return 600_000_000


def force_eval_curriculum(env_like: Any, start_k: float = 1.0, label: str = "") -> int:
    """Force model-test curriculum progress.

    This is intentionally used only in model_test.py.
    Training code must not call this function.

    It traverses wrapper.env / unwrapped / raw_env chains and sets:
        global_steps = int(start_k * curriculum_total_steps)

    Reason:
        Some model tests reset the env after setting start_k, or never propagate
        start_k into the raw env. This caused final checkpoints to be evaluated
        at Stage 0 with zero command.
    """
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

        # Some envs store current stage / command buffers derived from global_steps.
        # Do not manually overwrite commands here; let env.step() recompute them.
        # This avoids breaking task-specific command sampling.

    prefix = f"[CURRICULUM][{label}]" if label else "[CURRICULUM]"
    print(
        f"{prefix} forced start_k={k:.4f}, target_steps={target_steps:,}, "
        f"total_steps={total_steps:,}, updated_fields={changed}",
        flush=True,
    )

    return target_steps
