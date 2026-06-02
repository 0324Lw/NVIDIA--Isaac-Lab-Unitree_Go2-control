# Copyright (c) 2026
# Unitree Go2 Common: 模型评估动作推理工具。
#
# 本文件提供 skrl agent 评估阶段的兼容初始化和直接 policy 动作推理函数。
# 本文件不依赖 IsaacLab，不启动 AppLauncher，也不创建训练环境。
#
# 主要职责:
#   1. 兼容不同 skrl 版本的 agent.init() 调用形式；
#   2. 从 skrl / Gymnasium wrapper 返回的 states 中提取 policy observation tensor；
#   3. 在评估阶段应用 observation preprocessor；
#   4. 直接调用 policy 网络得到动作，并将动作裁剪到 [-1, 1]。
#
# 工程说明:
#   评估脚本使用 direct_policy_action() 是为了避免不同 skrl 版本中
#   agent.act() / GaussianMixin 采样路径的兼容差异，同时保持确定性评估路径清晰。
#
# Unitree Go2 Common: model evaluation action inference utilities.
#
# This file provides compatibility initialization and direct policy-action
# inference helpers for skrl agents during evaluation. It does not depend on
# IsaacLab, launch AppLauncher, or create training environments.
#
# Main responsibilities:
#   1. Support different skrl versions of agent.init();
#   2. Extract the policy observation tensor from states returned by skrl / Gymnasium wrappers;
#   3. Apply the observation preprocessor during evaluation;
#   4. Call the policy network directly and clamp actions to [-1, 1].
#
# Engineering notes:
#   Evaluation scripts use direct_policy_action() to avoid compatibility
#   differences in agent.act() / GaussianMixin sampling paths across skrl
#   versions while keeping deterministic evaluation behavior explicit.

from __future__ import annotations

import time
from typing import Any


def _torch():
    import torch

    return torch


def init_agent_compat(agent) -> None:
    """Initialize a skrl agent while tolerating older trainer_cfg handling."""

    try:
        agent.init(trainer_cfg={"timesteps": 1, "headless": True})
    except TypeError as exc:
        if "asdict" not in str(exc) and "dataclass" not in str(exc):
            raise
        print("[WARN] agent.init(trainer_cfg=dict) is not supported by this skrl build; fallback to agent.init().")
        agent.init()


def extract_policy_tensor(states: Any):
    """Extract the policy observation tensor from wrapper states."""

    torch = _torch()

    if isinstance(states, dict):
        for key in ["policy", "observations", "states", "obs"]:
            if key in states and torch.is_tensor(states[key]):
                return states[key]

        for value in states.values():
            if torch.is_tensor(value):
                return value

    if torch.is_tensor(states):
        return states

    raise RuntimeError(f"Cannot extract policy tensor from states type={type(states)}")


def _apply_observation_preprocessor(agent, obs, debug: bool = False, step: int = 0):
    prep = getattr(agent, "_observation_preprocessor", None) or getattr(agent, "observation_preprocessor", None)

    if prep is None:
        return obs

    if debug:
        print(f"[DEBUG][eval step {step}] before observation preprocessor", flush=True)

    t0 = time.time()

    try:
        out = prep(obs, train=False)
    except TypeError:
        try:
            out = prep(obs)
        except Exception as exc:
            if debug:
                print(f"[DEBUG][eval step {step}] preprocessor failed: {type(exc).__name__}: {exc}", flush=True)
            return obs
    except Exception as exc:
        if debug:
            print(f"[DEBUG][eval step {step}] preprocessor failed: {type(exc).__name__}: {exc}", flush=True)
        return obs

    if isinstance(out, tuple):
        out = out[0]

    if debug:
        print(f"[DEBUG][eval step {step}] after observation preprocessor, dt={time.time() - t0:.4f}s", flush=True)

    return out


def _get_policy(agent):
    try:
        return agent.models["policy"]
    except Exception:
        pass

    policy = getattr(agent, "policy", None)
    if policy is not None:
        return policy

    raise RuntimeError("Cannot find policy model from skrl agent.")


def _policy_forward(policy, obs, debug: bool = False, step: int = 0):
    if debug:
        print(f"[DEBUG][eval step {step}] before policy forward", flush=True)

    t0 = time.time()

    for attr in ["net", "network", "actor", "model"]:
        module = getattr(policy, attr, None)
        if module is not None and callable(module):
            out = module(obs)
            if isinstance(out, tuple):
                out = out[0]
            if debug:
                print(f"[DEBUG][eval step {step}] after policy.{attr}(obs), dt={time.time() - t0:.4f}s", flush=True)
            return out

    try:
        out = policy.compute({"observations": obs, "states": obs}, role="policy")
    except TypeError:
        out = policy.compute({"observations": obs, "states": obs})

    if isinstance(out, tuple):
        out = out[0]

    if debug:
        print(f"[DEBUG][eval step {step}] after policy.compute, dt={time.time() - t0:.4f}s", flush=True)

    return out


def direct_policy_action(agent, states, *, debug: bool = False, step: int = 0):
    """Return clipped deterministic policy actions for evaluation."""

    torch = _torch()

    if debug:
        print(f"[DEBUG][eval step {step}] before extract obs", flush=True)

    with torch.no_grad():
        obs = extract_policy_tensor(states)

        if debug:
            print(
                f"[DEBUG][eval step {step}] obs shape={tuple(obs.shape)}, "
                f"obs_min={obs.min().item():+.4f}, obs_max={obs.max().item():+.4f}",
                flush=True,
            )

        obs = _apply_observation_preprocessor(agent, obs, debug=debug, step=step)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, -10.0, 10.0)

        policy = _get_policy(agent)
        actions = _policy_forward(policy, obs, debug=debug, step=step)

        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        if debug:
            print(
                f"[DEBUG][eval step {step}] action shape={tuple(actions.shape)}, "
                f"action_min={actions.min().item():+.4f}, action_max={actions.max().item():+.4f}",
                flush=True,
            )

        return actions
