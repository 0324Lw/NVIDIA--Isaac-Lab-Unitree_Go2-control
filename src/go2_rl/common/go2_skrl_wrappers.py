# Copyright (c) 2026
# Unitree Go2 Common: skrl Gymnasium wrapper。
#
# 本文件提供 Unitree Go2 任务使用的 skrl 兼容 Gymnasium wrapper。
# 本文件不启动 IsaacLab AppLauncher，也不创建训练环境。
#
# Gymnasium API:
#   reset() -> obs_dict, info
#   step(action) -> obs_dict, reward, terminated, truncated, info
#
# obs_dict 格式:
#   policy: actor observation stack
#   critic: critic observation
#
# 支持的 critic 构造模式:
#   stack_actor:
#     policy = actor_stack
#     critic = actor_stack
#     当前 Task1 默认使用该模式。
#
#   stack_full_privileged:
#     policy = actor_stack
#     critic = full_privileged_obs_stack
#     该模式保留 use_privileged_obs=True 的旧行为，用于兼容已有调用。
#
#   actor_stack_plus_privileged_tail:
#     policy = actor_stack
#     critic = actor_stack + privileged_tail
#     该模式用于后续迁移 Task2 / Task3 的 asymmetric critic，不在本次直接接入。
#
# 工程说明:
#   Task2 / Task3 当前仍使用各自训练脚本内的本地 asymmetric wrapper。
#   这里先把通用 wrapper 的模式边界整理清楚，避免再次出现“完整 privileged obs 被整段堆叠”
#   导致 critic 维度偏大的问题。
#
# Unitree Go2 Common: skrl Gymnasium wrapper.
#
# This file provides a skrl-compatible Gymnasium wrapper for Unitree Go2 tasks.
# It does not launch IsaacLab AppLauncher or create training environments.
#
# Gymnasium API:
#   reset() -> obs_dict, info
#   step(action) -> obs_dict, reward, terminated, truncated, info
#
# obs_dict layout:
#   policy: actor observation stack
#   critic: critic observation
#
# Supported critic construction modes:
#   stack_actor:
#     policy = actor_stack
#     critic = actor_stack
#     This is the current default mode for Task1.
#
#   stack_full_privileged:
#     policy = actor_stack
#     critic = full_privileged_obs_stack
#     This keeps the legacy use_privileged_obs=True behavior for compatibility.
#
#   actor_stack_plus_privileged_tail:
#     policy = actor_stack
#     critic = actor_stack + privileged_tail
#     This mode is prepared for later Task2 / Task3 asymmetric critic migration,
#     but it is not wired into those tasks in this step.
#
# Engineering notes:
#   Task2 / Task3 still use their task-local asymmetric wrappers in the current
#   codebase. This module only clarifies the common wrapper mode boundaries so
#   full privileged observations are not accidentally stacked when a task expects
#   actor_stack + privileged_tail.

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from go2_rl.common.info_utils import to_float, write_scalars


class Go2FrameStackWrapper(gym.Env):
    """Frame-stack wrapper for skrl IsaacLab training and evaluation."""

    STACK_ACTOR = "stack_actor"
    STACK_FULL_PRIVILEGED = "stack_full_privileged"
    ACTOR_STACK_PLUS_PRIVILEGED_TAIL = "actor_stack_plus_privileged_tail"

    def __init__(
        self,
        env,
        log_dir: str,
        n_stack: int = 5,
        tb_log_interval_steps: int = 20,
        use_privileged_obs: bool = False,
        critic_mode: Optional[str] = None,
        privileged_tail_dim: Optional[int] = None,
    ):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device
        self.tb_log_interval_steps = int(tb_log_interval_steps)
        self.use_privileged_obs = bool(use_privileged_obs)

        if critic_mode is None:
            critic_mode = self.STACK_FULL_PRIVILEGED if self.use_privileged_obs else self.STACK_ACTOR
        self.critic_mode = str(critic_mode)

        valid_modes = {
            self.STACK_ACTOR,
            self.STACK_FULL_PRIVILEGED,
            self.ACTOR_STACK_PLUS_PRIVILEGED_TAIL,
        }
        if self.critic_mode not in valid_modes:
            raise ValueError(f"Unsupported critic_mode={self.critic_mode!r}. Valid modes: {sorted(valid_modes)}")

        self.actor_single_dim = int(env.observation_space.shape[0])
        self.actor_stacked_dim = self.actor_single_dim * self.n_stack

        self.critic_single_dim: Optional[int]
        self.privileged_tail_dim: int

        if self.critic_mode == self.STACK_ACTOR:
            self.critic_single_dim = self.actor_single_dim
            self.privileged_tail_dim = 0
            self.critic_obs_dim = self.actor_stacked_dim

        elif self.critic_mode == self.STACK_FULL_PRIVILEGED:
            if not hasattr(env, "compute_privileged_obs"):
                raise RuntimeError("stack_full_privileged requires env.compute_privileged_obs().")

            if hasattr(env, "state_space"):
                self.critic_single_dim = int(env.state_space.shape[0])
            elif hasattr(env, "cfg") and hasattr(env.cfg, "num_privileged_obs"):
                self.critic_single_dim = int(env.cfg.num_privileged_obs)
            else:
                raise RuntimeError("Cannot infer privileged observation dimension for stack_full_privileged.")

            self.privileged_tail_dim = 0
            self.critic_obs_dim = int(self.critic_single_dim) * self.n_stack

        else:
            if not hasattr(env, "compute_privileged_obs"):
                raise RuntimeError("actor_stack_plus_privileged_tail requires env.compute_privileged_obs().")

            if privileged_tail_dim is not None:
                self.privileged_tail_dim = int(privileged_tail_dim)
            elif hasattr(env, "cfg") and hasattr(env.cfg, "num_privileged_obs"):
                self.privileged_tail_dim = int(env.cfg.num_privileged_obs) - self.actor_single_dim
            elif hasattr(env, "state_space"):
                self.privileged_tail_dim = int(env.state_space.shape[0]) - self.actor_single_dim
            else:
                raise RuntimeError("Cannot infer privileged tail dimension for actor_stack_plus_privileged_tail.")

            if self.privileged_tail_dim <= 0:
                raise RuntimeError(
                    "actor_stack_plus_privileged_tail requires positive privileged_tail_dim, "
                    f"got {self.privileged_tail_dim}."
                )

            self.critic_single_dim = None
            self.critic_obs_dim = self.actor_stacked_dim + self.privileged_tail_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.actor_stacked_dim,),
            dtype=np.float32,
        )

        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.critic_obs_dim,),
            dtype=np.float32,
        )

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.observation_space,
                "critic": self.state_space,
            }
        )

        self.action_space = env.action_space
        self.single_action_space = env.action_space

        self.actor_stack = torch.zeros(
            (self.num_envs, self.actor_stacked_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.critic_stack: Optional[torch.Tensor]
        if self.critic_mode in {self.STACK_ACTOR, self.STACK_FULL_PRIVILEGED}:
            assert self.critic_single_dim is not None
            self.critic_stack = torch.zeros(
                (self.num_envs, int(self.critic_single_dim) * self.n_stack),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.critic_stack = None

        self.writer = SummaryWriter(log_dir) if self.tb_log_interval_steps != 0 else None
        self.global_env_steps = 0
        self.local_step_count = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def _compute_actor_obs(self) -> torch.Tensor:
        if hasattr(self.env, "_compute_obs"):
            return self.env._compute_obs()
        raise RuntimeError("Wrapped env does not provide _compute_obs().")

    def _compute_privileged_obs(self) -> torch.Tensor:
        if not hasattr(self.env, "compute_privileged_obs"):
            raise RuntimeError("Wrapped env does not provide compute_privileged_obs().")
        return self.env.compute_privileged_obs()

    def _get_stack_critic_obs(self) -> torch.Tensor:
        if self.critic_mode == self.STACK_FULL_PRIVILEGED:
            return self._compute_privileged_obs()
        return self._compute_actor_obs()

    def _get_privileged_tail(self) -> torch.Tensor:
        raw_priv = self._compute_privileged_obs()
        tail = raw_priv[:, self.actor_single_dim :]

        if tail.shape[-1] != self.privileged_tail_dim:
            raise RuntimeError(
                f"privileged_tail dim mismatch: got {tail.shape[-1]}, expected {self.privileged_tail_dim}"
            )

        return tail

    def _build_critic_obs(self) -> torch.Tensor:
        if self.critic_mode == self.ACTOR_STACK_PLUS_PRIVILEGED_TAIL:
            return torch.cat([self.actor_stack, self._get_privileged_tail()], dim=-1)

        if self.critic_stack is None:
            raise RuntimeError("critic_stack is unexpectedly None for stack critic mode.")

        return self.critic_stack

    def _pack(self):
        critic = self._build_critic_obs()
        if critic.shape[-1] != self.critic_obs_dim:
            raise RuntimeError(f"critic obs dim mismatch: got {critic.shape[-1]}, expected {self.critic_obs_dim}")

        return {
            "policy": self.actor_stack.clone(),
            "critic": critic.clone(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        actor_obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.actor_stack[:, i * self.actor_single_dim : (i + 1) * self.actor_single_dim] = actor_obs

        if self.critic_mode in {self.STACK_ACTOR, self.STACK_FULL_PRIVILEGED}:
            critic_obs = self._get_stack_critic_obs()
            assert self.critic_stack is not None
            assert self.critic_single_dim is not None

            if critic_obs.shape[-1] != int(self.critic_single_dim):
                raise RuntimeError(
                    f"critic single obs dim mismatch: got {critic_obs.shape[-1]}, expected {self.critic_single_dim}"
                )

            for i in range(self.n_stack):
                self.critic_stack[
                    :,
                    i * int(self.critic_single_dim) : (i + 1) * int(self.critic_single_dim),
                ] = critic_obs

        self.last_info = info or {}
        return self._pack(), self.last_info

    @torch.no_grad()
    def step(self, action):
        actor_obs, reward, terminated, truncated, info = self.env.step(action)

        self.actor_stack[:, :-self.actor_single_dim] = self.actor_stack[:, self.actor_single_dim :].clone()
        self.actor_stack[:, -self.actor_single_dim :] = actor_obs

        if self.critic_mode in {self.STACK_ACTOR, self.STACK_FULL_PRIVILEGED}:
            critic_obs = self._get_stack_critic_obs()
            assert self.critic_stack is not None
            assert self.critic_single_dim is not None

            if critic_obs.shape[-1] != int(self.critic_single_dim):
                raise RuntimeError(
                    f"critic single obs dim mismatch: got {critic_obs.shape[-1]}, expected {self.critic_single_dim}"
                )

            self.critic_stack[:, : -int(self.critic_single_dim)] = self.critic_stack[
                :,
                int(self.critic_single_dim) :,
            ].clone()
            self.critic_stack[:, -int(self.critic_single_dim) :] = critic_obs

        dones = terminated | truncated
        if dones.any():
            ids = dones.nonzero(as_tuple=False).squeeze(-1)

            for i in range(self.n_stack):
                self.actor_stack[
                    ids,
                    i * self.actor_single_dim : (i + 1) * self.actor_single_dim,
                ] = actor_obs[ids]

            if self.critic_mode in {self.STACK_ACTOR, self.STACK_FULL_PRIVILEGED}:
                assert self.critic_stack is not None
                assert self.critic_single_dim is not None
                critic_obs = self._get_stack_critic_obs()

                for i in range(self.n_stack):
                    self.critic_stack[
                        ids,
                        i * int(self.critic_single_dim) : (i + 1) * int(self.critic_single_dim),
                    ] = critic_obs[ids]

        self.global_env_steps += self.num_envs
        self.local_step_count += 1
        self.last_info = info or {}
        self.last_reward_mean = to_float(reward) or 0.0
        self.last_done_count = int(dones.sum().detach().cpu().item())

        if (
            self.writer is not None
            and self.tb_log_interval_steps > 0
            and self.local_step_count % self.tb_log_interval_steps == 0
        ):
            write_scalars(self.writer, self.last_info.get("reward_components", {}), self.global_env_steps, "rewards")
            write_scalars(self.writer, self.last_info.get("events", {}), self.global_env_steps, "events")
            write_scalars(self.writer, self.last_info.get("telemetry", {}), self.global_env_steps, "telemetry")
            write_scalars(self.writer, self.last_info.get("curriculum", {}), self.global_env_steps, "curriculum")
            write_scalars(self.writer, self.last_info.get("debug", {}), self.global_env_steps, "debug")
            self.writer.add_scalar("rollout/reward_mean_raw", self.last_reward_mean, self.global_env_steps)
            self.writer.add_scalar("rollout/done_count", self.last_done_count, self.global_env_steps)

        return self._pack(), reward, terminated, truncated, self.last_info

    def close(self):
        try:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass

        try:
            self.env.close()
        except Exception:
            pass
