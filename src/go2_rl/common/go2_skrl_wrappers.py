from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from go2_rl.common.info_utils import to_float, write_scalars


class Go2FrameStackWrapper(gym.Env):
    """Frame-stack wrapper for skrl IsaacLab training.

    Reset returns:
        {"policy": actor_stack, "critic": critic_stack}

    Step returns:
        {"policy": actor_stack, "critic": critic_stack}, reward, terminated, truncated, info

    For Task1:
        actor_obs_dim = 87
        critic_obs_dim = 87
        policy stack = 87 * 5 = 435
        critic stack = 87 * 5 = 435

    For Task2 and later:
        env can expose compute_privileged_obs()
        actor stack and critic stack may have different dims.
    """

    def __init__(
        self,
        env,
        log_dir: str,
        n_stack: int = 5,
        tb_log_interval_steps: int = 20,
        use_privileged_obs: bool = False,
    ):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device
        self.tb_log_interval_steps = int(tb_log_interval_steps)
        self.use_privileged_obs = bool(use_privileged_obs)

        self.actor_single_dim = int(env.observation_space.shape[0])

        if self.use_privileged_obs and hasattr(env, "state_space") and hasattr(env, "compute_privileged_obs"):
            self.critic_single_dim = int(env.state_space.shape[0])
        else:
            self.critic_single_dim = self.actor_single_dim

        self.actor_stacked_dim = self.actor_single_dim * self.n_stack
        self.critic_stacked_dim = self.critic_single_dim * self.n_stack

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.actor_stacked_dim,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.critic_stacked_dim,),
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
        self.critic_stack = torch.zeros(
            (self.num_envs, self.critic_stacked_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.writer = SummaryWriter(log_dir) if tb_log_interval_steps != 0 else None
        self.global_env_steps = 0
        self.local_step_count = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def _get_critic_obs(self) -> torch.Tensor:
        if self.use_privileged_obs and hasattr(self.env, "compute_privileged_obs"):
            return self.env.compute_privileged_obs()
        return self.env._compute_obs()

    def _pack(self):
        return {
            "policy": self.actor_stack.clone(),
            "critic": self.critic_stack.clone(),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        actor_obs, info = self.env.reset(seed=seed, options=options)
        critic_obs = self._get_critic_obs()

        for i in range(self.n_stack):
            self.actor_stack[:, i * self.actor_single_dim : (i + 1) * self.actor_single_dim] = actor_obs
            self.critic_stack[:, i * self.critic_single_dim : (i + 1) * self.critic_single_dim] = critic_obs

        self.last_info = info or {}
        return self._pack(), self.last_info

    @torch.no_grad()
    def step(self, action):
        actor_obs, reward, terminated, truncated, info = self.env.step(action)
        critic_obs = self._get_critic_obs()

        self.actor_stack[:, :-self.actor_single_dim] = self.actor_stack[:, self.actor_single_dim :].clone()
        self.actor_stack[:, -self.actor_single_dim :] = actor_obs

        self.critic_stack[:, :-self.critic_single_dim] = self.critic_stack[:, self.critic_single_dim :].clone()
        self.critic_stack[:, -self.critic_single_dim :] = critic_obs

        dones = terminated | truncated
        if dones.any():
            ids = dones.nonzero(as_tuple=False).squeeze(-1)

            for i in range(self.n_stack):
                self.actor_stack[
                    ids,
                    i * self.actor_single_dim : (i + 1) * self.actor_single_dim,
                ] = actor_obs[ids]
                self.critic_stack[
                    ids,
                    i * self.critic_single_dim : (i + 1) * self.critic_single_dim,
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
