from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from go2_rl.legacy.common.running_mean_std import RunningMeanStd


class Go2FrameStackNormWrapper(gym.Env):
    """Vectorized IsaacLab env wrapper for skrl.

    输入底层 env:
        reset -> obs [N, single_obs_dim], info
        step  -> obs [N, single_obs_dim], reward [N], terminated [N], truncated [N], info

    输出给 skrl:
        obs        [N, single_obs_dim * n_stack]
        reward     [N, 1]
        terminated [N, 1]
        truncated  [N, 1]

    训练时 update_obs_rms=True；模型测试时 update_obs_rms=False。
    """

    def __init__(
        self,
        env: gym.Env,
        n_stack: int = 5,
        obs_clip: float = 10.0,
        update_obs_rms: bool = True,
    ):
        super().__init__()
        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.update_obs_rms = bool(update_obs_rms)

        self.single_obs_dim = int(env.observation_space.shape[0])
        self.stacked_obs_dim = self.single_obs_dim * self.n_stack

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_obs_dim,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

        self.obs_stack = torch.zeros(
            (self.num_envs, self.stacked_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_rms = RunningMeanStd(
            shape=(self.stacked_obs_dim,),
            device=self.device,
            clip=float(obs_clip),
        )

        self.global_env_steps = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    def set_eval(self) -> None:
        self.update_obs_rms = False

    def set_train(self) -> None:
        self.update_obs_rms = True

    def _normalize(self, obs_stack: torch.Tensor) -> torch.Tensor:
        if self.update_obs_rms:
            self.obs_rms.update(obs_stack)
        return self.obs_rms.normalize(obs_stack)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.to(self.device).float()

        for i in range(self.n_stack):
            start = i * self.single_obs_dim
            end = (i + 1) * self.single_obs_dim
            self.obs_stack[:, start:end] = obs

        self.last_info = info or {}
        return self._normalize(self.obs_stack.clone()), self.last_info

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.to(device=self.device, dtype=torch.float32)

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.to(self.device).float()

        self.obs_stack[:, :-self.single_obs_dim] = self.obs_stack[:, self.single_obs_dim:].clone()
        self.obs_stack[:, -self.single_obs_dim:] = obs

        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                start = i * self.single_obs_dim
                end = (i + 1) * self.single_obs_dim
                self.obs_stack[ids, start:end] = obs[ids]

        self.global_env_steps += self.num_envs
        self.last_info = info or {}
        self.last_reward_mean = float(reward.detach().float().mean().cpu().item())
        self.last_done_count = int(done.detach().sum().cpu().item())

        states = self._normalize(self.obs_stack.clone())
        return (
            states,
            reward.view(self.num_envs, 1).to(self.device).float(),
            terminated.view(self.num_envs, 1).to(self.device),
            truncated.view(self.num_envs, 1).to(self.device),
            self.last_info,
        )

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
