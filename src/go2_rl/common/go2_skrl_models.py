# Copyright (c) 2026
# Unitree Go2 Common: skrl PPO 网络模型。
#
# 本文件定义 Unitree Go2 任务使用的 skrl PPO actor / critic 网络。
# 本文件不启动 IsaacLab AppLauncher，也不创建训练环境。
#
# 网络结构:
#   Go2Actor:
#     Gaussian policy，默认隐藏层为 512 -> 256 -> 128；
#     输出动作均值，并通过 log_std_parameter 提供对角高斯标准差。
#
#   Go2Critic:
#     Deterministic value critic，默认隐藏层为 512 -> 256 -> 128；
#     输入 state_space 对应的 critic observation，输出状态价值。
#
# 工程说明:
#   compute() 同时兼容 skrl 输入字典中的 observations 和 states 字段。
#   这样同一模型可以服务 actor-only 输入、asymmetric critic 输入以及不同 wrapper 返回格式。
#
# Unitree Go2 Common: skrl PPO network models.
#
# This file defines the skrl PPO actor / critic networks used by Unitree Go2
# tasks. It does not launch IsaacLab AppLauncher or create training environments.
#
# Network structure:
#   Go2Actor:
#     Gaussian policy with default hidden layers 512 -> 256 -> 128;
#     outputs action means and uses log_std_parameter for diagonal Gaussian std.
#
#   Go2Critic:
#     Deterministic value critic with default hidden layers 512 -> 256 -> 128;
#     takes critic observations from state_space and outputs state values.
#
# Engineering notes:
#   compute() supports both observations and states keys from skrl input
#   dictionaries. This keeps the same model compatible with actor-only inputs,
#   asymmetric critic inputs, and different wrapper output layouts.

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class Go2Actor(GaussianMixin, Model):
    """Gaussian actor for skrl PPO."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -1.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        hidden_dims=(512, 256, 128),
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=float(min_log_std),
            max_log_std=float(max_log_std),
            reduction="sum",
        )

        layers = []
        in_dim = int(observation_space.shape[0])
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, int(h)), nn.ELU()]
            in_dim = int(h)
        layers += [nn.Linear(in_dim, int(action_space.shape[0]))]
        self.net = nn.Sequential(*layers)

        self.log_std_parameter = nn.Parameter(
            torch.full(
                (int(action_space.shape[0]),),
                float(init_log_std),
                dtype=torch.float32,
            )
        )

        self.apply(self._orthogonal_init)

    @staticmethod
    def _orthogonal_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def compute(self, inputs, role):
        x = inputs.get("observations", inputs.get("states"))
        return self.net(x), {"log_std": self.log_std_parameter}


class Go2Critic(DeterministicMixin, Model):
    """State-value critic for skrl PPO."""

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        hidden_dims=(512, 256, 128),
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        layers = []
        in_dim = int(state_space.shape[0])
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, int(h)), nn.ELU()]
            in_dim = int(h)
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

        self.apply(Go2Actor._orthogonal_init)

    def compute(self, inputs, role):
        return self.net(inputs.get("states")), {}
