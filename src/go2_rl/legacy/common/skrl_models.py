from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


def _init_skrl_model(model_obj, observation_space, action_space, device) -> None:
    """兼容不同 skrl 版本的 Model 初始化接口。"""
    try:
        Model.__init__(
            model_obj,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
    except TypeError:
        Model.__init__(model_obj, observation_space, action_space, device)


def _init_gaussian_mixin(
    model_obj,
    clip_actions: bool,
    clip_log_std: bool,
    min_log_std: float,
    max_log_std: float,
    reduction: str,
) -> None:
    """兼容不同 skrl 版本的 GaussianMixin 初始化接口。"""
    try:
        GaussianMixin.__init__(
            model_obj,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )
    except TypeError:
        GaussianMixin.__init__(
            model_obj,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            reduction,
        )


def _init_deterministic_mixin(model_obj, clip_actions: bool) -> None:
    """兼容不同 skrl 版本的 DeterministicMixin 初始化接口。"""
    try:
        DeterministicMixin.__init__(
            model_obj,
            clip_actions=clip_actions,
        )
    except TypeError:
        DeterministicMixin.__init__(model_obj, clip_actions)


class Go2GaussianPolicy(GaussianMixin, Model):
    """Gaussian policy for skrl PPO.

    注意：
    1. action 输出交给 skrl 和环境共同 clip 到 [-1, 1]。
    2. 环境内部还会做 action clamp + EMA，因此这里不再手动 tanh。
    3. compute 返回格式遵循 skrl GaussianMixin: mean, log_std, extras。
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
        max_log_std: float = 0.5,
        initial_log_std: float = -1.0,
        reduction: str = "sum",
    ):
        _init_skrl_model(self, observation_space, action_space, device)
        _init_gaussian_mixin(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )

        self.log_std_parameter = nn.Parameter(
            torch.full(
                (self.num_actions,),
                float(initial_log_std),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        states = inputs["states"]
        mean = self.net(states)
        return mean, self.log_std_parameter, {}


class Go2Value(DeterministicMixin, Model):
    """State-value function for skrl PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
    ):
        _init_skrl_model(self, observation_space, action_space, device)
        _init_deterministic_mixin(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), {}
