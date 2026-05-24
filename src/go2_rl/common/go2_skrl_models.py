from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class Go2Actor(GaussianMixin, Model):
    """Gaussian actor for skrl PPO.

    Compatible with the skrl PPO_CFG / StepTrainer format already verified
    in the user's IsaacLab environment.
    """

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
