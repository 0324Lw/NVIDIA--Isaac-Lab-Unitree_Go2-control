from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


class RunningMeanStd:
    """Small standalone RMS helper.

    Kept for model export / diagnostics.
    skrl training itself uses RunningStandardScaler.
    """

    def __init__(
        self,
        shape: Tuple[int, ...] | int,
        device: str | torch.device,
        eps: float = 1e-4,
        clip: float = 10.0,
    ):
        if isinstance(shape, int):
            shape = (shape,)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(float(eps), dtype=torch.float32, device=device)
        self.clip = float(clip)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float32, device=x.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(m_2 / total_count)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (x - self.mean) / torch.sqrt(self.var + 1e-8),
            -self.clip,
            self.clip,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count.detach().cpu(),
            "clip": self.clip,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count.copy_(state["count"].to(self.count.device))
        self.clip = float(state.get("clip", self.clip))
