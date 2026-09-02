from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class GaLoreAdamW(Optimizer):
    """Clean-room matrix-gradient projection research implementation.

    This favors correctness and inspectability over fused performance. Projection
    state is recomputed with SVD at a fixed interval; vector parameters use the
    ordinary full-rank AdamW equations.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        rank: int = 64,
        update_gap: int = 200,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr <= 0 or rank <= 0 or update_gap <= 0:
            raise ValueError("invalid GaLore hyperparameters")
        super().__init__(params, dict(lr=lr, rank=rank, update_gap=update_gap, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                state = self.state[parameter]
                step = int(state.get("step", 0)) + 1
                state["step"] = step
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                if gradient.ndim == 2 and group["rank"] < min(gradient.shape):
                    if "projection" not in state or (step - 1) % group["update_gap"] == 0:
                        u, _, vh = torch.linalg.svd(gradient, full_matrices=False)
                        state["left"] = gradient.shape[0] <= gradient.shape[1]
                        state["projection"] = (
                            u[:, : group["rank"]].contiguous()
                            if state["left"]
                            else vh[: group["rank"], :].contiguous()
                        )
                        state.pop("exp_avg", None)
                        state.pop("exp_avg_sq", None)
                    projection = state["projection"]
                    projected = projection.t() @ gradient if state["left"] else gradient @ projection.t()
                else:
                    projected = gradient
                    state["projection"] = None
                first = state.setdefault("exp_avg", torch.zeros_like(projected))
                second = state.setdefault("exp_avg_sq", torch.zeros_like(projected))
                first.mul_(beta1).add_(projected, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(projected, projected, value=1 - beta2)
                update = first / (1 - beta1**step)
                denominator = second.div(1 - beta2**step).sqrt_().add_(group["eps"])
                update = update / denominator
                projection = state["projection"]
                if projection is not None:
                    update = projection @ update if state["left"] else update @ projection
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
        return loss
