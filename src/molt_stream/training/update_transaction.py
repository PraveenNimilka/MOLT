"""Rollback input/RNG state before an uncommitted accumulated update.

Intended for pure-forward decoder training: optimizer.step must not have run,
and model forward must not mutate trainable parameters or persistent buffers.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Protocol

import torch


class StatefulBatcher(Protocol):
    def state_dict(self) -> dict[str, object]: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...


@dataclass(frozen=True)
class UpdateTransaction:
    batcher_state: dict[str, object]
    python_rng: Any
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None

    @classmethod
    def capture(cls, batcher: StatefulBatcher, *, cuda: bool) -> UpdateTransaction:
        return cls(batcher.state_dict(), random.getstate(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if cuda else None)

    def rollback(self, batcher: StatefulBatcher, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        batcher.load_state_dict(self.batcher_state)
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
