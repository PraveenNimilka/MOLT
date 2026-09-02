from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CounterfactualRateResult:
    """Result of an exact rollback-and-commit optimizer-rate probe."""

    selected_index: int
    selected_learning_rates: tuple[float, ...]
    scores: tuple[float, ...]
    probe_seconds: float


def _parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    identities: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in identities:
                raise ValueError("optimizer parameter appears in more than one group")
            identities.add(id(parameter))
            parameters.append(parameter)
    if not parameters:
        raise ValueError("optimizer contains no parameters")
    if any(parameter.grad is None for parameter in parameters):
        raise ValueError("every optimizer parameter must have a gradient before probing")
    return parameters


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    learning_rates: Sequence[float],
) -> None:
    if len(learning_rates) != len(optimizer.param_groups):
        raise ValueError("each candidate must provide one learning rate per parameter group")
    if any(not math.isfinite(rate) or rate <= 0 for rate in learning_rates):
        raise ValueError("candidate learning rates must be finite and positive")
    for group, rate in zip(optimizer.param_groups, learning_rates, strict=True):
        group["lr"] = float(rate)


def counterfactual_optimizer_step(
    optimizer: torch.optim.Optimizer,
    candidate_learning_rates: Sequence[Sequence[float]],
    score: Callable[[], float],
) -> CounterfactualRateResult:
    """Evaluate rate candidates from one gradient, then commit the best exactly.

    Parameters and optimizer state are restored before every candidate. The
    final committed update is therefore the same update a standalone optimizer
    would have produced from the pre-probe state at the selected rates. The
    score callback must be deterministic for a scientific comparison and must
    not mutate the optimizer or trainable parameters.
    """
    if not candidate_learning_rates:
        raise ValueError("at least one counterfactual rate candidate is required")
    parameters = _parameters(optimizer)
    parameter_state = [parameter.detach().clone() for parameter in parameters]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    original_learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)

    def restore() -> None:
        with torch.no_grad():
            for parameter, value in zip(parameters, parameter_state, strict=True):
                parameter.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))

    scores: list[float] = []
    started = time.perf_counter()
    try:
        for learning_rates in candidate_learning_rates:
            restore()
            _set_learning_rates(optimizer, learning_rates)
            optimizer.step()
            value = float(score())
            if not math.isfinite(value):
                raise ValueError("counterfactual score must be finite")
            scores.append(value)
        selected = min(range(len(scores)), key=scores.__getitem__)
        restore()
        selected_rates = tuple(float(value) for value in candidate_learning_rates[selected])
        _set_learning_rates(optimizer, selected_rates)
        optimizer.step()
    except BaseException:
        restore()
        _set_learning_rates(optimizer, original_learning_rates)
        raise
    return CounterfactualRateResult(
        selected_index=selected,
        selected_learning_rates=selected_rates,
        scores=tuple(scores),
        probe_seconds=time.perf_counter() - started,
    )
