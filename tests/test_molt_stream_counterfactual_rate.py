from __future__ import annotations

import copy

import pytest
import torch

from molt_stream.methods.counterfactual_rate import counterfactual_optimizer_step


def _loss(first: torch.nn.Parameter, second: torch.nn.Parameter) -> torch.Tensor:
    return (first.square().sum() + 3.0 * second.square().sum())


def test_counterfactual_step_matches_fresh_adamw_commit_exactly():
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    second = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    optimizer = torch.optim.AdamW(
        [{"params": [first], "lr": 1e-3}, {"params": [second], "lr": 1e-3}],
        weight_decay=0.01,
    )
    _loss(first, second).backward()
    initial_parameters = (first.detach().clone(), second.detach().clone())
    initial_gradients = (first.grad.detach().clone(), second.grad.detach().clone())
    candidates = ((1e-3, 2e-3), (1e-3, 4e-3), (1e-3, 8e-3))

    result = counterfactual_optimizer_step(
        optimizer,
        candidates,
        lambda: float(_loss(first, second).detach()),
    )

    reference_first = torch.nn.Parameter(initial_parameters[0].clone())
    reference_second = torch.nn.Parameter(initial_parameters[1].clone())
    reference_first.grad = initial_gradients[0].clone()
    reference_second.grad = initial_gradients[1].clone()
    reference = torch.optim.AdamW(
        [
            {"params": [reference_first], "lr": result.selected_learning_rates[0]},
            {"params": [reference_second], "lr": result.selected_learning_rates[1]},
        ],
        weight_decay=0.01,
    )
    reference.step()

    torch.testing.assert_close(first, reference_first, rtol=0, atol=0)
    torch.testing.assert_close(second, reference_second, rtol=0, atol=0)
    assert result.scores[result.selected_index] == min(result.scores)


def test_counterfactual_step_restores_nonempty_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameter.square().sum().backward()
    before = copy.deepcopy(optimizer.state_dict())

    counterfactual_optimizer_step(
        optimizer,
        ((1e-3,), (2e-3,)),
        lambda: float(parameter.square().sum().detach()),
    )

    assert int(optimizer.state[parameter]["step"]) == int(before["state"][0]["step"]) + 1


def test_counterfactual_step_rolls_back_on_invalid_score():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.square().sum().backward()
    original = parameter.detach().clone()

    with pytest.raises(ValueError, match="finite"):
        counterfactual_optimizer_step(optimizer, ((1e-3,),), lambda: float("nan"))

    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert not optimizer.state
