import torch

from molt_stream.training.galore import GaLoreAdamW


def test_full_rank_path_matches_torch_adamw_one_step():
    initial = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    left = torch.nn.Parameter(initial.clone())
    right = torch.nn.Parameter(initial.clone())
    left.grad = torch.tensor([[0.2, -0.3], [0.4, -0.5]])
    right.grad = left.grad.clone()
    reference = torch.optim.AdamW([left], lr=1e-2, weight_decay=0.1)
    candidate = GaLoreAdamW([right], lr=1e-2, rank=2, weight_decay=0.1)
    reference.step()
    candidate.step()
    assert torch.allclose(left, right, atol=2e-7, rtol=2e-7)


def test_low_rank_state_is_smaller_than_full_matrix():
    parameter = torch.nn.Parameter(torch.randn(32, 64))
    optimizer = GaLoreAdamW([parameter], lr=1e-3, rank=4, update_gap=10)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["exp_avg"].numel() == 4 * 64
    assert state["exp_avg_sq"].numel() == 4 * 64
