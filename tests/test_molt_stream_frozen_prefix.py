import copy

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.training.frozen_prefix import FrozenPrefixReplay


def tiny_model():
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.0))
    model.requires_grad_(False)
    model.model.layers[2].requires_grad_(True)
    return model


def forward(model, x):
    return model.model(input_ids=x, use_cache=False, return_dict=True).last_hidden_state


@pytest.mark.parametrize("checkpoint", [False, True])
def test_hit_preserves_output_gradients_and_rng_and_restores_methods(checkpoint):
    torch.manual_seed(2)
    baseline = tiny_model()
    if checkpoint:
        baseline.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    candidate = copy.deepcopy(baseline)
    x = torch.arange(8).unsqueeze(0)
    original = candidate.model.layers[0].forward
    with FrozenPrefixReplay(candidate, boundary=2, max_bytes=4096) as cache:
        cold = forward(candidate, x)
        rng = torch.get_rng_state().clone()
        hot = forward(candidate, x)
        assert torch.equal(cold, hot)
        assert torch.equal(rng, torch.get_rng_state())
        reference = forward(baseline, x)
        reference.square().sum().backward()
        hot.square().sum().backward()
        for (name, a), (_, b) in zip(baseline.named_parameters(), candidate.named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
        assert cache.hits == 1 and cache.misses == 1
        assert cache.resident_bytes == 8 * 32 * 4
    assert candidate.model.layers[0].forward == original
    assert cache.resident_bytes == 0


def test_cache_is_bounded_and_misses_after_eviction():
    model = tiny_model()
    with FrozenPrefixReplay(model, boundary=2, max_bytes=1024) as cache:
        x = torch.arange(8).unsqueeze(0)
        forward(model, x)
        forward(model, x + 1)
        forward(model, x)
        assert cache.hits == 0 and cache.misses == 3
        assert cache.resident_bytes <= 1024


def test_prefix_updates_are_rejected_not_served_from_stale_cache():
    model = tiny_model()
    with FrozenPrefixReplay(model, boundary=2, max_bytes=4096):
        x = torch.arange(8).unsqueeze(0)
        forward(model, x)
        with torch.no_grad():
            model.model.layers[0].mlp.up_proj.weight.add_(0.1)
        with pytest.raises(CapabilityError, match="changed"):
            forward(model, x)


def test_trainable_prefix_and_stochastic_attention_are_rejected():
    model = tiny_model()
    model.model.layers[0].requires_grad_(True)
    with pytest.raises(CapabilityError, match="frozen"):
        FrozenPrefixReplay(model, boundary=2, max_bytes=4096)
    model.model.layers[0].requires_grad_(False)
    model.config.attention_dropout = 0.1
    with pytest.raises(CapabilityError, match="dropout"):
        FrozenPrefixReplay(model, boundary=2, max_bytes=4096)


def test_custom_masks_are_rejected_and_exception_restores_forward():
    model = tiny_model()
    original = model.model.forward
    with pytest.raises(CapabilityError, match="arguments"):
        with FrozenPrefixReplay(model, boundary=2, max_bytes=4096):
            model.model(input_ids=torch.arange(8).unsqueeze(0), attention_mask=torch.ones(1, 8))
    assert model.model.forward == original


def test_replay_preserves_repeated_adamw_updates():
    torch.manual_seed(41)
    reference = tiny_model()
    candidate = copy.deepcopy(reference)
    optimizers = [torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
                  for model in (reference, candidate)]
    x = torch.arange(8).unsqueeze(0)
    with FrozenPrefixReplay(candidate, boundary=2, max_bytes=4096) as cache:
        for _ in range(3):
            for model, optimizer in zip((reference, candidate), optimizers):
                optimizer.zero_grad(set_to_none=True)
                forward(model, x).square().sum().backward()
                optimizer.step()
            for a, b in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert cache.hits == 2


def test_too_small_cache_does_not_silently_reduce_workload():
    model = tiny_model()
    x = torch.arange(8).unsqueeze(0)
    with FrozenPrefixReplay(model, boundary=2, max_bytes=1) as cache:
        torch.testing.assert_close(forward(model, x), forward(model, x), rtol=0, atol=0)
        assert cache.hits == 0 and cache.misses == 2
        assert cache.resident_bytes == 0
