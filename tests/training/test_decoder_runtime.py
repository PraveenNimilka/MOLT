from __future__ import annotations

import pytest
import torch

from molt_stream.training.decoder_runtime import static_decoder_forward


def _models():
    from transformers import Gemma2Config, Gemma2Model
    from transformers import LlamaConfig, LlamaModel
    from transformers import Qwen2Config, Qwen2Model

    common = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
    }
    return (
        Qwen2Model(Qwen2Config(**common)),
        LlamaModel(LlamaConfig(**common)),
        Gemma2Model(
            Gemma2Config(
                **common,
                head_dim=8,
                sliding_window=32,
            )
        ),
    )


@pytest.mark.parametrize("model", _models())
def test_explicit_decoder_traversal_matches_transformers(model) -> None:
    model.eval()
    tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    with torch.no_grad():
        reference = model(input_ids=tokens, use_cache=False).last_hidden_state
        candidate = static_decoder_forward(model, tokens)
    torch.testing.assert_close(candidate, reference, rtol=0.0, atol=0.0)


def test_explicit_decoder_traversal_preserves_input_gradient() -> None:
    model = _models()[0]
    model.eval()
    tokens = torch.tensor([[2, 3, 4, 5]])
    reference = model(input_ids=tokens, use_cache=False).last_hidden_state.square().mean()
    reference.backward()
    expected = model.embed_tokens.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    candidate = static_decoder_forward(model, tokens).square().mean()
    candidate.backward()
    torch.testing.assert_close(
        model.embed_tokens.weight.grad, expected, rtol=0.0, atol=0.0
    )
