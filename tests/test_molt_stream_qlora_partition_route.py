import pytest
import torch

from molt_stream.training.qlora import _shifted_causal_loss
from molt_stream.core.errors import CapabilityError


def test_qwen_partition_route_matches_full_loss_and_decoder_gradients():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(17)
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=32, tie_word_embeddings=False)
    model = transformers.Qwen2ForCausalLM(config)
    model.get_output_embeddings().weight.requires_grad_(False)
    model.eval()
    inputs = torch.arange(8).unsqueeze(0)
    full = _shifted_causal_loss(model, inputs, inputs + 1)
    full.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    chunked = _shifted_causal_loss(model, inputs, inputs + 1, chunk_size=3)
    chunked.backward()
    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-6)
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-6, rtol=1e-5)


def test_qwen_partition_route_refuses_a_trainable_head():
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(vocab_size=64,
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2))
    with pytest.raises(CapabilityError, match="frozen"):
        _shifted_causal_loss(model, torch.zeros(1, 4, dtype=torch.long),
                             torch.ones(1, 4, dtype=torch.long), chunk_size=2)


@pytest.mark.parametrize("family", ["llama", "gemma"])
def test_partition_route_supports_other_registered_decoder_families(family):
    transformers = pytest.importorskip("transformers")
    config_class = {
        "llama": transformers.LlamaConfig,
        "gemma": transformers.GemmaConfig,
    }[family]
    model_class = {
        "llama": transformers.LlamaForCausalLM,
        "gemma": transformers.GemmaForCausalLM,
    }[family]
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = model_class(config)
    model.get_output_embeddings().weight.requires_grad_(False)
    inputs = torch.arange(8).unsqueeze(0)

    full = _shifted_causal_loss(model, inputs, inputs + 1)
    chunked = _shifted_causal_loss(model, inputs, inputs + 1, chunk_size=7)

    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-6)
