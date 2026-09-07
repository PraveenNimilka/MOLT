import pytest
import torch

from molt_stream.training.selective_checkpoint import configure_checkpoint_stride


@pytest.mark.parametrize("family", ["qwen", "llama", "gemma"])
def test_selective_checkpoint_preserves_loss_and_all_gradients(family):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(37)
    values = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )
    if family == "qwen":
        model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**values))
    elif family == "llama":
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**values))
    else:
        model = transformers.Gemma2ForCausalLM(
            transformers.Gemma2Config(**values, head_dim=8, sliding_window=32)
        )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    x = torch.arange(8).unsqueeze(0)
    reference = model(x, labels=x).loss
    reference.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    assert configure_checkpoint_stride(model, 2) == 2
    candidate = model(x, labels=x).loss
    candidate.backward()
    torch.testing.assert_close(reference, candidate)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name])


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_invalid_stride_is_rejected(stride):
    with pytest.raises(ValueError):
        configure_checkpoint_stride(torch.nn.Linear(2, 2), stride)
