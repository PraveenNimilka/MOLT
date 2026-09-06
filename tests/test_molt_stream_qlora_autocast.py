import torch
import pytest
from dataclasses import replace

from molt_stream.training.qlora import _shifted_causal_loss, _build_qlora_model
from molt_stream.training.qlora import _FROZEN_HEAD_COMPUTE_CACHE
from molt_stream.core.specs import DataSpec, TrainingMode, TrainingSpec
from molt_stream.training.qlora import train_qlora
from molt_stream.measurement.thermal import MicrobatchThermalDecision
from molt_stream.experiments.store import AtomicCheckpointStore


def test_autocast_uses_bfloat16_gemm_but_float32_loss_and_parameter_gradients():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)
            self.output_dtype = None

        def forward(self, *, input_ids, use_cache):
            logits = self.linear(torch.nn.functional.one_hot(input_ids, 4).float())
            self.output_dtype = logits.dtype
            return type("Output", (), {"logits": logits})()

    model = Model()
    inputs = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 0]])
    loss = _shifted_causal_loss(model, inputs, targets, autocast=True)
    loss.backward()
    assert model.output_dtype == torch.bfloat16
    assert loss.dtype == torch.float32
    assert model.linear.weight.grad.dtype == torch.float32
    assert torch.isfinite(model.linear.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nf4_lora_amp_with_nonreentrant_checkpoint_has_finite_gradients(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(17)
    config = transformers.Qwen2Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    base = tmp_path / "base"
    transformers.Qwen2ForCausalLM(config).save_pretrained(base)
    spec = TrainingSpec(mode=TrainingMode.QLORA, base_model=str(base),
                        data=DataSpec(str(tmp_path / "unused.bin"), context_length=8))
    model = _build_qlora_model(replace(spec, qlora_autocast=True))
    model.train()
    inputs = torch.arange(8, device="cuda").unsqueeze(0)
    loss = _shifted_causal_loss(model, inputs, inputs + 1, autocast=True)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert torch.isfinite(loss)
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_frozen_head_bf16_cache_is_nonpersistent_and_matches_master(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    config = transformers.Qwen2Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    base = tmp_path / "base"
    transformers.Qwen2ForCausalLM(config).save_pretrained(base)
    spec = TrainingSpec(
        mode=TrainingMode.QLORA,
        base_model=str(base),
        data=DataSpec(str(tmp_path / "unused.bin"), context_length=8),
        qlora_autocast=True,
        qlora_loss_chunk_size=8,
        qlora_precompute_head_gradient=True,
    )
    model = _build_qlora_model(spec)
    cache = getattr(model, _FROZEN_HEAD_COMPUTE_CACHE)
    master = model.get_base_model().get_output_embeddings().weight
    assert cache.dtype == torch.bfloat16
    torch.testing.assert_close(cache, master.to(torch.bfloat16), rtol=0, atol=0)
    assert _FROZEN_HEAD_COMPUTE_CACHE not in model.state_dict()




@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nf4_partial_update_stop_and_resume_matches_uninterrupted(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(17)
    config = transformers.Qwen2Config(vocab_size=256, hidden_size=128,
        intermediate_size=256, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64)
    base = tmp_path / "base"
    transformers.Qwen2ForCausalLM(config).save_pretrained(base)
    tokens = tmp_path / "tokens.bin"
    tokens.write_bytes(bytes(range(128)))
    spec = TrainingSpec(mode=TrainingMode.QLORA, base_model=str(base),
        data=DataSpec(str(tokens), context_length=8, storage_dtype="uint8"),
        max_steps=2, gradient_accumulation=2, qlora_autocast=True,
        artifacts_dir=str(tmp_path / "runs"), thermal_target_c=80, thermal_abort_c=85)
    safe = lambda *a, **k: MicrobatchThermalDecision(0., 40.)
    monkeypatch.setattr("molt_stream.training.qlora.wait_for_thermal_headroom", safe)
    reference = AtomicCheckpointStore(train_qlora(spec)).load()
    calls = [0]
    def interrupt(*args, **kwargs):
        calls[0] += 1
        # Eight validation checks, then pre-forward/post-forward/post-backward.
        return MicrobatchThermalDecision(0., 85., "simulated thermal stop") if calls[0] == 11 else safe()
    monkeypatch.setattr("molt_stream.training.qlora.wait_for_thermal_headroom", interrupt)
    run = train_qlora(spec)
    partial = AtomicCheckpointStore(run).load()
    assert partial["step"] == partial["tokens"] == 0
    assert partial["batcher"]["cursor"] == 0
    monkeypatch.setattr("molt_stream.training.qlora.wait_for_thermal_headroom", safe)
    resumed = AtomicCheckpointStore(train_qlora(spec, resume=run)).load()
    for name, tensor in reference["adapters"].items():
        assert torch.equal(tensor, resumed["adapters"][name])
    assert torch.equal(reference["torch_rng"], resumed["torch_rng"])
    assert all(torch.equal(a, b) for a, b in zip(reference["cuda_rng"], resumed["cuda_rng"]))
    assert reference["tokens"] == resumed["tokens"]
    assert reference["python_rng"] == resumed["python_rng"]
    assert reference["batcher"]["cursor"] == resumed["batcher"]["cursor"]
    for key, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            other = resumed["optimizer"]["state"][key][name]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
