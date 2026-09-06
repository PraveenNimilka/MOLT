"""Optimizer selection must preserve FP32 states and AdamW update semantics."""

from dataclasses import replace

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import DataSpec, StreamSpec, TrainingMode, TrainingSpec
from molt_stream.training.qlora import _build_qlora_optimizer


@pytest.fixture
def optimizer_spec(tmp_path):
    tokens = tmp_path / "tokens.bin"
    tokens.write_bytes(bytes(range(64)))
    return TrainingSpec(
        mode=TrainingMode.QLORA,
        data=DataSpec(str(tokens), context_length=8, storage_dtype="uint8"),
        base_model="test-model",
    )


def test_fused_optimizer_is_opt_in_and_round_trips(optimizer_spec):
    assert optimizer_spec.qlora_fused_optimizer is False
    fused = replace(optimizer_spec, qlora_fused_optimizer=True)
    fused.validate()
    assert TrainingSpec.from_dict(fused.to_dict()) == fused


def test_qlora_autocast_is_opt_in_and_round_trips(optimizer_spec):
    assert optimizer_spec.qlora_autocast is False
    enabled = replace(optimizer_spec, qlora_autocast=True)
    enabled.validate()
    assert TrainingSpec.from_dict(enabled.to_dict()) == enabled


def test_qlora_loss_backend_is_explicit_and_round_trips(optimizer_spec):
    assert optimizer_spec.qlora_loss_backend == "analytical"
    triton = replace(
        optimizer_spec,
        qlora_loss_chunk_size=8,
        qlora_precompute_head_gradient=True,
        qlora_loss_backend="triton",
    )
    triton.validate()
    assert TrainingSpec.from_dict(triton.to_dict()) == triton


def test_qlora_loss_backend_rejects_invalid_or_inert_configuration(optimizer_spec):
    with pytest.raises(ValueError, match="must be analytical"):
        replace(optimizer_spec, qlora_loss_backend="magic").validate()
    with pytest.raises(ValueError, match="requires qlora_precompute_head_gradient"):
        replace(optimizer_spec, qlora_loss_backend="triton").validate()


def test_qlora_attention_backend_is_validated_and_round_trips(optimizer_spec):
    assert optimizer_spec.qlora_attention_backend == "sdpa"
    eager = replace(optimizer_spec, qlora_attention_backend="eager")
    eager.validate()
    assert TrainingSpec.from_dict(eager.to_dict()) == eager
    with pytest.raises(ValueError, match="must be sdpa or eager"):
        replace(optimizer_spec, qlora_attention_backend="flash-magic").validate()




@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_qlora_autocast_rejects_non_boolean_configuration(optimizer_spec, invalid):
    config = optimizer_spec.to_dict()
    config["qlora_autocast"] = invalid
    with pytest.raises(ValueError, match="must be a boolean"):
        TrainingSpec.from_dict(config)


def test_qlora_autocast_requires_qlora_cuda_spec(optimizer_spec):
    with pytest.raises(ValueError, match="requires QLoRA on CUDA"):
        replace(
            optimizer_spec, qlora_autocast=True, stream=StreamSpec(device="cpu")
        ).validate()
    with pytest.raises(ValueError, match="requires QLoRA on CUDA"):
        replace(optimizer_spec, qlora_autocast=True, mode=TrainingMode.PRETRAIN).validate()


def test_activation_offload_is_explicit_and_excludes_checkpointing(optimizer_spec):
    assert optimizer_spec.qlora_activation_offload is False
    enabled = replace(
        optimizer_spec,
        qlora_activation_offload=True,
        activation_checkpointing=False,
    )
    assert TrainingSpec.from_dict(enabled.to_dict()).qlora_activation_offload is True
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(enabled, activation_checkpointing=True).validate()


def test_activation_compression_is_explicit_and_excludes_other_storage_modes(optimizer_spec):
    assert optimizer_spec.qlora_activation_compression_bits is None
    enabled = replace(
        optimizer_spec,
        qlora_activation_compression_bits=4,
        activation_checkpointing=False,
    )
    assert TrainingSpec.from_dict(enabled.to_dict()).qlora_activation_compression_bits == 4
    with pytest.raises(ValueError, match="excludes checkpointing"):
        replace(enabled, activation_checkpointing=True).validate()
    with pytest.raises(ValueError, match="null or 4"):
        replace(enabled, qlora_activation_compression_bits=8).validate()


def test_frozen_head_cache_policy_round_trips(optimizer_spec):
    assert optimizer_spec.qlora_cache_frozen_head is True
    disabled = replace(optimizer_spec, qlora_cache_frozen_head=False)
    assert TrainingSpec.from_dict(disabled.to_dict()).qlora_cache_frozen_head is False


def test_bf16_tied_embedding_policy_requires_autocast_qlora(optimizer_spec):
    enabled = replace(optimizer_spec, qlora_restore_tied_embedding_bf16=True)
    assert TrainingSpec.from_dict(enabled.to_dict()).qlora_restore_tied_embedding_bf16 is True
    with pytest.raises(ValueError, match="require autocast QLoRA"):
        replace(enabled, qlora_autocast=False).validate()


def test_bf16_frozen_rmsnorm_policy_requires_autocast_qlora(optimizer_spec):
    enabled = replace(optimizer_spec, qlora_frozen_rmsnorm_bf16=True)
    assert TrainingSpec.from_dict(enabled.to_dict()).qlora_frozen_rmsnorm_bf16 is True
    with pytest.raises(ValueError, match="requires autocast QLoRA"):
        replace(enabled, qlora_autocast=False).validate()


def test_microbatch_guard_must_fit_between_cruise_and_abort(optimizer_spec):
    enabled = replace(
        optimizer_spec,
        thermal_target_c=67.0,
        thermal_microbatch_guard_c=70.0,
        thermal_abort_c=72.0,
    )
    restored = TrainingSpec.from_dict(enabled.to_dict())
    restored.validate()
    assert restored.thermal_microbatch_guard_c == 70.0

    with pytest.raises(ValueError, match="thermal_microbatch_guard_c"):
        replace(enabled, thermal_microbatch_guard_c=66.0).validate()
    with pytest.raises(ValueError, match="thermal_microbatch_guard_c"):
        replace(enabled, thermal_microbatch_guard_c=72.0).validate()


def test_intra_step_layer_guard_round_trip_and_validation(optimizer_spec):
    enabled = replace(
        optimizer_spec,
        qlora_intra_step_boundary_layer=14,
        qlora_intra_step_pause_ms=10.0,
    )
    restored = TrainingSpec.from_dict(enabled.to_dict())
    assert restored.qlora_intra_step_boundary_layer == 14
    assert restored.qlora_intra_step_pause_ms == 10.0
    restored.validate()
    with pytest.raises(ValueError, match="requires a boundary layer"):
        replace(enabled, qlora_intra_step_boundary_layer=None).validate()


def test_scheduled_nf4_down_projection_requires_autocast_cuda_qlora(optimizer_spec):
    enabled = replace(
        optimizer_spec,
        qlora_autocast=True,
        qlora_scheduled_nf4_down_projection=True,
        stream=replace(optimizer_spec.stream, device="cuda"),
    )
    restored = TrainingSpec.from_dict(enabled.to_dict())
    restored.validate()
    assert restored.qlora_scheduled_nf4_down_projection is True

    with pytest.raises(ValueError, match="scheduled NF4 down projection"):
        replace(enabled, qlora_autocast=False).validate()


def test_bf16_adapter_shadows_require_autocast_cuda_qlora(optimizer_spec):
    enabled = replace(
        optimizer_spec,
        qlora_autocast=True,
        qlora_bf16_adapter_shadows=True,
        stream=replace(optimizer_spec.stream, device="cuda"),
    )
    restored = TrainingSpec.from_dict(enabled.to_dict())
    restored.validate()
    assert restored.qlora_bf16_adapter_shadows is True
    with pytest.raises(ValueError, match="BF16 adapter shadows"):
        replace(enabled, qlora_autocast=False).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(enabled, qlora_scheduled_nf4_down_projection=True).validate()


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_fused_optimizer_rejects_non_boolean_configuration(optimizer_spec, invalid):
    config = optimizer_spec.to_dict()
    config["qlora_fused_optimizer"] = invalid
    with pytest.raises(ValueError, match="must be a boolean"):
        TrainingSpec.from_dict(config)


def test_fused_optimizer_requires_qlora_cuda_spec(optimizer_spec):
    with pytest.raises(ValueError, match="requires QLoRA on CUDA"):
        replace(
            optimizer_spec, qlora_fused_optimizer=True, stream=StreamSpec(device="cpu")
        ).validate()
    with pytest.raises(ValueError, match="requires QLoRA on CUDA"):
        replace(
            optimizer_spec, qlora_fused_optimizer=True, mode=TrainingMode.PRETRAIN
        ).validate()


def test_default_optimizer_retains_fp32_states_and_excludes_frozen_weights(optimizer_spec):
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    optimizer = _build_qlora_optimizer(model, optimizer_spec)
    assert optimizer.defaults.get("fused") in (None, False)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    assert model.bias not in optimizer.state
    assert optimizer.state[model.weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[model.weight]["exp_avg_sq"].dtype == torch.float32


def test_fused_optimizer_refuses_cpu_parameters_without_fallback(optimizer_spec):
    with pytest.raises(CapabilityError):
        _build_qlora_optimizer(
            torch.nn.Linear(3, 2), replace(optimizer_spec, qlora_fused_optimizer=True)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_cuda_optimizer_matches_fp32_adamw_reference(optimizer_spec):
    initial = torch.linspace(-0.3, 0.3, 6, device="cuda").reshape(2, 3)
    baseline = torch.nn.Linear(3, 2, bias=False, device="cuda")
    candidate = torch.nn.Linear(3, 2, bias=False, device="cuda")
    with torch.no_grad():
        baseline.weight.copy_(initial)
        candidate.weight.copy_(initial)
    reference = _build_qlora_optimizer(baseline, optimizer_spec)
    fused = _build_qlora_optimizer(
        candidate, replace(optimizer_spec, qlora_fused_optimizer=True)
    )
    assert fused.defaults["fused"] is True
    for step in range(10):
        gradient = torch.sin(initial + step * 0.1)
        baseline.weight.grad = gradient.clone()
        candidate.weight.grad = gradient.clone()
        reference.step()
        fused.step()
        torch.testing.assert_close(candidate.weight, baseline.weight, rtol=1e-6, atol=1e-7)
        for name in ("exp_avg", "exp_avg_sq"):
            assert fused.state[candidate.weight][name].dtype == torch.float32
            torch.testing.assert_close(
                fused.state[candidate.weight][name], reference.state[baseline.weight][name],
                rtol=1e-6, atol=1e-7,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_optimizer_refuses_reduced_precision_adapter_states(optimizer_spec):
    model = torch.nn.Linear(3, 2, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(CapabilityError):
        _build_qlora_optimizer(model, replace(optimizer_spec, qlora_fused_optimizer=True))
