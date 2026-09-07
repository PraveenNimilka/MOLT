from __future__ import annotations

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.training.qlora import _prepare_frozen_bf16_kbit_training


class TinyRMSNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))


class TinyQuantizedModel(torch.nn.Module):
    is_loaded_in_4bit = True

    def __init__(self, *, tied: bool = True) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        if tied:
            self.head.weight = self.embedding.weight
        self.norm = TinyRMSNorm()
        self.bias = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        self.checkpoint_kwargs = None

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs


def test_frozen_bf16_prepare_preserves_only_final_bf16_storage() -> None:
    model = TinyQuantizedModel()
    embedding_pointer = model.embedding.weight.data_ptr()
    norm_pointer = model.norm.weight.data_ptr()
    result = _prepare_frozen_bf16_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "preserve_rng_state": False,
        },
    )

    assert result is model
    assert model.embedding.weight.data_ptr() == embedding_pointer
    assert model.norm.weight.data_ptr() == norm_pointer
    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == torch.bfloat16
    assert model.bias.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.checkpoint_kwargs == {
        "use_reentrant": False,
        "preserve_rng_state": False,
    }


def test_frozen_bf16_prepare_rejects_untied_embeddings() -> None:
    with pytest.raises(CapabilityError, match="tied"):
        _prepare_frozen_bf16_kbit_training(
            TinyQuantizedModel(tied=False),
            use_gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )


def test_frozen_bf16_prepare_rejects_reentrant_checkpointing() -> None:
    with pytest.raises(CapabilityError, match="non-reentrant"):
        _prepare_frozen_bf16_kbit_training(
            TinyQuantizedModel(),
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
