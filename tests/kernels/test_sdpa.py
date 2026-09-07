from types import SimpleNamespace

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.kernels.sdpa import _bf16_sdpa_forward, enable_bf16_fused_sdpa_boundary

pytest.importorskip("transformers")


def test_bf16_sdpa_boundary_preserves_shape_and_narrows_promoted_qk():
    module = SimpleNamespace(is_causal=True)
    q = torch.randn(1, 2, 8, 4, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)

    output, weights = _bf16_sdpa_forward(module, q, k, v, None)

    assert output.shape == (1, 8, 2, 4)
    assert output.dtype == torch.bfloat16
    assert weights is None


def test_bf16_sdpa_boundary_expands_grouped_kv_heads():
    module = SimpleNamespace(
        is_causal=True,
        num_key_value_groups=2,
        config=SimpleNamespace(_molt_sdpa_kernel="efficient"),
    )
    q = torch.randn(1, 4, 8, 4, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)

    output, _ = _bf16_sdpa_forward(module, q, k, v, None)

    assert output.shape == (1, 8, 4, 4)


def test_bf16_sdpa_boundary_rejects_non_sdpa_model():
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen2", _attn_implementation="eager")
    )
    with pytest.raises(CapabilityError, match="loaded with SDPA"):
        enable_bf16_fused_sdpa_boundary(model, "cudnn")
