"""Necessary geometry invariants; not proof of BF16 model equivalence."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from molt_stream.core.specs import DataSpec
from molt_stream.data.bytes import MMapTokenBatcher


@pytest.mark.parametrize("sequential", [True, False])
def test_geometry_preserves_windows_including_file_wrap(tmp_path, sequential):
    path = tmp_path / "tokens.bin"
    path.write_bytes(bytes(range(99)))
    spec = DataSpec(str(path), context_length=8, storage_dtype="uint8", sequential=sequential)
    first = MMapTokenBatcher(spec, seed=17, device="cpu")
    second = MMapTokenBatcher(replace(spec), seed=17, device="cpu")
    for _ in range(30):
        a = [first.batch(1) for _ in range(4)]
        b = [second.batch(2) for _ in range(2)]
        for index in (0, 1):
            assert torch.equal(torch.cat([item[index] for item in a]),
                               torch.cat([item[index] for item in b]))


def test_equal_token_weighted_microbatches_preserve_reference_adamw_update():
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(4, 8, 5, generator=generator, dtype=torch.float64)
    targets = torch.randint(0, 7, (4, 8), generator=generator)
    initial = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    results = []
    for batch in (1, 2, 4):
        weight = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([weight], lr=0.001)
        for start in range(0, 4, batch):
            loss = F.cross_entropy(
                (inputs[start:start + batch] @ weight).reshape(-1, 7),
                targets[start:start + batch].flatten(), reduction="sum",
            ) / targets.numel()
            loss.backward()
        optimizer.step()
        results.append(weight.detach())
    for result in results[1:]:
        torch.testing.assert_close(result, results[0], rtol=1e-12, atol=1e-12)
