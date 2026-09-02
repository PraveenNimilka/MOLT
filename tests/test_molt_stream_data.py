from array import array

import psutil
import pytest
import torch

from molt_stream.core.specs import DataSpec
from molt_stream.data.bytes import MMapTokenBatcher


def test_contiguous_mmap_batches_and_state_resume(tmp_path):
    path = tmp_path / "tokens.bin"
    values = array("i", range(1000))
    with path.open("wb") as handle:
        values.tofile(handle)
    spec = DataSpec(str(path), context_length=8, packing="contiguous")
    before = psutil.Process().memory_info().rss
    first = MMapTokenBatcher(spec, seed=7, device="cpu")
    x1, y1 = first.batch(3)
    state = first.state_dict()
    expected = first.batch(3)
    second = MMapTokenBatcher(spec, seed=999, device="cpu")
    second.load_state_dict(state)
    actual = second.batch(3)
    assert torch.equal(x1[:, 1:], y1[:, :-1])
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert first.storage_backend == "mmap"
    assert psutil.Process().memory_info().rss - before < 100_000_000


def test_runtime_context_preserves_contiguous_token_order(tmp_path):
    path = tmp_path / "tokens.bin"
    values = array("i", range(1000))
    with path.open("wb") as handle:
        values.tofile(handle)
    spec = DataSpec(str(path), context_length=16, packing="contiguous")
    batcher = MMapTokenBatcher(spec, seed=7, device="cpu")

    short_x, short_y = batcher.batch(4, context_length=4)
    full_x, full_y = batcher.batch(1, context_length=16)

    assert short_x.flatten().tolist() == list(range(16))
    assert short_y.flatten().tolist() == [*range(1, 5), *range(5, 9), *range(9, 13), *range(13, 17)]
    assert full_x.flatten().tolist() == list(range(16, 32))
    assert full_y.flatten().tolist() == list(range(17, 33))


def test_runtime_context_cannot_exceed_mapping_spec(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(bytes(range(128)))
    batcher = MMapTokenBatcher(
        DataSpec(str(path), context_length=8, storage_dtype="uint8"), seed=1, device="cpu"
    )
    with pytest.raises(ValueError, match="runtime context_length"):
        batcher.batch(1, context_length=9)
