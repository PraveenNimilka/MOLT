import random

import pytest
import torch

from molt_stream.core.errors import IntegrityError
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.core.specs import DataSpec, ModelSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.training.model import SmallCausalLM


def test_checkpoint_recovers_previous_after_primary_corruption(tmp_path):
    store = AtomicCheckpointStore(tmp_path)
    store.save({"version": 1})
    store.save({"version": 2})
    (tmp_path / "checkpoint.pt").write_bytes(b"corrupt")
    assert store.load()["version"] == 1


def test_checkpoint_preserves_rng_continuation(tmp_path):
    random.seed(11)
    torch.manual_seed(11)
    store = AtomicCheckpointStore(tmp_path)
    store.save({"python_rng": random.getstate(), "torch_rng": torch.get_rng_state()})
    expected = (random.random(), torch.rand(4))
    state = store.load()
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    actual = (random.random(), torch.rand(4))
    assert expected[0] == actual[0]
    assert torch.equal(expected[1], actual[1])


def test_checkpoint_rejects_when_both_generations_invalid(tmp_path):
    store = AtomicCheckpointStore(tmp_path)
    with pytest.raises(IntegrityError):
        store.resolve()


def test_thermal_abort_checkpoint_is_complete_and_recoverable(tmp_path):
    store = AtomicCheckpointStore(tmp_path)
    store.save({"step": 7, "termination_reason": "thermal_abort"})
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "checkpoint.complete.json").is_file()
    restored = store.load()
    assert restored == {"step": 7, "termination_reason": "thermal_abort"}


def test_model_optimizer_data_and_rng_resume_are_bit_exact(tmp_path):
    token_path = tmp_path / "tokens.bin"
    token_path.write_bytes(bytes(range(256)) * 4)
    data_spec = DataSpec(str(token_path), context_length=8, storage_dtype="uint8")
    model_spec = ModelSpec(vocab_size=256, context_length=8, layers=1, width=16, heads=2, hidden_width=32)

    def build():
        model = SmallCausalLM(model_spec)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batcher = MMapTokenBatcher(data_spec, seed=5, device="cpu")
        return model, optimizer, batcher

    random.seed(5)
    torch.manual_seed(5)
    model, optimizer, batcher = build()
    x, y = batcher.batch(2)
    loss = model.loss(x, y)
    loss.backward()
    optimizer.step()
    store = AtomicCheckpointStore(tmp_path / "run")
    store.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "batcher": batcher.state_dict(), "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    })

    optimizer.zero_grad(set_to_none=True)
    expected_batch = batcher.batch(2)
    expected_loss = model.loss(*expected_batch)
    expected_loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    restored_model, restored_optimizer, restored_batcher = build()
    state = store.load()
    restored_model.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    restored_batcher.load_state_dict(state["batcher"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    restored_optimizer.zero_grad(set_to_none=True)
    actual_batch = restored_batcher.batch(2)
    actual_loss = restored_model.loss(*actual_batch)
    actual_loss.backward()
    restored_optimizer.step()
    assert torch.equal(expected_batch[0], actual_batch[0])
    assert torch.equal(expected_batch[1], actual_batch[1])
    assert expected_loss.item() == actual_loss.item()
    for name, value in restored_model.state_dict().items():
        assert torch.equal(expected[name], value), name
