import pytest

from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import DataSpec, TrainingMode, TrainingSpec
from molt_stream.training.qlora import train_qlora


def test_qlora_refuses_missing_optional_runtime(tmp_path):
    tokens = tmp_path / "tokens.bin"
    tokens.write_bytes(bytes(range(64)))
    spec = TrainingSpec(
        mode=TrainingMode.QLORA,
        data=DataSpec(str(tokens), context_length=8, storage_dtype="uint8"),
        base_model="unavailable/model",
    )
    with pytest.raises(CapabilityError, match="missing optional packages"):
        train_qlora(spec)
