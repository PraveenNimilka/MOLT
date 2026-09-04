"""Self-contained synthetic smoke workload, independent of the working directory."""
from pathlib import Path
import random
import uuid

from molt_stream.core.specs import DataSpec, ModelSpec, StreamSpec, TrainingMode, TrainingSpec


def create_smoke_spec(root: Path) -> TrainingSpec:
    """Keep generated inputs with artifacts for inspection; never overwrite user data."""
    workspace = root.resolve() / f"smoke-{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    rng = random.Random(1337)
    for name in ("train.bin", "validation.bin"):
        (workspace / name).write_bytes(rng.randbytes(8192))
    return TrainingSpec(
        mode=TrainingMode.PRETRAIN,
        data=DataSpec(path=str(workspace / "train.bin"),
                      validation_path=str(workspace / "validation.bin"),
                      context_length=32, storage_dtype="uint8"),
        model=ModelSpec(context_length=32, layers=2, width=128, heads=4,
                        hidden_width=384, vocab_size=256),
        stream=StreamSpec(device="cuda", compute_dtype="bfloat16"),
        batch_size=8, gradient_accumulation=1, max_steps=20,
        learning_rate=0.001, seed=1337, artifacts_dir=str(workspace / "runs"),
    )
