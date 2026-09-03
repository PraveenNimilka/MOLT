"""Minimal self-contained example of using MOLT programmatically."""
from pathlib import Path
from molt_stream.core.specs import TrainingSpec, ModelSpec, DataSpec, StreamSpec
from molt_stream.core.profiles import apply_profile
from molt_stream.training.engine import train


def main():
    # 1. Define base specification
    spec = TrainingSpec(
        mode="pretrain",
        data=DataSpec(
            path="data/prepared/smoke/train.bin",
            context_length=32,
            storage_dtype="uint8",
        ),
        model=ModelSpec(
            context_length=32,
            layers=2,
            width=128,
            heads=4,
            hidden_width=384,
            vocab_size=256,
        ),
        stream=StreamSpec(
            device="cuda",
            compute_dtype="bfloat16",
        ),
        batch_size=8,
        gradient_accumulation=1,
        max_steps=20,
        learning_rate=0.001,
        seed=1337,
        artifacts_dir="artifacts/molt-stream/runs",
    )

    # 2. Apply a high-level policy profile (SPEED, BALANCED, COOL, ENERGY)
    spec = apply_profile(spec, "balanced")

    # 3. Execute training with closed-loop telemetry
    print("Starting MOLT training with BALANCED profile...")
    completed_run_dir = train(spec)
    print(f"Training completed successfully! Artifacts: {completed_run_dir}")


if __name__ == "__main__":
    main()
