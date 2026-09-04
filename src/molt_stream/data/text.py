"""Bounded UTF-8 text tokenization with explicit contiguous hold-out semantics."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from molt_stream.core.integrity import sha256


def prepare_text(source: str, tokenizer_path: str, output_dir: str,
                 validation_fraction: float = 0.1, base_model: str | None = None) -> dict[str, object]:
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    from transformers import AutoTokenizer
    input_path, destination = Path(source).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {destination}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model_path = None
    if base_model:
        from transformers import AutoConfig
        model_path = Path(base_model).resolve()
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if max(tokenizer.get_vocab().values()) >= config.vocab_size:
            raise ValueError("Tokenizer IDs exceed the base model vocabulary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="molt-prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        package = stage / "dataset"
        package.mkdir()
        combined = stage / "tokens.bin"
        count = 0
        with input_path.open("r", encoding="utf-8") as text, combined.open("wb") as binary:
            while chunk := text.read(65536):
                ids = tokenizer.encode(chunk, add_special_tokens=False)
                np.asarray(ids, dtype="<i4").tofile(binary)
                count += len(ids)
        validation_count = max(1, int(count * validation_fraction))
        training_count = count - validation_count
        if min(training_count, validation_count) < 2:
            raise ValueError("Not enough tokens for separate training and validation data")
        with combined.open("rb") as binary:
            for name, tokens in (("train.bin", training_count), ("validation.bin", validation_count)):
                remaining = tokens * 4
                with (package / name).open("wb") as output:
                    while remaining:
                        block = binary.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OSError("Unexpected end of prepared token stream")
                        output.write(block)
                        remaining -= len(block)
        tokenizer.save_pretrained(package / "tokenizer")
        metadata = {"schema_version": 1, "source": str(input_path),
                    "source_sha256": sha256(input_path), "storage_dtype": "int32",
                    "train_tokens": training_count, "validation_tokens": validation_count,
                    "vocab_size": len(tokenizer), "tokenizer": str(destination / "tokenizer"),
                    "split": "contiguous token tail; no shuffle; not a document-level split",
                    "tokenization": "UTF-8 chunks of at most 65536 characters, without special tokens",
                    "train_sha256": sha256(package / "train.bin"),
                    "validation_sha256": sha256(package / "validation.bin")}
        (package / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if model_path is not None:
            from molt_stream.core.workspace import workspace_paths
            runs = workspace_paths("runs")
            training = {
                "mode": "qlora", "base_model": str(model_path),
                "data": {"path": str(destination / "train.bin"),
                         "validation_path": str(destination / "validation.bin"),
                         "context_length": min(128, training_count - 1, validation_count - 1),
                         "storage_dtype": "int32"},
                "max_steps": 20, "batch_size": 1,
                "artifacts_dir": str(runs[0] if runs else destination / "runs"),
            }
            (package / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
        os.rename(package, destination)
    return {"directory": str(destination), "training_config": str(destination / "training.json")
            if model_path is not None else None, **metadata}
