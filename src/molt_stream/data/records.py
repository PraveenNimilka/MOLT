"""Bounded record-oriented preparation for JSONL and Parquet datasets."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from molt_stream.core.integrity import sha256
from molt_stream.data.text import _validate_model_token_ids


SUPPORTED_SCHEMAS = ("auto", "text", "messages", "prompt-completion")


def _jsonl_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            yield line_number, value


def _parquet_records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is not installed. Re-run install.ps1 or install MOLT with the 'data' extra."
        ) from exc
    row_number = 0
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256):
        for value in batch.to_pylist():
            row_number += 1
            if not isinstance(value, dict):
                raise ValueError(f"Parquet row {row_number} must decode to an object")
            yield row_number, value


def _records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _jsonl_records(path)
    if suffix == ".parquet":
        return _parquet_records(path)
    raise ValueError("Record preparation supports .jsonl and .parquet inputs")


def _record_count(path: Path) -> int:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support is not installed. Re-run install.ps1 or install MOLT with the 'data' extra."
            ) from exc
        return parquet.ParquetFile(path).metadata.num_rows
    raise ValueError("Record preparation supports .jsonl and .parquet inputs")


def _render_record(record: Mapping[str, Any], tokenizer: Any, schema: str,
                   text_column: str, messages_column: str,
                   prompt_column: str, completion_column: str,
                   row_number: int) -> tuple[str, str]:
    selected = schema
    if selected == "auto":
        if isinstance(record.get(text_column), str):
            selected = "text"
        elif isinstance(record.get(messages_column), list):
            selected = "messages"
        elif isinstance(record.get(prompt_column), str) and isinstance(record.get(completion_column), str):
            selected = "prompt-completion"
        else:
            raise ValueError(
                f"Cannot infer schema at row {row_number}; expected '{text_column}', "
                f"'{messages_column}', or '{prompt_column}' plus '{completion_column}'"
            )
    if selected == "text":
        value = record.get(text_column)
        if not isinstance(value, str):
            raise ValueError(f"Row {row_number} column '{text_column}' must be text")
        return value, selected
    if selected == "messages":
        messages = record.get(messages_column)
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {row_number} column '{messages_column}' must be a non-empty message list")
        for message in messages:
            if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)):
                raise ValueError(f"Row {row_number} has an invalid role/content message")
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("The selected tokenizer has no chat template; use a compatible chat model or text schema")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False), selected
    if selected == "prompt-completion":
        prompt, completion = record.get(prompt_column), record.get(completion_column)
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError(
                f"Row {row_number} columns '{prompt_column}' and '{completion_column}' must be text"
            )
        if getattr(tokenizer, "chat_template", None):
            value = tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ], tokenize=False, add_generation_prompt=False)
        else:
            value = prompt + "\n" + completion
        return value, selected
    raise ValueError(f"schema must be one of: {', '.join(SUPPORTED_SCHEMAS)}")


def _write_training_config(package: Path, destination: Path, model_path: Path,
                           train_tokens: int, validation_tokens: int) -> None:
    from molt_stream.core.workspace import workspace_paths
    runs = workspace_paths("runs")
    value = {
        "mode": "qlora", "base_model": str(model_path),
        "data": {
            "path": str(destination / "train.bin"),
            "validation_path": str(destination / "validation.bin"),
            "context_length": min(128, train_tokens - 1, validation_tokens - 1),
            "storage_dtype": "int32", "sequential": True, "packing": "contiguous",
        },
        "max_steps": 20, "batch_size": 1,
        "artifacts_dir": str(runs[0] if runs else destination / "runs"),
    }
    (package / "training.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare_records(source: str, tokenizer_path: str, output_dir: str,
                    validation_fraction: float = 0.1, base_model: str | None = None,
                    *, schema: str = "auto", text_column: str = "text",
                    messages_column: str = "messages", prompt_column: str = "prompt",
                    completion_column: str = "completion") -> dict[str, object]:
    """Tokenize records with a deterministic document-level holdout and bounded memory."""
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"schema must be one of: {', '.join(SUPPORTED_SCHEMAS)}")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    from transformers import AutoTokenizer
    input_path, destination = Path(source).resolve(), Path(output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {destination}")
    # Security boundary: tokenizer_path is local and network access is disabled.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        tokenizer_path, local_files_only=True
    )
    model_path = Path(base_model).resolve() if base_model else None
    model_vocab_size = None
    if model_path is not None:
        from transformers import AutoConfig
        # Security boundary: model_path is local and network access is disabled.
        config = AutoConfig.from_pretrained(  # nosec B615
            model_path, local_files_only=True
        )
        model_vocab_size = int(config.vocab_size)

    row_count = _record_count(input_path)
    validation_rows = max(1, int(row_count * validation_fraction))
    training_rows = row_count - validation_rows
    if training_rows < 1:
        raise ValueError("At least two non-empty records are required for a document-level split")

    destination.parent.mkdir(parents=True, exist_ok=True)
    detected_schemas: set[str] = set()
    token_counts = {"train": 0, "validation": 0}
    eos = tokenizer.eos_token_id
    with tempfile.TemporaryDirectory(prefix="molt-prepare-", dir=destination.parent) as temporary:
        package = Path(temporary) / "dataset"
        package.mkdir()
        train_file = (package / "train.bin").open("wb")
        validation_file = (package / "validation.bin").open("wb")
        try:
            for index, (row_number, record) in enumerate(_records(input_path)):
                rendered, detected = _render_record(
                    record, tokenizer, schema, text_column, messages_column,
                    prompt_column, completion_column, row_number,
                )
                detected_schemas.add(detected)
                if len(detected_schemas) > 1:
                    raise ValueError("Auto-detected record schemas are mixed; select one --schema explicitly")
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                if eos is not None and (not ids or ids[-1] != eos):
                    ids.append(eos)
                _validate_model_token_ids(ids, model_vocab_size)
                split = "train" if index < training_rows else "validation"
                np.asarray(ids, dtype="<i4").tofile(train_file if split == "train" else validation_file)
                token_counts[split] += len(ids)
        finally:
            train_file.close()
            validation_file.close()
        if min(token_counts.values()) < 2:
            raise ValueError("Not enough tokens for separate training and validation data")
        tokenizer.save_pretrained(package / "tokenizer")
        metadata = {
            "schema_version": 2, "source": str(input_path), "source_sha256": sha256(input_path),
            "source_format": input_path.suffix.lower().lstrip("."),
            "record_schema": sorted(detected_schemas), "storage_dtype": "int32",
            "train_records": training_rows, "validation_records": validation_rows,
            "train_tokens": token_counts["train"], "validation_tokens": token_counts["validation"],
            "vocab_size": model_vocab_size or len(tokenizer),
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer": str(destination / "tokenizer"),
            "split": "deterministic contiguous record tail; documents do not cross the split",
            "train_sha256": sha256(package / "train.bin"),
            "validation_sha256": sha256(package / "validation.bin"),
        }
        (package / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if model_path is not None:
            _write_training_config(package, destination, model_path, *token_counts.values())
        os.rename(package, destination)
    return {
        "directory": str(destination),
        "training_config": str(destination / "training.json") if model_path is not None else None,
        **metadata,
    }
