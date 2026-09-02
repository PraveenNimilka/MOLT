from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from molt_stream.core.errors import IntegrityError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class AtomicCheckpointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, stem: str) -> tuple[Path, Path]:
        return self.root / f"{stem}.pt", self.root / f"{stem}.complete.json"

    def save(self, state: dict[str, Any]) -> Path:
        target, metadata = self._paths("checkpoint")
        previous, previous_metadata = self._paths("checkpoint.previous")
        temporary = self.root / "checkpoint.pt.tmp"
        torch.save(state, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        record = {"schema_version": 1, "sha256": sha256(temporary), "bytes": temporary.stat().st_size}
        if target.exists():
            os.replace(target, previous)
            if metadata.exists():
                os.replace(metadata, previous_metadata)
        os.replace(temporary, target)
        _atomic_json(metadata, record)
        return target

    @staticmethod
    def _valid(data: Path, metadata: Path) -> bool:
        try:
            record = json.loads(metadata.read_text("utf-8"))
            return data.stat().st_size == record["bytes"] and sha256(data) == record["sha256"]
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False

    def resolve(self) -> Path:
        for stem in ("checkpoint", "checkpoint.previous"):
            data, metadata = self._paths(stem)
            if self._valid(data, metadata):
                return data
        raise IntegrityError(f"no valid checkpoint in {self.root}")

    def load(self, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        return torch.load(self.resolve(), map_location=map_location, weights_only=False)
