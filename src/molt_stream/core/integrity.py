"""Filesystem integrity primitives shared by discovery and checkpoint loading."""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_checkpoint(data: Path, metadata: Path) -> bool:
    try:
        record = json.loads(metadata.read_text("utf-8"))
        return data.stat().st_size == record["bytes"] and sha256(data) == record["sha256"]
    except (OSError, KeyError, ValueError, TypeError):
        return False
