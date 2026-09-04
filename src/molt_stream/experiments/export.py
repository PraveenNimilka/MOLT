"""Publish a verified run bundle without overwriting existing user files."""
import json
import os
from pathlib import Path
import shutil
import tempfile

from molt_stream.experiments.store import AtomicCheckpointStore


def export_run(run: str, output_dir: str) -> dict[str, str]:
    source, target = Path(run).resolve(), Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Export destination already exists: {target}")
    checkpoint = AtomicCheckpointStore(source).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="molt-export-", dir=target.parent) as temporary:
        package = Path(temporary) / "run"
        package.mkdir()
        shutil.copy2(checkpoint, package / "checkpoint.pt")
        shutil.copy2(checkpoint.with_suffix(".complete.json"), package / "checkpoint.complete.json")
        shutil.copy2(source / "spec.resolved.json", package / "spec.resolved.json")
        if (source / "metrics.summary.json").exists():
            shutil.copy2(source / "metrics.summary.json", package / "metrics.summary.json")
        (package / "export.json").write_text(json.dumps({
            "schema_version": 1, "format": "MOLT checkpoint bundle",
            "base_weights_included": False, "datasets_included": False,
            "portability": "Update recorded local paths on the destination machine; not a standalone model",
        }, indent=2), encoding="utf-8")
        os.rename(package, target)
    return {"directory": str(target), "format": "MOLT checkpoint bundle"}
