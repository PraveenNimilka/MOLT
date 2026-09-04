"""A single explicit workspace manifest, with legacy discovery as a fallback."""
import json
from pathlib import Path


def workspace_paths(kind: str) -> list[Path]:
    current = Path.cwd()
    for parent in (current, *current.parents):
        for path in (parent / "molt-workspace.json", parent / "molt-workspace/molt-workspace.json"):
            if path.is_file():
                data = json.loads(path.read_text("utf-8"))
                if data.get("schema_version") != 1:
                    raise ValueError(f"Unsupported workspace manifest: {path}")
                target = Path(data["paths"][kind])
                return [(path.parent / target).resolve() if not target.is_absolute() else target]
    return []
