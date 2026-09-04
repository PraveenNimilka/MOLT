"""Installation identity plus capability detection, without training or installs."""
import subprocess
import sys
from pathlib import Path

from molt_stream import __version__


def doctor() -> dict[str, object]:
    from molt_stream.training.stream_benchmark import inspect_capabilities
    result = inspect_capabilities()
    source = Path(__file__).resolve()
    result.update(version=__version__, python_executable=sys.executable,
                  package_path=str(source.parents[1]), environment_prefix=sys.prefix,
                  virtual_environment=sys.prefix != sys.base_prefix,
                  checks_scope="Dependency detection, not model-fit or runtime verification")
    root = source.parents[3]
    result["git_commit"] = None
    if (root / ".git").exists():
        try:
            check = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, timeout=3, check=True)
            result["git_commit"] = check.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    result["next_step"] = "molt config --init; then prepare data and validate your training config"
    if sys.prefix == sys.base_prefix:
        result["environment_warning"] = "Global Python installation; use your checkout's .venv/Scripts/molt.exe"
    return result
