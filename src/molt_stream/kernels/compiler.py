from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import torch
from torch import nn


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def configure_windows_compiler_cache() -> Path:
    """Select a short deterministic path to avoid Win32 MAX_PATH failures."""
    root = Path(".c").resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_HOME", str(root))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    return root


def prepare_execution_model(model: ModuleT, backend: str) -> ModuleT | nn.Module:
    if backend == "eager":
        return model
    configure_windows_compiler_cache()
    modes = {
        "compile-max-autotune": "max-autotune",
        "compile-max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
    }
    return torch.compile(model, mode=modes[backend])
