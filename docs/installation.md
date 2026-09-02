# Installation and runtime selection

MOLT never treats a CPU PyTorch installation as CUDA-capable. Always run
`molt --json inspect` after installation and check `cuda_available` before using
a CUDA or QLoRA profile.

## Recommended source checkout on Windows

The repository pins PyTorch 2.8 and declares the official CUDA 12.8 PyTorch
index for `uv`:

```powershell
uv sync --frozen
uv run molt --json inspect
```

For the QLoRA engine:

```powershell
uv sync --frozen --extra qlora
$env:MOLT_QWEN_MODEL = "D:\path\to\your\local\Qwen2-checkpoint"
uv run molt --json inspect
```

`inspect` must show `cuda_available: true` and the QLoRA dependencies as
available before a QLoRA run. Missing capability is an error; MOLT does not
silently fall back to full-precision or CPU training.

## Installing a built wheel

Python wheel metadata cannot carry a custom package-index selection. A generic
`pip install molt_ai_infrastructure-*.whl` may therefore resolve the CPU PyTorch
wheel. For NVIDIA CUDA, install the appropriate official PyTorch wheel for the
host first, then install MOLT and verify with `inspect`. Follow PyTorch's current
official selector rather than copying an old CUDA command from this document:
<https://pytorch.org/get-started/locally/>.

The local clean-install audit intentionally verified both states:

- the MOLT 0.2.0 wheel and its declared dependencies installed into an empty
  Python 3.12 environment;
- the packaged CLI and `inspect` command ran;
- that generic environment reported `torch 2.8.0+cpu` and
  `cuda_available: false`, correctly refusing to present itself as GPU-ready.

## Dataset paths

Portable configs use environment variables instead of private absolute paths:

```powershell
$env:MOLT_DATA_ROOT = "D:\datasets\molt"
$env:MOLT_QWEN_MODEL = "D:\models\Qwen2-0.5B"
```

Unset variables produce an explicit configuration error. Resolved absolute
paths are preserved in each run's `spec.resolved.json` for reproducibility.
