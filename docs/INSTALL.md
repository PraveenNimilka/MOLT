# Installing MOLT on Windows

## One setup command (from a complete repository checkout)

Stop any training before synchronizing an existing environment. In Command Prompt:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

First-time users can download/extract the GitHub repository, open its folder, and
run that command. With Git already installed, the complete Command Prompt line is:

```bat
git clone https://github.com/PraveenNimilka/MOLT.git && cd MOLT && powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

Private repositories require GitHub access. Existing users should use
`git pull --ff-only origin main` inside their checkout, then run the setup command;
do not clone over an existing directory.

The installer:
- Uses existing uv, or downloads the pinned official Astral uv installer into
  `.tools` and installs uv there without changing your permanent PATH.
- Uses the lockfile and Python 3.12 to synchronize a project-local `.venv`.
  uv can download managed Python if necessary.
- Installs QLoRA dependencies, bounded JSONL/Parquet preparation support, and
  the PyTorch 2.8 / Triton Windows 3.4 pair. Parquet uses locked PyArrow.
- Checks dependency imports, a small CUDA backward pass, and a compiled backward
  pass compared with eager results. Failures return a nonzero exit code.
- Does not install drivers, alter antivirus, suspend apps, or install MSVC/CUDA
  development tools. Expect several GB of downloads and additional disk usage.
  Review the script before running; execution-policy bypass applies to this
  invocation, not permanent policy.

Launch without requiring a global PATH change:

```bat
.venv\Scripts\molt.exe --ui inspect
```

If uv is already on PATH, `uv run molt --ui inspect` also works.

## Eager-only installation

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -EagerOnly
```

This omits Triton and the compile probe. Use `"execution_backend": "eager"`.
It still requires working NVIDIA CUDA for the installation check.
Use `-Plan` to print the commands without downloads or environment changes.

## What the capability flags actually mean

- CUDA: needs a compatible NVIDIA driver and CUDA-enabled PyTorch.
- QLoRA packages: dependency presence alone does not prove an 8B workload fits.
  The setup probe imports these libraries but does not fine-tune a real model.
- Triton / torch.compile: the Windows extra installs the matching Triton version.
  Only the explicit compiled runtime test establishes that the tested graph runs.
- Liger: optional kernels, **not included in this Windows setup**. Installing it
  just to turn a flag green may introduce unsupported combinations; MOLT's full
  Liger integration has not been validated by this installer.
- MSVC, nvcc, Ninja, CUDA_HOME: optional native-extension development tooling.
  These may remain absent while eager/compiled training works. To develop native
  extensions, install Visual Studio Build Tools with Desktop development with C++,
  a compatible NVIDIA CUDA Toolkit, and Ninja; use a Developer terminal so
  `cl` is discoverable. A detected toolchain is not a successful extension build.

## Limitations and troubleshooting

An NVIDIA driver must already be installed. For CUDA initialization errors,
check `nvidia-smi` and the official driver/toolkit compatibility documentation.
Do not infer a minimum driver version from MOLT's old installation guide.

Compilation may take minutes on first use. A passing small probe does not establish
model fit, full optimizer correctness, sustained performance, or thermal safety.
No datasets or model weights are downloaded. Start with a small real workload.
The setup script stops on failure; it never labels a failed compile check successful.
Logs print the failing command's diagnostics. Existing run artifacts are not removed.

## Sources

Implementation verification: 100 local tests passed, including mocked installer
success/failure paths and non-mutating plan mode. The real CUDA eager and compiled
backward probes passed on the development RTX 4060 / PyTorch 2.8 environment.
The initial uv bootstrap/download path has not been end-to-end tested on a fresh
Windows installation; this remains an external reproduction requirement.

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
  and [installer options](https://docs.astral.sh/uv/reference/installer/).
- [uv locked synchronization](https://docs.astral.sh/uv/concepts/projects/sync/).
- [Triton Windows compatibility and setup](https://github.com/triton-lang/triton-windows).
- [Liger upstream requirements](https://github.com/linkedin/Liger-Kernel).
