# Installing MOLT on Windows

## Recommended per-user installation

Open PowerShell and run one command:

```powershell
$p="$env:TEMP\molt-install.ps1"; Invoke-WebRequest https://raw.githubusercontent.com/PraveenNimilka/MOLT/v0.12.0/install-global.ps1 -OutFile $p; if ((Get-FileHash $p -Algorithm SHA256).Hash -ne "75f7de635562447f4246a78634db56b4d11b4664b4c868036d3da1a8ffdeb7f5") { throw "MOLT installer hash mismatch" }; powershell -NoProfile -ExecutionPolicy Bypass -File $p
```

No repository checkout, pre-existing Python installation, administrator terminal,
pipx, or project virtual environment is required. Review the versioned script at
the URL before running it if your security policy requires manual inspection.

The installer uses these stable per-user paths:

| Purpose | Path |
| --- | --- |
| Runtime | `%LOCALAPPDATA%\MOLT\runtime` |
| Launcher | `%LOCALAPPDATA%\MOLT\bin\molt.cmd` |
| Reusable package cache | `%LOCALAPPDATA%\MOLT\cache\uv` |
| Runtime manager | `%LOCALAPPDATA%\MOLT\installer.ps1` |

It downloads a pinned, SHA-256-verified uv bootstrap, provisions Python 3.12 when
needed, explicitly installs `torch==2.8.0+cu128` from PyTorch's CUDA 12.8 index,
installs the complete QLoRA/data/Windows runtime, and runs dependency, CUDA
backward, and compiled-backward verification. It puts the managed launcher first
in the user PATH and reports lower-priority MOLT launchers rather than silently
deleting unrelated Python installations.

Open a new terminal after installation:

```powershell
molt --version
molt doctor
molt
```

## Runtime management

These commands work from any directory:

```powershell
molt update
molt repair
molt uninstall
```

- `update` queries PyPI for the latest published MOLT version, updates the one
  managed runtime, verifies it, and retains the package cache.
- `repair` reinstalls the current MOLT package and reruns every verification.
- `uninstall` removes the managed runtime, launcher, installer, and cache after
  confirmation. It never removes models, datasets, or run directories.
- Add `--dry-run` to any management command to inspect its resolved paths without
  changing the machine.

The persistent uv cache is shared by every project, so the CUDA PyTorch package
is not downloaded separately for each model or dataset folder.

## Reproducible repository installation

Developers who need the exact lock file can still create a checkout-local
environment:

```powershell
git clone --branch v0.12.0 --depth 1 https://github.com/PraveenNimilka/MOLT.git
Set-Location MOLT
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\.venv\Scripts\molt.exe doctor
```

Use `-EagerOnly` to omit Triton and the compile probe, or `-Plan` to show the
locked synchronization plan without changing files.

## What is not changed

Neither installer modifies GPU drivers, antivirus, fan curves, persistent GPU
clocks, application priority, or power settings. Model weights and datasets are
never downloaded implicitly.

## Troubleshooting

If `molt doctor` reports an unexpected Python executable, inspect launchers:

```powershell
Get-Command molt -All | Select-Object Source
```

The first result after opening a new terminal should be:

```text
%LOCALAPPDATA%\MOLT\bin\molt.cmd
```

Run `molt repair` if CUDA packages or runtime files were modified. A successful
small CUDA/compile probe does not establish that a particular model fits VRAM or
that a sustained workload is thermally safe; guided training performs a real
two-update fit test before the full run.

`winget install MOLT` is not yet a live command. It requires a separately
versioned distributable and acceptance into Microsoft's WinGet package index.
The managed installer is the supported beginner path for 0.12.0.

## Upstream sources

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [PyTorch CUDA installation](https://pytorch.org/get-started/locally/)
- [Triton Windows](https://github.com/triton-lang/triton-windows)
