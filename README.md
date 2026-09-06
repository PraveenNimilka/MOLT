# MOLT AI Infrastructure

**Local model training. Hardware-aware execution. Measurable results.**

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-22c55e.svg)](pyproject.toml)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-4b5563.svg)](docs/releases/0.10.0-alpha.2.md)
[![Tests](https://github.com/PraveenNimilka/MOLT/actions/workflows/ci.yml/badge.svg)](https://github.com/PraveenNimilka/MOLT/actions/workflows/ci.yml)
[![PyPI publishing](https://github.com/PraveenNimilka/MOLT/actions/workflows/publish.yml/badge.svg)](https://github.com/PraveenNimilka/MOLT/actions/workflows/publish.yml)

MOLT is a Windows-first training runtime for developers and researchers working
on consumer NVIDIA laptops and workstations. It brings small-model pretraining,
QLoRA fine-tuning, thermal pacing, checkpoint recovery, and experiment reporting
into one command-line workflow.

**Current release: 0.10.0a2 · Open-source research alpha.** Suitable for evaluation and
controlled experiments. Production use requires workload-specific validation;
MOLT does not currently offer a commercial support SLA or certified reliability.

[Install](#install) · [Quick start](#quick-start) · [Documentation](#documentation) · [Support](SUPPORT.md) · [Contribute](CONTRIBUTING.md)

## Install

### Verified Windows setup

Run in **Windows Command Prompt**, from a directory where you want a new
`MOLT` folder:

```bat
git clone https://github.com/PraveenNimilka/MOLT.git && cd MOLT && powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

**Prerequisites:** Git, Windows 10/11 x64, a supported NVIDIA GPU with a compatible
driver, internet access, and several GB of free disk space. Repository access is
required. Stop existing training before installation or updates.

The installer provisions a project-local `.venv`, obtains uv if needed, and
installs the locked CUDA PyTorch, QLoRA, and Windows Triton dependencies.
It then checks dependency imports, CUDA backward computation, and a small compiled
backward pass against eager results. A failed check stops setup with an error.

No administrator rights are required by the MOLT script. It does not install GPU
drivers, change Defender settings, suspend applications, or adjust power limits.
Review [install.ps1](install.ps1) before running it; the execution-policy override
applies only to that PowerShell invocation.

**Already downloaded the repository?** Open its folder and run:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

Use `-EagerOnly` to omit Triton and compilation checks, or `-Plan` to preview setup
without installing anything. Full prerequisites and troubleshooting are in the
[installation guide](docs/INSTALL.md).

### Install from PyPI

`moltengine` is published on PyPI. Install CUDA PyTorch from its official index
first; otherwise pip can resolve the CPU-only wheel on Windows:

```powershell
py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
py -m pip install "moltengine[qlora,data,windows-fusion]==0.10.0a2"
```

Then verify the active Python environment and CUDA runtime:

```powershell
molt doctor
```

For a reproducible source checkout, install the signed release tag rather than
an unversioned branch:

```powershell
py -m pip install "moltengine[qlora,data,windows-fusion] @ git+https://github.com/PraveenNimilka/MOLT.git@v0.10.0-alpha.2"
```

Maintainer publishing and supply-chain instructions are in
[docs/PUBLISHING.md](docs/PUBLISHING.md).

## Quick start

From the repository folder, launch the guided interface:

```bat
.venv\Scripts\molt.exe
```

Or inspect the environment directly:

```bat
.venv\Scripts\molt.exe --ui inspect
```

The explicit executable path works without activating a virtual environment or
adding MOLT to your global PATH. If uv is available on PATH, you can also use
`uv run molt`.

### Recommended workflow

**Install → doctor → prepare → fit test → train → evaluate → export.**

`molt doctor` identifies exactly which Python environment and source checkout you
are running. `molt config --init` creates a workspace manifest.
`info` and `inspect` remain supported aliases for their older diagnostic views.

Text, JSONL, and Parquet preparation and text generation are available. Use
`molt prepare --help` and `molt generate --help`; use `molt research --help` for
experimental benchmarks. Legacy top-level research commands remain compatible.

### Train with your own data

MOLT does not download model weights or datasets during setup. A common local
fine-tuning flow is now three commands:

```bat
.venv\Scripts\molt.exe prepare data\examples.jsonl --model models\Qwen --output molt-workspace\datasets\examples
.venv\Scripts\molt.exe fit-test --config molt-workspace\datasets\examples\training.json
.venv\Scripts\molt.exe --ui train --config molt-workspace\datasets\examples\training.json
```

Preparation accepts UTF-8 `.txt`, line-delimited JSON objects (`.jsonl`), and
`.parquet`. Common `text`, chat `messages`, and `prompt`/`completion` records are
recognized. JSONL and Parquet are streamed and split at record boundaries; they
are not advertised as supporting every possible third-party schema.

1. Choose a profile from [configs/](configs/) and update its data, model, and
   artifact paths. Example paths are not bundled datasets.
2. Validate the configuration before allocating training resources.
3. Start training, then inspect the resulting run artifacts.

For example, after adapting `configs/molt-stream-production.json`:

```bat
.venv\Scripts\molt.exe train --config configs\molt-stream-production.json --dry-run
.venv\Scripts\molt.exe --ui train --config configs\molt-stream-production.json
```

A dry run validates the specification and referenced data paths. It does not
prove that the model fits in VRAM or that a full run will complete.

Use the guided resume command to select a saved run:

```bat
.venv\Scripts\molt.exe --ui resume
```

See the [CLI reference](docs/cli.md) for profiles, run reports, and automation.

### Interchange formats

| Stage | MOLT format | Compatibility boundary |
| --- | --- | --- |
| Input | `.txt`, `.jsonl`, `.parquet` | Common text/chat/prompt-completion schemas; custom columns are configurable. |
| Training data | little-endian int32 memory-mapped `.bin` | MOLT's documented flat-token format; not claimed to be Megatron indexed-dataset format. |
| QLoRA export | PEFT `adapter_model.safetensors` | Hugging Face Transformers/PEFT and compatible serving stacks; base weights remain required. |
| Consumer export | LoRA `.gguf` through an official llama.cpp checkout | llama.cpp-compatible architectures; requires a compatible GGUF base model. |
| MOLT recovery | verified atomic `.pt` bundle | Exact MOLT resume state, including optimizer/RNG state; not a serving format. |

```bat
rem Auto selects PEFT safetensors for QLoRA and a MOLT bundle for scratch training
.venv\Scripts\molt.exe export --run RUN_DIRECTORY --output-dir exported-adapter

rem Optional consumer adapter conversion; llama.cpp is deliberately not bundled
.venv\Scripts\molt.exe export --run RUN_DIRECTORY --format gguf --llama-cpp C:\src\llama.cpp --output-dir exported-gguf
```

## What MOLT provides

| Capability | Purpose |
| --- | --- |
| Small-model pretraining | Train supported causal language models from scratch. |
| Resident NF4 QLoRA | Adapt supported pretrained models using quantized base weights and LoRA. |
| Memory-mapped datasets | Read token batches without eagerly loading the complete dataset. |
| Thermal pacing | Adjust compute duty cycle using sampled GPU temperature and configured limits. |
| Hardware telemetry | Record GPU board power, energy, temperature, and memory where supported. |
| Atomic checkpoints | Stage and hash checkpoint files before publication; validate saved state on load. |
| Experiment reporting | Preserve configuration and measured outcomes for workload comparisons. |
| Guided and scriptable CLI | Use interactive workflows or structured JSON output. |

Experimental layer streaming and optimization components remain research paths.
They should not be interpreted as universal support for streaming arbitrary
Hugging Face models or as validated improvements over tuned baselines.

### Verified single-machine endurance result

One preregistered engineering run used Windows 11 and an RTX 4060 Laptop GPU to
train all 28 Qwen2.5-1.5B decoder layers with 9,232,384 active LoRA parameters,
context 512, batch 1, accumulation 4, and fused FP32 AdamW. Graphics clocks were
temporarily constrained to 1,500-1,650 MHz and restored afterward.

| Metric | Measured result |
| --- | ---: |
| Duration / tokens | 3,725.7 s / 4,915,200 |
| End-to-end throughput | 1,319.3 tokens/s |
| Training-loop / compute throughput | 1,320.6 / 1,399.5 tokens/s |
| Peak GPU temperature | 71 C |
| Early-to-late rate ratio | 97.3% |
| Board energy | 0.03625 J/token |
| Cooling recoveries / discarded tokens | 0 / 0 |
| Held-out NLL | 1.6635 to 1.1045 |
| PyTorch allocation / total NVML use | 2.956 / 3.75 GiB |

This is a **single-machine, single-seed result**, not an official comparison
with Unsloth, a universal throughput claim, or proof of a novel algorithm.
Independent reproduction and the registered multi-seed AB/BA comparison remain
open evidence gates.

### Safe endurance profile

On supported NVIDIA Windows systems, an Administrator can explicitly authorize
MOLT's measured endurance clock range. MOLT restores automatic clocks in a
`finally` block on completion, error, Ctrl+C, or thermal stop:

```powershell
.venv\Scripts\molt.exe optimize-gpu --profile endurance --config YOUR_CONFIG.json
```

The command never changes clocks without interactive confirmation (or an
explicit `-y` for automation), refuses non-elevated execution, and verifies the
clock range recorded by run telemetry. It does not change firmware, fan curves,
Defender, other applications, or unsupported laptop power limits.

See the [all-layer thermal frontier](docs/research/all-layer-memory-thermal-frontier-2026-09-06.md) and [negative results register](docs/negative-results.md) for full methodology, limitations, and rejected experiments.

## Readiness, performance, and safety

MOLT distinguishes **installed dependencies**, **successful runtime checks**, and
**validated training workloads**. They are not interchangeable.

- **Model fit is workload-dependent.** An 8B dependency-readiness flag does not
  guarantee an 8B model fits in 8 GB of VRAM.
- **Memory mapping is not zero-RAM.** Mapped pages, OS caches, batches, model
  weights, and optimizer state still consume memory.
- **Pacing is software control, not hardware protection.** It cannot guarantee
  flat temperatures or prevent every throttle, OOM, or power shutdown.
- **Checkpoints reduce recovery risk, not all data loss.** Unsaved steps can be
  lost; checksums and atomic publication do not guarantee survival of every
  filesystem or hardware failure.
- **Optional tools may remain unavailable.** Liger and native MSVC/nvcc development
  tooling are not required for every training path and are not included in the
  default installer.
- **Prioritized mode affects only MOLT.** Its temporary process priority is
  restored after normal completion or a training exception. Batch geometry,
  thermal settings, other applications, and Defender are not silently changed.

Historical measurements and methodology are retained in the
[benchmarking guide](docs/benchmarking.md). Results with different validation
quality are not proof of an equal-quality speed or energy advantage. This release
does not claim a universal throughput target or a new training-algorithm breakthrough.

See the [release verification notes](docs/releases/0.10.0-alpha.2.md) for automated tests
and physical runtime checks. Fresh-machine bootstrap and sustained workload
behavior still require broader reproduction. The live CI badge represents the
latest GitHub test status.

## Update an existing installation

From your existing checkout, with training stopped:

```bat
git status
git pull --ff-only origin main
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

If Git reports local changes or diverged history, resolve them before updating;
do not force-reset your work. Setup does not delete model weights, datasets, or
run artifacts. Back up important checkpoints before changing environments.

Older MOLT versions could add Defender exclusions. These are not removed
automatically because ownership cannot be inferred safely. Review unwanted
entries manually in Windows Security; see the [0.9.1 release notes](docs/releases/0.9.2.md).

## Documentation

- [Installation and troubleshooting](docs/INSTALL.md)
- [Command-line reference](docs/cli.md)
- [Architecture](docs/architecture.md)
- [Benchmark methodology](docs/benchmarking.md)
- [Development guide](docs/development.md)
- [Release notes](docs/releases/0.10.0-alpha.2.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License and feedback

MOLT is distributed under the [MIT License](LICENSE), which permits commercial
use subject to its terms. Model weights, datasets, and dependencies retain their
own licenses.

MIT is an open-source license: recipients may inspect, build, modify, and
redistribute the published source while preserving the required license notice.
Public source code cannot simultaneously be made technically uncopyable. Keep
future proprietary services or private optimization modules outside this public
repository and obtain qualified legal advice before changing the licensing model.

Report reproducible bugs through [GitHub Issues](https://github.com/PraveenNimilka/MOLT/issues).
Include the commit, Python/PyTorch versions, relevant configuration, and error
traceback. Remove credentials, private data, and sensitive paths before sharing.
