# MOLT Release Baseline Record

**Created Date:** 2026-09-04  
**Git Branch:** `release/v1-cli-hardening` (branched from `main` at `9d0b346`)  
**Base Commit Hash:** `9d0b34676be3a4ebfa055628df77aa2f8fe5368a`  
**Base Commit Message:** `docs: Recreate professional README with 20M benchmark, dual-gear architecture, and usage guide`  
**Repository Remote:** `https://github.com/PraveenNimilka/MOLT`

---

## 1. System & Environment Baseline

* **Operating System:** Windows 11 Build 26200
* **Python Runtime:** Python `3.12.13` (64-bit)
* **Package Name:** `moltengine`
* **Latest Release:** `0.11.0a5`
* **Package Version:** `0.11.0a5` (from `pyproject.toml`; `main` may include unreleased changes)
* **Console Entry Point:** `molt = "molt_stream.cli:main"`

### Installed Runtime Dependencies `[MEASURED]`
* `torch`: `2.8.0+cu128` (CUDA 12.8)
* `transformers`: `5.16.1`
* `peft`: `0.20.0`
* `bitsandbytes`: `0.50.2`
* `accelerate`: `1.14.0`
* `nvidia-ml-py`: `13.610.43`
* `psutil`: `7.2.2`
* `numpy`: `2.5.2`
* `pytest`: `9.1.1`
* `triton-windows`: `3.4.0.post21`
* `hatchling`: (build backend)

> [!IMPORTANT]
> **Repository Boundary Standard:** Unsloth is **NOT** part of the MOLT repository, runtime dependency graph, `pyproject.toml`, or `requirements.txt`. MOLT is an independent research runtime. Competitor environments remain strictly external.

---

## 2. Existing CLI Commands & Architecture

Current CLI entry points mapped in `molt_stream.cli:build_parser`:
* `molt train --config <path> [--mode-select normal|prioritize] [--resume <path>]`
* `molt resume --run <path> [--galore]`
* `molt evaluate --run <path>`
* `molt report --run <path>`
* `molt compare --left <path> --right <path>`
* `molt stream-benchmark --config <path>`
* `molt throughput --config <path>`
* `molt counterfactual-rate --config <path>`

---

## 3. Core Source Modules

* `src/molt_stream/cli.py`: Command-line interface and terminal cards.
* `src/molt_stream/core/specs.py`: Strongly typed dataclass configurations (`TrainingSpec`, `ModelSpec`, `DataSpec`, `StreamSpec`).
* `src/molt_stream/core/system_tuning.py`: Windows process priority, Defender exclusions, background app suspension, and Dual-Gear intercooler configuration.
* `src/molt_stream/training/engine.py`: High-level training dispatch (`pretrain` vs `qlora`).
* `src/molt_stream/training/qlora.py`: Resident 4-bit NF4 QLoRA execution loop with closed-loop telemetry and pacing.
* `src/molt_stream/streaming/engine.py`: CPU/GPU layered streaming engine.
* `src/molt_stream/data/bytes.py`: `MMapTokenBatcher` for zero-copy $O(1)$ memory-mapped disk streaming.
* `src/molt_stream/experiments/store.py`: `AtomicCheckpointStore` with two-phase commit and SHA-256 integrity verification.
* `src/molt_stream/measurement/telemetry.py`: 10 Hz NVML hardware poller (temperatures, power, clocks, VRAM).
* `src/molt_stream/measurement/thermal.py`: Dual-gear duty cycle calculation and predictive cooling governor.

---

## 4. Test Suite Baseline `[MEASURED]`

* **Test Framework:** `pytest 9.1.1`
* **Test Files:** 16 files under `tests/`
* **Collected Tests:** 76 unit tests
* **Test Execution Result:** **76 passed in 17.81s** (0 failures, 0 warnings, 0 skips)

---

## 5. Performance Smoke-Test Baseline `[MEASURED]`

Executed on RTX 4060 Laptop GPU using `configs/molt-stream-smoke.json` (20 steps, 5,120 tokens):
* **Execution Time:** 1.690 seconds
* **Active Token Rate:** 3,028.8 tokens/second
* **GPU Board Energy:** 18.81 Joules
* **Peak CUDA Memory:** 28.07 MB
* **Monotonic Loss Progression:** 71.15 -> 36.35 -> 13.41 -> 7.48 (Validation NLL: 6.83)

---

## 6. Critical Protected Files (DO NOT MODIFY ALGORITHMS)

The following files contain validated research mathematics and MUST NOT have their mathematical or functional behavior altered:
* `src/molt_stream/measurement/thermal.py`: Dual-gear thermal controller mathematics and duty-cycle calculation.
* `src/molt_stream/training/qlora.py`: QLoRA execution loop, loss computations, and gradient propagation.
* `src/molt_stream/data/bytes.py`: Memory-mapped token batching.
* `src/molt_stream/experiments/store.py`: Atomic two-phase checkpoint commit and SHA-256 verification.

---

## 7. Known Issues & Hardening Goals

1. **CLI Complexity:** Users currently must pass raw JSON files with full paths (`molt train --config configs/molt-stream-61m.json`).
2. **Missing Workspace Discovery:** No automatic detection of local models (`models/`) or datasets (`datasets/`).
3. **Missing Guided Landing Menu:** Running bare `molt` prints argparse help instead of an interactive menu.
4. **Missing High-Level Profiles:** No unified `SPEED`, `BALANCED`, `COOL`, `ENERGY` user-facing profiles.
5. **Missing Standard Subcommands:** No `molt info`, `molt config`, `molt benchmark` commands.
6. **Installation:** Needs single-command reproducible installation (`pip install -e .` or `pip install .`) with clear documentation.
