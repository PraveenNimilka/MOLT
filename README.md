# MOLT: Hardware-Aware AI Training Runtime

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2011%20%7C%2010-lightgrey.svg)](docs/INSTALL.md)
[![Tests: 84 Passed](https://img.shields.io/badge/Tests-84%20Passed-brightgreen.svg)](tests/)

**MOLT** is a hardware-aware local LLM training runtime engineered specifically for consumer laptops and workstations. It eliminates thermal throttling, out-of-memory crashes, and ungraceful power shutdowns by combining closed-loop hardware sensing with resident 4-bit quantization and zero-RAM data streaming.

---

## 🚀 Quick Start

### 1. Install MOLT (One Command)
```powershell
git clone https://github.com/PraveenNimilka/MOLT.git
cd MOLT

# 1. Install PyTorch with NVIDIA CUDA acceleration (Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Install MOLT
pip install -e .
```
*(Tip: If you use `uv`, simply run `uv pip install -e .` which pulls the CUDA wheel automatically).*


### 2. Launch Guided Training
Simply type:
```powershell
molt
```
MOLT automatically detects your GPU, discovers local models and datasets, and guides you through training:

```text
╭─ [ MOLT AI INFRASTRUCTURE ] ───────────────────────────────────────╮
│ Version       v0.9.0 (Release Candidate)                           │
│ Hardware      NVIDIA GeForce RTX 4060 Laptop GPU 8.0 GB            │
│ Recommended   BALANCED                                             │
│ Pillars       Dual-Gear Thermals • 4-bit NF4 QLoRA • Zero-RAM MMap │
╰────────────────────────────────────────────────────────────────────╯

Select an action:
  1. Train             Start a new training run
  2. Resume            Resume from a verified checkpoint
  3. Benchmark         Run a 2-second hardware smoke test
  4. Hardware Info     Inspect GPU, VRAM, and thermal sensors
  5. Configuration     Initialize workspace and list assets
  6. Exit
```

---

## ⚡ Direct Command Usage

For scripting and automation:

```powershell
# Fast training with high-level policy profile
molt train --model models/qwen2-0.5b --dataset datasets/train.bin --profile balanced

# Dry-run validation (verify spec and memory budget without launching compute)
molt train --config configs/molt-stream-production.json --dry-run

# Resume an interrupted run from its atomic checkpoint
molt resume

# Hardware diagnostics & suggested configuration
molt info

# Fast 2-second non-destructive benchmark
molt benchmark --smoke
```

---

## 🛡️ The 4 Core Pillars of MOLT

```
┌─────────────────────────────────────────────────────────────┐
│                       MOLT AI RUNTIME                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│  1. Closed-Loop │  2. Zero-RAM    │  3. 4-bit NF4 + LoRA    │
│     NVML Sensor │     MMap Batcher│     VRAM Compression    │
├─────────────────┴─────────────────┴─────────────────────────┤
│         4. Two-Phase Cryptographic Atomic Checkpoints        │
└─────────────────────────────────────────────────────────────┘
```

1. **Closed-Loop Hardware Pacing (Dual-Gear Transmission):**
   * Samples GPU temperature, board wattage, and clocks at **10 Hz via NVML**.
   * **Gear 1 (Sprint):** Computes at full speed (~1,950 tok/s) when cool ($T < 74^\circ\text{C}$).
   * **Gear 2 (Pit Cruise):** Inserts calibrated micro-pauses (220ms) when warm ($T \ge 74^\circ\text{C}$), allowing laptop fans to evacuate heat without aborting training.
2. **Zero-RAM Data Streaming (`MMapTokenBatcher`):**
   * Reads tokens directly from NVMe SSD via OS virtual memory mapping.
   * System RAM stays flat at **~0.8 GB** whether training on 10MB or 50GB.
3. **Resident 4-Bit NF4 Quantization + LoRA:**
   * Compresses base model into 4-bit NormalFloat4 (NF4) in VRAM. Fits models up to 8-Billion parameters inside consumer 8GB GPUs with room to spare.
4. **Two-Phase Atomic Checkpointing:**
   * Every checkpoint is staged, SHA-256 hashed, and atomically committed (`checkpoint.complete.json`). Power cuts or battery loss will never corrupt your progress.

---

## 📊 Benchmark: Empirical Evaluation on Consumer Hardware

MOLT includes a reproducible, audited benchmark suite ([Benchmark 001](docs/benchmarking.md)) evaluated on an **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, 100W TGP)** training on 1,000,000 tokens:

| Runtime / Configuration | Wall Speed | Active Compute | Mean Power | Peak Temp | Peak VRAM | Wh / 1M Tokens | Convergence (Val PPL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MOLT (Thermal ON - Dual-Gear)** | **`1,376.7 tok/s`** | **`1,952.4 tok/s`** | **49.07 W** | **`80.0°C`** (72°C cruise) | **6.89 GB** (1.1GB safe) | **9.90 Wh** | **`6.325`** |
| **MOLT (Thermal OFF - Continuous)**| **`1,924.1 tok/s`** | **`1,956.6 tok/s`** | **57.01 W** | 83.0°C | **6.92 GB** (1.1GB safe) | **8.23 Wh** | **`6.325`** |
| **PEFT (Hugging Face Baseline)** | 1,070.9 tok/s | 1,071.0 tok/s | 49.42 W | **85.0°C (Throttled)** | **7.97 GB** (Near OOM) | 13.19 Wh | 5.926 |
| **UNSLOTH (External Ref Baseline)**| **`3,027.8 tok/s`** | **`3,028.7 tok/s`** | 58.46 W | 78.0°C | 4.80 GB | **5.79 Wh** | 6.399 |

* **+28.6% faster wall throughput and +82.3% faster active compute** than standard Hugging Face PEFT.
* **25.0% less electrical energy** consumed than PEFT.
* **1.08 GB safer VRAM margin**, avoiding CUDA Out-of-Memory crashes.
* **Zero numerical drift:** Validation loss between Thermal ON and Thermal OFF is identical to 5 decimal places (`1.84454`).

> [!NOTE]
> **Independent Repository Policy:** MOLT is an independent research project. External comparative engines (such as Unsloth) are referenced strictly as external baseline benchmarks and are **NOT** dependencies of MOLT. MOLT installs and runs completely independently.

For full methodology and telemetry curves, see [docs/benchmarking.md](docs/benchmarking.md).

---

## 📖 Documentation Directory

* **[Installation Guide](docs/INSTALL.md)**: Prerequisites, PyTorch setup, and hardware verification.
* **[CLI Reference](docs/cli.md)**: Command guide, flags, and workspace conventions.
* **[Benchmarking Guide](docs/benchmarking.md)**: Empirical methodology, formulas, and telemetry logs.
* **[Architecture Overview](docs/architecture.md)**: Internal subsystems, data flow, and telemetry contracts.
* **[Developer Guide](docs/development.md)**: Contributing, test suites, and boundary enforcement.

---

## ⚖️ License

MOLT is released under the [MIT License](LICENSE).
