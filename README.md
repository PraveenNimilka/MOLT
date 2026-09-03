# MOLT: Hardware-Aware Closed-Loop LLM Fine-Tuning Runtime

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CUDA 12.4+](https://img.shields.io/badge/cuda-12.4%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: Research Alpha](https://img.shields.io/badge/status-0.2.0--alpha-orange.svg)](docs/vision-and-scope.md)
[![Hardware: NVIDIA RTX 4060 Mobile](https://img.shields.io/badge/tested%20on-RTX%204060%20Laptop%208GB-purple.svg)](docs/audits/hardware-capability-2026-09-01.md)

> **"MOLT is a hardware-aware LLM fine-tuning runtime that uses closed-loop GPU telemetry, thermal-aware workload pacing, memory-mapped data streaming, and fault-tolerant checkpointing to sustain long-running QLoRA workloads on thermally constrained consumer GPUs."**

Traditional machine-learning frameworks (PyTorch native, Hugging Face, DeepSpeed) assume enterprise data-center conditions: unrestricted power delivery and massive liquid-cooling radiators. When executed on consumer hardware—especially thin laptops with shared CPU/GPU copper heatpipes—unconstrained compute saturates the cooling assembly within minutes, driving junction temperatures to thermal limits and triggering motherboard emergency power cutoffs (`PROCHOT`).

**MOLT transforms LLM adaptation from an open-loop brute-force process into a feedback-controlled, thermally aware workload.** By dynamically pacing training cadence based on real-time NVML silicon telemetry, MOLT maintains thermal equilibrium, prevents hardware emergency shutdowns, and sustains high-throughput training across tens of millions of tokens on a standard consumer laptop.

---

## 📊 Current Progress & Audited Benchmark (20 Million Tokens)

In our latest empirical milestone, MOLT completed a full **20,000,000-token continuous QLoRA adaptation** of a 494M-parameter foundation model on an audited consumer gaming laptop without human intervention, thermal shutdowns, or hardware failures.

### Audited Benchmark Summary (`TEST 107` Milestone)

| Metric | Measured Value | Analysis & Context |
| :--- | :--- | :--- |
| **Base Model** | **Qwen2-0.5B-Instruct** | 494,032,768 parameters (24 layers, 14 attention heads, 896 width) |
| **Adaptation Method** | **4-Bit NF4 QLoRA** | 4,399,104 trainable LoRA parameters (~0.89% of base; ~99.11% frozen) |
| **Cumulative Tokens Trained** | **`19,999,744 tokens`** | 19,531 continuous gradient update steps ($batch=4, seq=256$) |
| **Total Wall-Clock Time** | **`3h 12m 32.9s`** (11,552.9s) | End-to-end elapsed runtime from resume checkpoint to completion |
| **Active Compute Time** | **`2h 09m 11.9s`** (7,751.9s) | Pure GPU forward, backward, and optimizer execution time |
| **Thermal Pacing Time** | **`1h 02m 13.1s`** (3,733.2s) | Duty-cycled cooling intervals distributed across 19,531 steps |
| **Overall Session Speed** | **`1,686.5 tokens/sec`** | End-to-end wall-clock throughput (including all cooling intervals) |
| **Pure Compute Speed** | **`2,513.4 tokens/sec`** | Instantaneous throughput during active Tensor Core updates |
| **Peak PyTorch CUDA Allocation**| **`5.73 GB`** | Peak memory allocated by PyTorch for activations and gradients |
| **Peak Total GPU Memory** | **`6.66 GB`** | Total GPU memory reserved including driver allocator cache |
| **Post-Step Resident Memory** | **`~0.82 GB`** (820 MB) | Memory occupied by resident model parameters between steps |
| **Total Board Energy** | **`543,334.9 Joules`** | **0.1509 kWh** (measured via NVML board power integration) |
| **Mean GPU Board Power** | **`47.04 Watts`** | Reduced from unconstrained ~75W, matching laptop thermal equilibrium |
| **Electricity Cost** | **`~$0.022` (~2.2 Cents)** | Calculated at standard commercial rate ($0.15 / kWh) |
| **Thermal Profile** | **`68°C – 73°C` sustained** | Peak transient: 84.0°C (safely below 85.0°C abort wall); finish: 62.0°C |
| **Loss & Convergence** | **`4.35` ➔ `0.0104`** | Validation NLL: 0.03078 (Validation Perplexity: **`1.03126`**) |
| **Hardware Reliability** | **`0 Shutdowns / 0 Aborts`** | Continuous operation with zero data loss or process crashes |

---

## 🏛️ Core Technical Architecture

MOLT is composed of four integrated, evidence-gated subsystems:

```text
  ┌─────────────────────────────────────────────────────────────────┐
  │                    MOLT-STREAM RUNTIME                          │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Data Subsystem  │    │ Compute Subsystem│    │  Thermal Cadence │
│  (Zero-RAM mmap) │    │(NF4 QLoRA Base)  │    │ (NVML Feedback)  │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         ▼                       ▼                       ▼
 Flat binary disk        Frozen 4-bit weights     ~9 Hz NVML telemetry
 streams mapped via      + 4.4M LoRA adapters     samples temperature,
 OS virtual memory.      updated via AdamW.       power & clock events.
 Small working set       5.73 GB peak CUDA        Dynamically shifts
 independent of corpus   buffer footprint.        between Gear 1 & 2.
 size.
```

### 1. Closed-Loop Thermal Cadence (Dual-Gear Controller)
Rather than executing compute unconstrained until thermal throttling trips downclocking, MOLT uses a **closed-loop feedback controller** (`DualGearThermalController`) operating over NVML telemetry sampled at approximately 9 Hz (~111ms intervals):
* **Gear 1 (Full Throttle Sprint):** When $T_{\text{GPU}} < 74.0^\circ\text{C}$, steps execute with `0.0ms` pause, yielding ~2,500–3,000 tok/s compute speed.
* **Gear 2 (Thermal Pacing):** When $T_{\text{GPU}} \ge 74.0^\circ\text{C}$, the runtime downshifts into a 220ms duty-cycle interval between steps. This reduces mean GPU power dissipation to ~47W, allowing cooling fans to evacuate heat in steady state without pausing or interrupting training.
* **Automatic Shift-Up:** Returns to Gear 1 when the silicon cools to $\le 65.0^\circ\text{C}$.
* **Emergency Boundary:** A hard software safety limit at $85.0^\circ\text{C}$ guarantees immediate atomic checkpointing before any hardware emergency cutoff could occur.

### 2. Memory-Mapped Dataset Streaming (`mmap`)
Memory-mapped streaming keeps application-level dataset memory approximately independent of total dataset size by accessing token data on demand from flat binary `.bin` files via `torch.from_file` rather than materializing token arrays in Python memory.

### 3. Clean-Room 4-Bit NF4 Quantization + LoRA
Base foundation model weights are packed into 4-bit NormalFloat4 blocks with block-level scale factors. Low-rank decomposition adapters ($r=8, \alpha=16$) are attached across all linear projections (`q, k, v, o, gate, up, down`). Gradients are calculated strictly for the 4.4M adapter parameters, keeping ~99.11% of the base model frozen.

### 4. Fault-Tolerant Atomic State & Checkpoint Commit
Checkpoints are committed using a staged two-phase pattern:
1. State is serialized to a temporary file (`checkpoint.pt.tmp`).
2. A cryptographic **SHA-256 hash** is computed and written to `checkpoint.complete.json`.
3. The file is atomically committed to `checkpoint.pt`, with `checkpoint.previous.pt` retained as a validated fallback.
4. Resuming restores optimizer momentum/variance tensors, the exact token stream cursor position, and the RNG seed state.

---

## 💻 System Requirements

* **Operating System:** Windows 10/11 (x86_64) or Linux (Ubuntu 22.04+).
* **GPU:** NVIDIA GPU with CUDA Compute Capability $\ge 7.5$ (Tested on RTX 4060 Laptop 8GB, RTX 3060/4070/4080/4090).
* **VRAM:** Minimum 6 GB VRAM for 0.5B models; 8 GB recommended.
* **Python:** Version `3.11` or higher.
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv) (strongly recommended) or standard `pip`.

---

## 🛠️ Installation Guide

### 1. Clone the Repository
```powershell
git clone https://github.com/PraveenNimilka/MOLT.git
cd MOLT
```

### 2. Environment Setup (Recommended: `uv`)
Using `uv` ensures fast, deterministic dependency resolution:

```powershell
# Install dependencies into a dedicated virtual environment
uv sync

# Install MOLT in editable research mode
uv pip install -e .
```

*(Alternatively, with standard Python:)*
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 3. Verify Hardware & CUDA Runtime
Run the hardware inspection utility to verify that NVML telemetry and CUDA acceleration are ready:

```powershell
molt inspect
```
*Expected output:* A verified terminal card listing your GPU model, VRAM capacity, driver version, and NVML telemetry status.

---

## 🚀 Quickstart & Usage

### Step 1: Prepare & Tokenize Dataset
Tokenize your corpus into memory-mapped binary chunks:

```powershell
molt prepare --config configs/qwen2_specialist.json
```

### Step 2: Launch Training (Interactive Terminal UI)
Launch training with MOLT's real-time terminal dashboard. You will be prompted to select an execution profile:

```powershell
molt --ui train --config configs/qwen2_specialist.json
```

```text
╭─ [ Execution Profile ] ──────────────────────────────────────────────────╮
│                                                                          │
│ 1. Normal Mode        Standard thermal limits & background defaults      │
│ 2. Prioritized Mode   High CPU Priority, Dual-Gear pacing, Defender hook │
│                                                                          │
╰──────────────────────────────────────────────────────────────────────────╯
Select mode [1=Normal, 2=Prioritized] (default: 1): 2
```

### Step 3: Non-Interactive / Headless Training
For scripts, remote servers, or CI pipelines, specify the execution mode via CLI:

```powershell
molt train --config configs/qwen2_specialist.json --mode-select prioritize
```

### Step 4: Resume an Interrupted Run
MOLT automatically detects previous checkpoints, verifies SHA-256 integrity, and restores exact step, optimizer, and token cursor positions:

```powershell
molt --ui resume --run "runs/20m_specialist/20260903-032451-qlora-96c3b0e7"
```

### Step 5: Evaluate & Benchmark Runs
Generate comprehensive loss, perplexity, and telemetry summary reports:

```powershell
# Evaluate validation perplexity
molt evaluate --run "runs/20m_specialist/20260903-032451-qlora-96c3b0e7"

# Print comprehensive execution report
molt report --run "runs/20m_specialist/20260903-032451-qlora-96c3b0e7"
```

---

## ⚙️ Configuration Reference

Training configurations are defined in clean JSON manifests:

```json
{
  "mode": "qlora",
  "base_model": "path/to/Qwen2-0.5B-Instruct",
  "artifacts_dir": "runs/20m_specialist",
  "batch_size": 4,
  "gradient_accumulation": 1,
  "learning_rate": 0.00015,
  "max_steps": 19531,
  "evaluation_interval": 500,
  "model": {
    "layers": 24,
    "width": 896,
    "hidden_width": 4864,
    "heads": 14,
    "context_length": 256,
    "vocab_size": 151936
  },
  "data": {
    "path": "data/train_20m.bin",
    "validation_path": "data/val_20m.bin",
    "context_length": 256,
    "storage_dtype": "int32",
    "packing": "contiguous",
    "sequential": true
  },
  "stream": {
    "compute_dtype": "bfloat16",
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_target_modules": "all-linear",
    "quant_block_size": 64
  },
  "thermal_control_mode": "dual-gear",
  "thermal_target_c": 74.0,
  "thermal_cruise_max_c": 65.0,
  "thermal_pause_seconds": 0.22,
  "thermal_abort_c": 85.0
}
```

---

## 📖 Research Documentation & Evidence Standard

MOLT follows an explicit **evidence-gated research standard**. Every performance claim must be accompanied by raw machine-readable JSON metrics, environment audits, and reproducibility instructions.

* [Hardware Capability Audit](docs/audits/hardware-capability-2026-09-01.md)
* [Repository Baseline Audit](docs/audits/repository-audit-2026-09-01.md)
* [MOLT-Stream Core Architecture Record](docs/research/molt-stream-core.md)
* [Hypotheses & Falsification Protocols](docs/research/hypotheses.md)
* [Prior-Art Comparison Matrix](docs/research/prior-art-matrix.md)
* [Commercial Readiness Scorecard](docs/commercial-readiness-scorecard.md)

### Evidence Vocabulary Standard:
* **Fact:** Directly measured on local audited hardware or cited from verified primary sources.
* **Reported:** A metric claimed by external prior art but not independently reproduced here.
* **Interpretation:** A reasoned engineering conclusion derived from facts.
* **Hypothesis:** An unverified engineering expectation requiring a controlled test.

---

## 📄 License & Citation

MOLT is released as open-source research software under the Apache 2.0 License.

```bibtex
@software{molt2026,
  author = {Praveen Nimilka and Contributors},
  title = {MOLT: Hardware-Aware Closed-Loop LLM Fine-Tuning Runtime},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PraveenNimilka/MOLT}}
}
```
