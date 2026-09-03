# MOLT Command-Line Interface (CLI) Reference

MOLT provides both a **guided interactive menu** and **direct command-line flags** for advanced scripting.

---

## 1. Guided Interactive Mode

Simply type:

```powershell
molt
```

MOLT inspects your hardware and displays an interactive landing menu:
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

## 2. Command Reference

### `molt train`
Start an AI model training run.

```powershell
# Interactive training (prompts for model, dataset, profile)
molt train

# Fast training with discovered assets and policy profile
molt train --model QWEN --dataset datasets/train.bin --profile balanced

# Training from an explicit JSON configuration
molt train --config configs/molt-stream-production.json

# Pre-flight validation without starting CUDA compute
molt train --config configs/molt-stream-smoke.json --dry-run

# Non-interactive automated execution
molt train --config configs/molt-stream-smoke.json -y
```

**Options:**
* `--config <path>`: Path to JSON configuration file.
* `--model <path>`: Path or name of base model directory.
* `--dataset <path>`: Path to binary token file (`.bin`).
* `--profile {speed,balanced,cool,energy}`: High-level thermal policy profile.
* `--steps <int>`: Override max optimization steps.
* `--batch-size <int>`: Micro-batch size.
* `--dry-run`: Validate spec and memory estimation without starting compute.
* `-y, --yes`: Skip confirmation prompt.
* `--debug`: Output complete Python tracebacks on error.

---

### `molt resume`
Resume an interrupted run from its cryptographic checkpoint.

```powershell
# Interactive resume (lists recent runs and verification status)
molt resume

# Resume a specific run directory
molt resume --run runs/20260903-113243-qlora-493938ec
```

---

### `molt info`
Display real-time hardware diagnostics and suggested training configurations.

```powershell
molt info
```

---

### `molt benchmark`
Measure sustained token throughput and verify training correctness.

```powershell
# Fast 2-second non-destructive smoke benchmark
molt benchmark --smoke

# Sustained benchmark with custom config
molt benchmark --config configs/molt-stream-benchmark.json --steps 50
```

---

### `molt config`
Workspace asset discovery and initialization.

```powershell
# Initialize standard molt-workspace/ directory
molt config --init

# List detected models, datasets, and previous runs
molt config --list
```

---

## 3. High-Level Training Profiles

| Profile | Thermal Ceiling | Cooling Cadence | Intended Use Case |
| :--- | :---: | :---: | :--- |
| **`SPEED`** | `82.0°C` | `50ms` | Maximum throughput (~1,950 tok/s); desktop cards with high airflow. |
| **`BALANCED`** | `74.0°C` | `220ms` | High throughput (~1,380 tok/s) with Dual-Gear cooling; gaming laptops. |
| **`COOL`** | `68.0°C` | `300ms` | Conservative thermal limit for warm rooms or quiet fans. |
| **`ENERGY`** | `70.0°C` | `250ms` | Duty cycle optimized for energy efficiency per token. |
