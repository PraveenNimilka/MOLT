# Installing MOLT

MOLT is designed for **single-command installation** on modern Windows laptops and workstations equipped with NVIDIA GPUs.

---

## 1. Quick Start Installation

Clone the repository and install in editable mode:

```powershell
git clone https://github.com/PraveenNimilka/MOLT.git
cd MOLT
pip install -e .
```

Or build and install the standalone wheel:

```powershell
pip install .
```

---

## 2. System Requirements

* **Operating System:** Windows 11 or Windows 10 (64-bit)
* **Python Version:** `3.12` (Python 3.12.x recommended)
* **NVIDIA GPU:** GeForce RTX 30-series, 40-series, or RTX Ada Generation (4GB+ VRAM; 8GB recommended)
* **NVIDIA Driver:** Driver version 550.00 or higher
* **PyTorch:** PyTorch with CUDA acceleration (`torch>=2.4.0` with CUDA 12.4 or 12.8)

If installing PyTorch separately:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## 3. Verify Your Installation

Run the hardware diagnostic command:

```powershell
molt info
```

You should see your detected GPU, VRAM, and the recommended profile:
```text
╭─ [ MOLT HARDWARE DIAGNOSTICS ] ─────────────────────────────╮
│ NVIDIA GPU          NVIDIA GeForce RTX 4060 Laptop GPU      │
│ VRAM                8.0 GB (7.4 GB free)                    │
│ Suggested Profile   BALANCED                                │
╰─────────────────────────────────────────────────────────────╯
```

Run the 2-second non-destructive smoke benchmark:

```powershell
molt benchmark --smoke
```

If the smoke benchmark reports `[ PASS ]`, your installation is 100% complete and ready to train.

---

## 4. Repository Boundary Notice

MOLT is a completely independent training runtime. **Unsloth is NOT a dependency of MOLT** and is not required for installation, normal operation, or benchmarking.
