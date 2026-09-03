# MOLT Examples & Workspace Templates

This directory contains examples demonstrating how to use the MOLT AI runtime for local training, profiling, and benchmarking.

---

## 1. Directory Structure Conventions

When training locally, MOLT automatically discovers models and datasets in:

```text
molt-workspace/
├── models/       <-- Place Hugging Face / safetensors model directories here
├── datasets/     <-- Place binary token files (.bin) here
├── runs/         <-- Checkpoints and metrics summaries are stored here
└── configs/      <-- Optional custom training configuration files
```

To create this structure automatically in your current directory:
```powershell
molt config --init
```

---

## 2. Python API Example

See `train_custom.py` for a self-contained script demonstrating how to programmatically construct a `TrainingSpec`, apply a high-level policy profile (e.g. `BALANCED`), and train.
