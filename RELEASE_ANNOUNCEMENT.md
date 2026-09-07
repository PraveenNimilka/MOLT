# MOLT 0.11.0 Alpha 3

MOLT is a Windows-first local LLM fine-tuning runtime for consumer NVIDIA GPUs.
It provides guided data preparation, real workload fit testing, QLoRA training,
hardware telemetry, thermal controls, verified recovery, and adapter export.

The latest internal six-pair diagnostic screens reduced mean elapsed time,
board energy, and allocator peak versus the tested Unsloth configuration across
Qwen, Llama-family, and Gemma-family paths. Results and limitations are published
in the README; proprietary implementation details are intentionally omitted.

Install:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "moltengine[qlora,data,windows-fusion]==0.11.0a3"
molt doctor
```

This alpha is intended for evaluation and controlled workloads. Controlled-clock
comparison, Soup comparison, larger-model endurance, and independent reproduction
remain open before a general superiority claim.
