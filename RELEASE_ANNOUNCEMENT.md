# MOLT 0.11.0 Alpha 7

MOLT 0.11.0 Alpha 7 adds a complete pinned beginner workflow, a measured
75-second demonstration with public logs, and one consolidated diagnostic
reproduction page. Training behavior and the a3 comparison measurements are
unchanged.

MOLT is a Windows-first local LLM fine-tuning runtime for consumer NVIDIA GPUs.
It provides guided data preparation, real workload fit testing, QLoRA training,
hardware telemetry, thermal controls, verified recovery, and adapter export.

The latest internal six-pair diagnostic screens reduced mean elapsed time,
board energy, and allocator peak versus the tested Unsloth configuration across
Qwen, Llama-family, and Gemma-family paths. Results and limitations are published
in the README. Internal optimization rationale and development profiling records
are not part of the documented API.

Install:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install moltengine==0.11.0a7
molt setup
molt doctor
```

MOLT is source-available under PolyForm Shield 1.0.0 and is intended for
evaluation and controlled workloads. Controlled-clock
comparison, Soup comparison, larger-model endurance, and independent reproduction
remain open before a general superiority claim.
