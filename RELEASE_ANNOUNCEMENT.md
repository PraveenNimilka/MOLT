# MOLT 0.11.0 Alpha 9

MOLT 0.11.0 Alpha 9 provides a verified shared Windows runtime, a single-command
installer, conflict-safe PATH precedence, persistent CUDA package caching, and
`molt update`, `molt repair`, and `molt uninstall`. It supersedes Alpha 8 by
making repair and update safe while the Windows launcher is running. Training
behavior and the a3 comparison measurements are unchanged.

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
$p="$env:TEMP\molt-install.ps1"; Invoke-WebRequest https://raw.githubusercontent.com/PraveenNimilka/MOLT/v0.11.0-alpha.9/install-global.ps1 -OutFile $p; if ((Get-FileHash $p -Algorithm SHA256).Hash -ne "9597912ea5443ac8773d9959d3ad1acc0d32513d254c2a796875455f8e8d2798") { throw "MOLT installer hash mismatch" }; powershell -NoProfile -ExecutionPolicy Bypass -File $p
molt doctor
```

MOLT is source-available under PolyForm Shield 1.0.0 and is intended for
evaluation and controlled workloads. Controlled-clock
comparison, Soup comparison, larger-model endurance, and independent reproduction
remain open before a general superiority claim.
