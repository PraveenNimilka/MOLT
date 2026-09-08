# MOLT 0.11.0 Alpha 8

MOLT 0.11.0 Alpha 8 adds a verified shared Windows runtime, a single-command
installer, conflict-safe PATH precedence, persistent CUDA package caching, and
`molt update`, `molt repair`, and `molt uninstall`. Training behavior and the a3
comparison measurements are unchanged.

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
$p="$env:TEMP\molt-install.ps1"; Invoke-WebRequest https://raw.githubusercontent.com/PraveenNimilka/MOLT/v0.11.0-alpha.8/install-global.ps1 -OutFile $p; if ((Get-FileHash $p -Algorithm SHA256).Hash -ne "275591f0bbb05ce164b96387c61b1d1a572cda9479e8ca7a60e399ca7cc74119") { throw "MOLT installer hash mismatch" }; powershell -NoProfile -ExecutionPolicy Bypass -File $p
molt doctor
```

MOLT is source-available under PolyForm Shield 1.0.0 and is intended for
evaluation and controlled workloads. Controlled-clock
comparison, Soup comparison, larger-model endurance, and independent reproduction
remain open before a general superiority claim.
