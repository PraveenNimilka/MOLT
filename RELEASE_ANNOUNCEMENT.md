# MOLT 0.12.0

MOLT 0.12.0 advances the Windows-first local QLoRA runtime with exact
frozen-vocabulary loss execution, scheduled NF4 backward work, CUDA-graph
memory controls, low-overhead update attribution, bounded gated-MLP
replay/offload experiments, and more precise thermal pacing.

The release keeps the managed one-command installation, persistent CUDA package
cache, conflict-safe launcher, verified update/repair flow, guided preparation,
fit testing, telemetry, checkpoint recovery, and PEFT adapter export.

Install from Windows PowerShell:

```powershell
$p="$env:TEMP\molt-install.ps1"; Invoke-WebRequest https://raw.githubusercontent.com/PraveenNimilka/MOLT/v0.12.0/install-global.ps1 -OutFile $p; if ((Get-FileHash $p -Algorithm SHA256).Hash -ne "75f7de635562447f4246a78634db56b4d11b4664b4c868036d3da1a8ffdeb7f5") { throw "MOLT installer hash mismatch" }; powershell -NoProfile -ExecutionPolicy Bypass -File $p
molt doctor
```

The latest clean one-million-target Qwen 1.5B development screen showed a
23.00% end-to-end time reduction, 38.45% higher training throughput, 1.07% lower
measured board energy, 12.09% lower allocated VRAM, and 9.37% lower reserved
VRAM than the tested Unsloth arm. The candidate did not win every measurement:
sampled whole-GPU peak was 4.65% higher and peak temperature was 7 C higher.
This single-seed screen is not a universal superiority claim.

MOLT remains a source-available research release under PolyForm Shield 1.0.0.
Controlled-clock five-seed endurance, Soup comparison, large-model endurance,
and independent reproduction remain open.
