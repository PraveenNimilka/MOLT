# Complete beginner quickstart

This is one exact Windows PowerShell recipe for MOLT `0.12.0`. It uses the
Apache-2.0 `Qwen/Qwen2.5-0.5B-Instruct` model at immutable revision
`7ae557604adf67be50417f59c2c2f167def9a775` and MOLT's 30-record demonstration
dataset. The dataset is intentionally tiny: this proves the complete local
workflow, not useful general model quality.

## Requirements

- Windows 10/11 x64 and an NVIDIA GPU with a compatible driver
- about 6 GB of free disk space for the managed runtime and model
- internet access for the initial downloads

## 1. Install the shared runtime

Open PowerShell in an empty working directory and run:

```powershell
$p="$env:TEMP\molt-install.ps1"; Invoke-WebRequest https://raw.githubusercontent.com/PraveenNimilka/MOLT/v0.12.0/install-global.ps1 -OutFile $p; if ((Get-FileHash $p -Algorithm SHA256).Hash -ne "75f7de635562447f4246a78634db56b4d11b4664b4c868036d3da1a8ffdeb7f5") { throw "MOLT installer hash mismatch" }; powershell -NoProfile -ExecutionPolicy Bypass -File $p
molt --version
molt doctor
```

Expected: `molt 0.12.0`; `doctor` must report `cuda_available: true`, the
NVIDIA GPU name, and `torch: 2.8.0+cu128`. Stop if CUDA is unavailable.

## 2. Download the exact model and sample files

```powershell
$moltPython="$env:LOCALAPPDATA\MOLT\runtime\Scripts\python.exe"
& $moltPython -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775', local_dir='model', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE','README.md'])"
New-Item -ItemType Directory -Force quickstart | Out-Null
$base = 'https://raw.githubusercontent.com/PraveenNimilka/MOLT/v0.12.0/examples/beginner'
Invoke-WebRequest "$base/molt-demo.jsonl" -OutFile quickstart/molt-demo.jsonl
Invoke-WebRequest "$base/configure.py" -OutFile quickstart/configure.py
Invoke-WebRequest "$base/reload_adapter.py" -OutFile quickstart/reload_adapter.py
```

The model's own license is Apache-2.0. The demonstration dataset and helper
scripts are distributed under MOLT's repository license.

## 3. Prepare and validate the exact configuration

```powershell
molt --json prepare quickstart/molt-demo.jsonl --model model --output quickstart/prepared
& $moltPython quickstart/configure.py quickstart/prepared/training.json
molt fit-test --config quickstart/prepared/training.json
```

Expected preparation facts:

```text
record_schema: prompt-completion
train_records: 27
validation_records: 3
train_tokens: 1291
validation_tokens: 135
source_sha256: eba10123241b2bf33d674bd5fdaa4d81b46e7adb773004c7375aa4ab2e7d6ef4
```

The generated configuration is explicit: context 128, batch 1, accumulation 1,
20 updates, learning rate `2e-4`, seed 1337, rank-8 all-linear QLoRA, fused FP32
AdamW, BF16 autocast, exact partitioned loss, and the joint static CUDA graph.
The fit test must end with `state: completed` before the full run.

## 4. Train, locate the run, and export

```powershell
molt train --config quickstart/prepared/training.json --mode-select normal -y
$run = Get-ChildItem quickstart/prepared/runs -Directory |
  Where-Object Name -ne 'fit-tests' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
molt report --run $run.FullName
molt export --run $run.FullName --output-dir quickstart/adapter
```

Expected: training ends in `COMPLETED`, records 2,560 session tokens and 20
committed updates, and export reports `Hugging Face PEFT safetensors adapter`.
Exact speed, energy, temperature, and NLL vary with the machine. The reference
RTX 4060 Laptop run is preserved in the [demonstration log](demo/launch-demo.json).

## 5. Reload and use the adapter

```powershell
python quickstart/reload_adapter.py --model model --adapter quickstart/adapter
```

The reference run prints:

```text
Adapter loaded: ...\quickstart\adapter
Response: THERMAL-READY
```

The exact response is a deterministic smoke-test target for this tiny dataset;
it is not evidence of general model quality. The adapter directory contains
`adapter_model.safetensors`, `adapter_config.json`, and `export.json`. Keep the
base model directory: an adapter is not a standalone model.

## Safety and licenses

MOLT is **source-available under PolyForm Shield 1.0.0**, not OSI open source.
The Qwen model and every other model or dataset retain their own licenses.
Do not treat this short smoke test as production validation or a benchmark.
