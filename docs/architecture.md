# Public product boundary

MOLT exposes one command-line workflow for local model preparation, fit testing,
training, recovery, evaluation, and export.

## Supported boundary

- Local Hugging Face-format model and tokenizer directories
- Text, JSONL, and Parquet input preparation
- Scratch-training and supported QLoRA configurations
- Hardware telemetry and explicit thermal policy
- Verified atomic checkpoints and resumable runs
- PEFT safetensors and supported GGUF adapter export

Stable customer operations are exposed through the `molt` CLI. Experimental
runtime paths remain capability-gated and may change during the alpha series.

Implementation-specific kernel design, execution schedules, memory plans,
mathematical derivations, and profiling strategy are proprietary and are not
documented in the public product boundary.

For usage, see the [README](../README.md) and [CLI reference](cli.md). For trust
boundaries, see the [security policy](../SECURITY.md).
