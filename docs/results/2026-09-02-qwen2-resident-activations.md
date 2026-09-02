# EXP-1031 — Qwen2 resident-activation QLoRA

Date: 2026-09-02  
Decision: **Milestone 3 engineering gate passed; no novelty claim**

## Research question

Can a QLoRA workload that already fits in GPU memory avoid PEFT's default
activation recomputation and convert the additional VRAM into repeatably lower
end-to-end time and GPU-board energy at identical validation quality?

PEFT documents that `prepare_model_for_kbit_training` enables gradient
checkpointing by default and that it saves memory at the expense of a slower
backward pass. PyTorch describes the same established compute-memory tradeoff.
This experiment therefore tests configuration selection and measurement; it is
not a new training algorithm.

Primary sources:

- <https://huggingface.co/docs/peft/package_reference/peft_model>
- <https://docs.pytorch.org/docs/stable/checkpoint>

## Controlled workload

- Local Qwen2 causal LM: 494,032,768 stored parameters, NF4 QLoRA
- Trainable rank-8 adapter parameters: 4,399,104
- Context 256, physical batch 2, accumulation 2: 1,024 tokens/update
- Twenty AdamW updates, fixed sequential token order and four fixed validation
  batches per evaluation
- Seeds 1337, 2027 and 4099 in paired AB/BA execution order
- Baseline: PEFT default activation checkpointing enabled
- Candidate: activation checkpointing disabled; ordinary activations retained
- RTX 4060 Laptop GPU 8GB, Windows 11; ordinary desktop applications remained
  open and recorded mean system CPU load was approximately 12%

The only intended independent variable was `activation_checkpointing`. Final
NLL was identical within every seed, which is expected because Qwen2 and the
LoRA configuration use zero dropout and both arms consumed the same tokens in
the same optimizer batches.

## End-to-end results

Positive values favor resident activations.

| Seed | End-to-end time | Update time | Board energy | Final-NLL change |
|---:|---:|---:|---:|---:|
| 1337 | +12.83% | +12.81% | +12.22% | 0.000% |
| 2027 | +21.96% | +23.72% | +22.30% | 0.000% |
| 4099 | +24.51% | +26.06% | +25.27% | 0.000% |
| Median | **+21.96%** | **+23.72%** | **+22.30%** | **0.000%** |

## Interpolated common-quality result

Evaluation checkpoints contain cumulative end-to-end seconds and trapezoidally
integrated NVML board joules. Linear interpolation to validation NLL 2.10 gave:

| Seed | Time to NLL 2.10 | Energy to NLL 2.10 |
|---:|---:|---:|
| 1337 | +12.05% | +10.57% |
| 2027 | +21.91% | +22.79% |
| 4099 | +27.88% | +28.87% |
| Median | **+21.91%** | **+22.79%** |

## Explicit tradeoffs and limitations

- Peak PyTorch allocation increased from 1.67–1.90 GiB to 3.27–3.45 GiB.
- Peak total NVML GPU usage increased to 4.71–4.96 GiB. This remained reliable
  with the desktop workload, but it is not suitable when that headroom is absent.
- Peak temperature was 59–67 C and no arm aborted. Both arms reported NVIDIA's
  software thermal clock-event bit, so this is no proof of zero throttling; it
  is evidence of no thermal regression between arms.
- NVML board energy excludes CPU, display and wall-power energy.
- Twenty updates are enough for the preregistered isolated-method gate, not for
  a general model-quality or long-duration thermal claim.
- The result is one model size on one machine. Cross-model and external
  reproduction belong to Milestone 4.

## Failed alternatives preserved

- Exact-loss token partitioning saved 16–34% allocation but slowed updates by
  6.7–9.0% (EXP-1030).
- Physical batch 4 at the same effective batch improved time by only 3.9–8.3%
  and increased energy by 2.2–8.7%; rejected.
- BF16 autocast on the LoRA path reduced one probe's energy by 19.6% but slowed
  end-to-end time by 17.2%; rejected.
- Fused CUDA AdamW reduced observed throughput on this small adapter optimizer;
  it was reverted.

## Reproduction

The immutable arm configurations are:

- `configs/molt-stream-qwen2-0.5b-qlora-checkpointed-activations.json`
- `configs/molt-stream-qwen2-0.5b-qlora-resident-activations.json`

Run each arm from the same starting-temperature band and alternate arm order:

```powershell
uv sync --extra qlora
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\molt.exe --json train --config configs\molt-stream-qwen2-0.5b-qlora-checkpointed-activations.json
.\.venv\Scripts\molt.exe --json train --config configs\molt-stream-qwen2-0.5b-qlora-resident-activations.json
```

Raw paired artifacts, including resolved specifications, evaluation traces,
telemetry summaries, checkpoints and digests, are under
`artifacts/qwen2-0.5b/milestone3-reproduction/`.
