# MARS v1: correctness first, then measured resource shaping

Status: research candidate, **not a demonstrated new algorithm or competitive advantage**.
Date: 2026-09-04. Input: user's `MOLT_ Adaptive AI Training Runtime.md`.

## Correctness finding

Test 110's local custom model applies Q/K RMS normalization but its exported
configuration selects Qwen2ForCausalLM. That loader reported discarding those
weights. The custom model source confirms those operations belong in forward.
MOLT now rejects nonempty missing/unexpected/mismatched/error loading reports
before attaching adapters. No original checkpoint/configuration was modified.
Restoring this custom model needs an audited adapter and reference-logit test,
not renaming it Qwen3. Its training provenance is not established here; QLoRA
of randomly initialized frozen weights is not equivalent to full pretraining.

## What competitors actually improve

| Primary source | Mechanism / evidence | Implication for MOLT |
| --- | --- | --- |
| [Unsloth checkpointing](https://unsloth.ai/blog/long-context) | Authors report asynchronous activation offload and memory benefits; hardware/workload-specific | Offload is prior art, not automatically faster on laptop PCIe |
| [Unsloth gradient accumulation](https://unsloth.ai/blog/gradient) | Correct token-normalization across accumulated batches | Preserve valid-token weighting, not merely batch-size products |
| [Unsloth repository](https://github.com/unslothai/unsloth) | Specialized training implementation and kernels | Accuracy is not created by tok/s; correct architecture, tokenizer, data and loss remain necessary |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware exact attention avoids full attention intermediates | Fewer memory transfers can improve speed without changing objective |
| [Cut Cross Entropy](https://arxiv.org/abs/2411.09009) | Avoids materializing full vocabulary logits | Existing MOLT partitioned loss must be compared with this prior art |
| [Zeus](https://ml.energy/zeus/optimize/batch_size_optimizer/) | Batch/power optimization targets time and energy to accuracy | Adaptive batch/power tuning alone is not MOLT novelty |

No competitor source code was copied or dependency added. Before distributing
any integration, review licenses of the exact pinned files and dependencies,
not just the repository badge; model and dataset licenses are separate.
Patent clearance and comprehensive prior-art review have not been performed.

## Refined candidate: confidence-gated thermal switch amortization

Keep the base model resident if it fits. At optimizer boundaries choose among
measured geometries `a=(microbatch, accumulation)` satisfying `B*G=E`.
Context, token order, optimizer, LR schedule and loss weighting stay fixed.
For E=1 this action space has no larger microbatch: do not silently change E.

Let `t_a` and `e_a` be seconds/update and board joules/update, with measured
uncertainty bounds. For a proposed remaining dwell of H updates and switching
costs C_t/C_e, accept a candidate only if conservative savings are positive:

```
H * (lower(t_current) - upper(t_candidate)) > C_t
H * (lower(e_current) - upper(e_candidate)) > C_e
upper(predicted_temperature(candidate, H)) < safety_boundary
upper(predicted_peak_memory(candidate)) < usable_memory
```

Require a prevalidated <=1% NLL regression bound, minimum dwell and fresh
telemetry. Otherwise stay LOCKED; stale/missing safety telemetry is not proof
of safety. WATCH updates prediction; ADAPT is only at an optimizer boundary.
The hypothesis is that switch-cost and thermal-risk gating beats greedy
throughput tuning under changing background contention. **Novelty unproven**:
this also resembles constrained model-predictive and switching-cost control.
This rule is a proposal, not integrated production behavior.

The user's mixed-unit risk objective needs dimensionless normalization.
Throughput/power is tokens/joule (the same information as reciprocal
joules/token), not a new physical efficiency measure. GPU temperature cannot
certify CPU temperature. CPU temperature is currently unmeasured.

## First falsification experiment

`python -m molt_stream.training.mars_probe` runs independent fresh processes,
three seeds (1337,2027,3407), alternating geometry order, E=4: 1x4 versus 2x2.
It preserves each spec, stdout, stderr, process wall time and run artifact path;
errors or incomplete/thermal-aborted runs stop further trials, not disappear.
This is an inexpensive geometry kill test, **not a tuned-baseline comparison**.
Twelve updates cannot establish convergence, thermal equilibrium or 1% gains.

Current controlled model: intact local Qwen checkpoint, distinct from Test 110.
Data: locally available TinyStories-valid text tokenized with that model's own
tokenizer. A contiguous 10% tail is held out; this is not a document-disjoint
split or an uncontaminated public benchmark. Do not extrapolate its quality.
No system power limits, fan controls or other applications are changed.

Full acceptance remains preregistered: target validation NLL chosen before
trials, interpolated first crossing including setup/evaluation overhead, paired
seeds and confidence intervals, longest representative thermal runs, all costs
including calibration/switching. >=1% reproducible gain is useful, >=25% strong,
>=50% target. No time/energy regression; quality within1%; memory within fit
budget. Unknown sensors and uncrossed targets produce inconclusive results.

## Reproduction

From repository root, substitute your local model/data paths:

```powershell
.\.venv\Scripts\python.exe -m molt_stream.cli --json prepare --text-file "D:/Projects/MOLT AI/Dataset/TinyStories-valid.txt" --tokenizer "D:/Projects/MOLT TEST/AI MODELS/QWEN" --base-model "D:/Projects/MOLT TEST/AI MODELS/QWEN" --output-dir artifacts/mars-v1-20260904/dataset
.\.venv\Scripts\python.exe -m molt_stream.training.mars_probe --config artifacts/mars-v1-20260904/dataset/training.json --output artifacts/mars-v1-20260904/geometry --steps 12
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q
```

Outputs deliberately refuse existing directories. Dataset manifest records
source/token hashes and tokenizer location. Preserve original failed trials.
Before head-to-head Unsloth work, use a separate pinned environment and record
model/tokenizer hashes, driver/runtime, effective batch, masking, LoRA targets,
optimizer precision, evaluation and all startup costs. No Unsloth head-to-head
has been executed in this cycle.

## Observed kill-test results (not an acceptance benchmark)

All six runs completed, each 12 updates / 6,144 tokens, context128, effective
batch4, NF4 QLoRA all-linear adapters, default AdamW, checkpointing enabled.
Runtime: Python3.12.13, torch2.8.0+cu128, transformers5.16.1, peft0.20.0,
bitsandbytes0.50.2, RTX4060 Laptop8GB. Baseline source commit2d27940 plus
the loading-integrity patch in this working tree. No dependency changes.

| Seed | Process seconds 1x4 / 2x2 | Board J 1x4 / 2x2 | Final NLL 1x4 / 2x2 |
| --- | --- | --- | --- |
| 1337 | 25.845 / 19.263 | 469.53 / 470.87 | 2.15959 / 2.16115 |
| 2027 | 39.116 / 43.556 | 667.64 / 627.36 | 2.15154 / 2.15052 |
| 3407 | 27.119 / 17.272 | 564.54 / 461.92 | 2.15050 / 2.15005 |

Decision: **reject-or-inconclusive**. Two pairs fail a no-regression screen;
no repeatable joint advantage established. Differences smaller than noise are
not evidence of regression either. Do not promote on the favorable third seed.
CPU tests/read-only diagnostics overlapped parts of this first sweep; background
load was not controlled. Repeat in isolation AND under a specified concurrent
app workload. Peak GPU temperature and allocated VRAM remain in every summary.

The first sweep retains evaluation-time energy readings and summary telemetry,
but not individual sensor samples. Subsequent QLoRA runs now also persist
`telemetry.samples.json`. Board joules omit interpreter startup/teardown;
process seconds include them. No interpolated target was preregistered for this
short screen, so these are fixed-token costs, not time/energy-to-quality claims.
Dataset source SHA256:
`94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`.
Prepared tokens:4,114,577 train /457,175 validation; all six trials use identical
files. Full text tokenization does not mean all tokens were trained on.

Artifacts: `artifacts/mars-v1-20260904/geometry/manifest.json`, `screen.json`,
per-trial configs/logs/checkpoints/metrics; local artifacts are gitignored.
Custom-checkpoint metadata audit found exactly40 unexpected Q/K norm tensors,
with zero missing keys relative to its configured Qwen2 model. This establishes
the mismatch, not correctness of the rest of the custom implementation.

Next experiment: preregister a target on separate pilot data, collect longer
paired stationary and controlled-contention profiles, then test switch-payback
gating against the strongest fixed geometry and a greedy adaptive baseline.
Require gradient checks on the real quantized model before live switching.
Custom1B support needs its own adapter equivalence milestone first.
