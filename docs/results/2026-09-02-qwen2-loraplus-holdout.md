# Qwen2 0.5B LoRA+ early-stop holdout

**Evidence label:** locally observed, three frozen holdout seeds.  
**Novelty label:** established LoRA+ prior art; not a MOLT algorithm.

## Question and gate

Can an explicitly early-stopped LoRA+ schedule reach validation quality within
1% of a tuned vanilla QLoRA run in materially less end-to-end time and GPU-board
energy, without increasing peak allocation or causing a thermal abort?

The frozen holdout gate required every seed to satisfy all of the following:

- both arms complete from a 52–53 C idle start under the same enforced 100 W
  power state;
- candidate final validation NLL no more than 1% above its paired baseline;
- at least 1% lower total run seconds and measured GPU-board joules;
- no more than 1% peak allocated-memory regression;
- identical model, dataset, token order for the seed, batch geometry, evaluation
  cadence, activation policy, quantization, and LoRA coverage.

## Workload and treatment

Both arms used the local Qwen2 0.5B checkpoint (494,032,768 stored parameters),
NF4 quantization, rank-8 all-linear adapters (4,399,104 trainable parameters),
resident activations, context 256, microbatch 2, accumulation 2, contiguous mmap
tokens, and held-out evaluation every five updates.

- Baseline: ordinary AdamW, one learning rate of `2e-4`, 10 updates.
- Candidate: PEFT LoRA+, A learning rate `1e-4`, B/A ratio 14 (B learning rate
  `1.4e-3`), five updates.

Different update counts are intentional: this is an end-to-end
time/energy-to-quality comparison, not an equal-token throughput comparison.

## Frozen holdout results

| Seed | NLL baseline | NLL candidate | NLL delta | Time improvement | Board-energy improvement | Peak allocation delta | Peak C baseline/candidate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8111 | 2.101345 | 2.107526 | +0.294% | 34.797% | 47.899% | 0.000% | 76 / 70 |
| 12143 | 2.101524 | 2.097191 | -0.206% | 37.181% | 48.761% | 0.000% | 77 / 70 |
| 16381 | 2.097816 | 2.092904 | -0.234% | 37.167% | 48.984% | 0.000% | 77 / 72 |

Median time improvement was **37.167%** and median GPU-board-energy improvement
was **48.761%**. All three candidate arms completed, all quality deltas were
inside 1%, peak allocation was unchanged, and neither arm reported the NVML
thermal-throttle flag in these holdout pairs.

A deterministic 10,000-resample paired aggregation produced descriptive 95%
median intervals of 34.797–37.181% for time, 47.899–48.984% for board energy,
and -0.234–+0.294% for NLL change. With only three pairs, these intervals do not
substitute for a larger statistical or external reproduction.

Machine-readable comparisons:

- `artifacts/qwen2-0.5b/comparisons/loraplus-holdout-seed-8111.json`
- `artifacts/qwen2-0.5b/comparisons/loraplus-holdout-seed-12143.json`
- `artifacts/qwen2-0.5b/comparisons/loraplus-holdout-seed-16381.json`

## Prior art and interpretation

LoRA+ was introduced by Hayou, Ghosh, and Yu in 2024. It assigns different
learning rates to LoRA matrices A and B and reports faster convergence. PEFT
ships `create_loraplus_optimizer`. MOLT's contribution here is controlled local
measurement, thermal/power-state checks, atomic artifacts, and a strict
comparison gate—not invention of the optimization.

- Paper: <https://arxiv.org/abs/2402.12354>
- PEFT implementation/docs:
  <https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md>

The result strengthens MOLT's tuned QLoRA baseline and is commercially useful
for this narrow workload. It does not establish a unique moat or algorithmic
novelty.

## Failed tuning points and limitations

- Ratio 8 was effectively tied with vanilla at 10 updates.
- Ratio 16 initially looked stronger, but one tuning seed exceeded the 1% NLL
  gate and longer training became unstable and thermally aborted.
- Ratio 20 overshot and regressed NLL by 2.10% on the worst tuning seed.
- Runs are short (roughly 4–7 seconds after model setup), one model/data stream,
  one laptop, and one context/batch geometry.
- NVML measures GPU-board energy, not wall-plug, CPU, display, or cooling energy.
- Early stopping at five updates is part of the treatment. Long-run stability is
  unproven and no claim should extrapolate this percentage to full fine-tunes,
  other models, or pretraining.
- External reproduction and broader model/task validation remain required.
- A frozen context-512 transfer kill test failed: candidate NLL regressed 3.43%,
  peak temperature rose 3 C, and a new throttle flag appeared. The validated
  claim is therefore context 256 only; see EXP-1033.
- A context-256 10-update continuation also failed: NLL improved at step 5 but
  rebounded to 2.153 at step 10, 5.15% worse than a completed 20-update vanilla
  reference. The contemporaneous control thermally aborted, so no paired
  time/energy number was claimed. EXP-1032 is an early-quality result only.

## Reproduction

```powershell
uv sync --extra qlora
$env:MOLT_QWEN_MODEL = "D:\path\to\your\Qwen2-0.5B-checkpoint"

# Repeat each seed, alternating arm order and cooling to the same start band.
uv run molt --json train --config configs/molt-stream-qwen2-0.5b-qlora-loraplus.json --seed 8111
uv run molt --json train --config configs/molt-stream-qwen2-0.5b-qlora-time-target-baseline.json --seed 8111

uv run molt --json compare BASELINE_RUN CANDIDATE_RUN `
  --quality-tolerance-percent 1 `
  --minimum-improvement-percent 1 `
  --output artifacts/qwen2-0.5b/comparisons/reproduction.json
```
