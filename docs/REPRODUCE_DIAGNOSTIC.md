# Reproduce the public diagnostic comparison

This page is the single measurement contract behind the public Qwen,
Llama-family, and Gemma diagnostic figure. The result is strong early evidence,
not an official superiority claim: graphics clocks were uncontrolled, one Gemma
pair had a competitor power-envelope mismatch, Soup was not tested, and there
is no independent reproduction.

## Software and hardware

- MOLT result release: `0.11.0a3`, commit
  `5c2b805e90e094d0c67ed423970fa895117c160f`
- Current reproduction harness: `benchmarks/family_unsloth_matrix.py` and
  `benchmarks/qwen_unsloth_ab.py`
- Competitor: Unsloth `2026.9.2`
- PyTorch: `2.8.0+cu128`; CUDA runtime `12.8`; Triton Windows `3.4.0`
- Transformers in the isolated competitor runs: `5.5.0`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB, WDDM, 140 W enforced
  power-limit samples

Each engine ran in its own Python process and environment. Competitor packages
are not MOLT dependencies.

## Registered workloads

| Family | Exact model identity | Context | B/G | Steps | Target NLL | MOLT schedule |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen | Qwen2 0.5B; config SHA-256 `18e18a…f45`; weights SHA-256 `fdf756…b7fe` | 1,024 | 1/1 | 16 | 1.85 | stride 1, analytical exact loss, static graph |
| Llama | SmolLM2 360M; config SHA-256 `34f780…168a`; weights SHA-256 `7aaff6…40f` | 512 | 1/1 | 16 | 3.75 | stride 1, analytical exact loss, joint static graph |
| Gemma | Gemma 3 270M; config SHA-256 `36b398…ab9`; weights SHA-256 `4f3c61…09d2` | 512 | 1/1 | 16 | 4.90 | stride 3, exact chunk-32 loss, static graph |

All workloads used rank-8, alpha-16 all-linear LoRA, learning rate `2e-4`, exact
unfiltered shifted causal cross-entropy, and fused FP32 AdamW. Seeds were 1337,
2027, and 4099. Both `Unsloth → MOLT` and `MOLT → Unsloth` orders were run for
every seed: six pairs and twelve arms per family.

The complete hashes, dataset hashes, thermal parameters, and execution settings
are preserved by the three contract SHA-256 values in
[`0.11.0a3-diagnostic.json`](benchmarks/0.11.0a3-diagnostic.json):

- Qwen: `f6e39fd702f8da26369ab059874704d87ce47defe76463e2be6b84fb56c85d6b`
- Llama: `4f2b788b00af533c5b8e60f7a53b19d8a9ae462f7eadff77fe13e7f00972cbe0`
- Gemma: `b61ce7653ccfbe8d246fde0f88d4be1db51f3bff10e89e0e37a64d54f42f5960`

## Baseline and quality rules

The Unsloth arm used the same local weights, prepared token order, validation
tokens, adapter topology, learning rate, optimizer class/state precision,
effective batch, update count, and exact loss objective. Both arms used the same
predictive thermal policy: 50 °C target, 55 °C microbatch guard, 72 °C abort,
three-second lookahead, 1.5 °C stability band, 25 W average-power target, and a
one-to-three-second pause range.

A pair is eligible only when both arms complete, have the same trainable
parameter count, begin from the registered adapter state, and cross the same
held-out NLL. Crossing time and energy are linearly interpolated between adjacent
evaluations. Initial and final held-out NLL are also retained; all displayed
pairs reached their registered target. No training loss alone is accepted as a
quality comparison.

## Time, memory, and energy definitions

- **Elapsed time:** end-to-end seconds from the arm's recorded training session;
  public comparisons use interpolated time to the same held-out-NLL crossing.
- **Total measured arm time:** Qwen 737.458 s, Llama 420.602 s, Gemma 358.675 s;
  1,516.736 s across all 36 arms. This sum excludes the external cold-start waits
  between arms and is not presented as benchmark throughput.
- **Allocator peak:** `torch.cuda.max_memory_allocated()` reset immediately
  before model construction, then measured across model load, adapter creation,
  graph capture, training, evaluation, and checkpoint preparation. It is distinct
  from NVML board residency.
- **Energy:** GPU-board power sampled through NVML at 100 ms. Joules are the
  trapezoidal integral of measured board watts over monotonic timestamps;
  comparisons use energy interpolated at the same NLL crossing. CPU and wall
  energy are not included.

## Reproduction procedure

1. Prepare legal local copies of the three exact checkpoints and the token files
   matching the published SHA-256 values.
2. Create an isolated MOLT environment and an isolated Unsloth `2026.9.2`
   environment with the versions above.
3. Select the matching public template in `benchmarks/templates/` and verify
   every model/data hash before running.
4. Run the public matched harness once per family, changing only the template,
   local model/data paths, target NLL, and new output directory:

```powershell
python benchmarks/qwen_unsloth_ab.py `
  --molt-template benchmarks/templates/qwen-diagnostic.json `
  --unsloth-site D:/isolated/unsloth-site `
  --model D:/models/qwen2-0.5b `
  --train-data D:/data/qwen/train.bin `
  --validation-data D:/data/qwen/validation.bin `
  --target-nll 1.85 --output D:/new-results/qwen `
  --steps 16 --evaluation-interval 4 --validation-batches 1 `
  --clock-control managed --graphics-clock-min-mhz 1500 `
  --graphics-clock-max-mhz 1650 --start-temperature-c 50 `
  --cold-dwell-seconds 10 --expected-enforced-power-limit-watts 140 `
  --thermal-target-c 50 --thermal-guard-c 55 --thermal-abort-c 72 `
  --thermal-lookahead-seconds 3 --thermal-stability-band-c 1.5 `
  --thermal-power-target-watts 25 --thermal-initial-pause-seconds 1 `
  --thermal-max-pause-seconds 3 --telemetry-interval-seconds 0.1
```

Use `llama-diagnostic.json`, its matching paths and target `3.75` for Llama;
use `gemma-diagnostic.json`, its paths and target `4.90` for Gemma. Managed
clock control requires an Administrator terminal and supported NVIDIA clock
control. Use `--clock-control uncontrolled-diagnostic` only for a result that is
explicitly ineligible for promotion.

5. Preserve `protocol.contract.json`, `results.json`, per-arm metrics, telemetry,
   stdout/stderr, and environment freezes. Never reuse an existing output path.

The coordinator fails closed unless Qwen, Llama, and Gemma all contain the three
seeds and both run orders. For an official result, set and verify the same
supported graphics-clock range for both arms and reject every power-envelope
mismatch. The checked-in aggregate can be regenerated with
`py -3.12 tools/render_benchmark_summary.py`.
