# Public launch draft — do not post until the comparison gate passes

The requested headline claiming a 37% Unsloth victory is intentionally withheld:
the repository does not contain the corresponding three-seed raw artifacts.
Use the copy below for the current evidence level.

## Reddit / r/LocalLLaMA

### MOLT 0.11 alpha: one-hour all-layer Qwen2.5-1.5B QLoRA on an RTX 4060 Laptop, plus an iso-clock comparison harness

I am releasing a source-available alpha of MOLT, a Windows-first local training
runtime focused on laptop memory limits, thermal behavior, recoverable training,
and measurement rather than short tokens/second screenshots.

The strongest completed endurance run so far used Qwen2.5-1.5B with rank-8 LoRA
adapters across all 28 decoder layers, context 512, batch one, accumulation four,
and FP32 fused AdamW. On one RTX 4060 Laptop GPU, seed 1337 completed 2,400
updates and 4,915,200 predicted tokens in 3,725.65 seconds:

- 1,319.29 end-to-end tokens/s
- 0.03625 GPU-board J/token
- 71 C peak GPU temperature
- 97.3% last-quarter/first-quarter throughput stability
- held-out NLL changed from 1.66353 to 1.10445
- 2.956 GiB PyTorch allocation and 3.750 GiB peak NVML board use

The run used a temporary 1,500--1,650 MHz graphics-clock range that was restored
afterward. It recorded 190.09 seconds of pacing, so I am not claiming zero thermal
control overhead.

This is one machine and one seed. It does not prove that MOLT beats Unsloth or
that the method generalizes. The repository includes
`benchmarks/reproduce_isoclock.py`, which runs a matched 16-update MOLT/Unsloth
screen and retains 100 ms telemetry. The full acceptance gate is three seeds,
alternating run order, matched held-out quality, and at least 30 minutes per arm.

I would value independent reproduction, especially on other RTX 4060 laptop
power/cooling designs. Please include failed or thermally aborted results rather
than rerunning only the favorable arms.

Repository: https://github.com/PraveenNimilka/MOLT

License: PolyForm Shield 1.0.0 for the current 0.11 source. Earlier published
MIT revisions retain their MIT grant.

## Hacker News

### Show HN: MOLT — measured, thermally bounded QLoRA experiments on Windows laptops

MOLT is a Windows-first training runtime that combines resident NF4 QLoRA,
memory-mapped token data, exact chunked vocabulary loss, selective activation
checkpointing, NVML telemetry, and atomic checkpoint recovery.

One Qwen2.5-1.5B endurance run on an RTX 4060 Laptop GPU completed 4.915M predicted
tokens in 3,725.65 seconds (1,319.29 end-to-end tokens/s), measured 0.03625 GPU
board J/token, peaked at 71 C, and retained 97.3% of first-quarter throughput in
the last quarter. All 28 layers had active rank-8 LoRA adapters; held-out NLL
changed from 1.66353 to 1.10445.

Important limitations: this is one machine and one seed; board energy excludes
the rest of the laptop; the clock range was explicitly constrained; PyTorch
allocation was 2.956 GiB while total NVML use was 3.750 GiB; and the run recorded
190.09 seconds of pacing. No general competitor advantage is claimed.

The release adds a reproducible single-seed iso-clock MOLT/Unsloth harness. A
three-seed, alternating-order, matched-time-to-NLL experiment is still required
before publishing a competitor headline.

Code and methodology: https://github.com/PraveenNimilka/MOLT

## Promotion gate

Replace this draft with a competitor headline only after all of the following
are committed as immutable artifacts or linked by content digest:

1. three seeds (1337, 2027, 4099) and alternating AB/BA order;
2. at least 30 minutes per arm;
3. identical model, token stream, LoRA inventory, optimizer, and validation objective;
4. both arms below the thermal boundary with verified 1,500--1,650 MHz clocks;
5. time and board energy to the same interpolated held-out NLL;
6. aggregate statistics generated from every valid arm, with failures retained.
