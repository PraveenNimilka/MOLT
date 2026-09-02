# EXP-1028: zoned thermal cruise control

## Status

Mechanism and short steady-state gate passed. Every longer candidate failed at
least one temperature, throttle, throughput, or stability gate. This is
engineering integration of established thermal pacing, not a novelty claim.

## Rationale and prior art

NVIDIA documents separate software-thermal, hardware-thermal, power-cap and
hardware-slowdown clock reasons. The slowdown reason can reflect temperature,
external power brake, excessive power draw, or a clock/P-state transition, so
MOLT records the reason bitmask rather than diagnosing from speed alone:
<https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html>.

NVIDIA's performance guidance says thermal throttling is around 85 C for many
GPUs, but explicitly describes this as a device-defined threshold and notes that
poor cooling can reduce stabilized performance before a thermal throttle event:
<https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html>.
NVML exposes slowdown, shutdown and maximum-operating threshold types; on Ada,
NVIDIA recommends field queries rather than the legacy threshold API:
<https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html>.

Local observations on the RTX 4060 Laptop GPU were:

- `nvidia-smi` target temperature: 75 C;
- legacy NVML query: slowdown 91 C and shutdown 94 C;
- previous MOLT speed degradation/heat soak: 72-73 C;
- software cruise target selected: 65 C;
- protective ceiling selected: 75 C;
- software checkpoint-and-stop boundary selected: 80 C.

The 80 C choice is intentionally below the user-proposed 85-90 C range. It is a
software policy, not a claim about the silicon's damage threshold.

## Exact controller

For raw GPU temperature `T`, target `a=65`, cruise ceiling `b=75`, cruise pauses
`p_min=10 ms`, `p_max=25 ms`, protective pause `p_guard=100 ms`, and abort
boundary `c=80`:

1. `T < a`: `pause=0`, state `full-speed`.
2. `a <= T <= b`: let `x=(T-a)/(b-a)` and `s=x^2(3-2x)`; then
   `pause=p_min+s(p_max-p_min)`, state `cooling`.
3. `b < T < c`: `pause=p_guard`, state `protective-cooling`.
4. `T >= c`: do not launch another optimizer step; state `thermal-abort`, save
   model/adapters, optimizer, data cursor and all RNG states with the atomic
   checkpoint store.

The smoothstep curve has zero slope at both cruise boundaries, avoiding sharp
pause jumps from the normal mapping. The protective zone is deliberately not
limited to 25 ms: that limit previously proved too weak once heat had escaped
the normal cruise window.

CUDA work is asynchronous. MOLT calls `torch.cuda.synchronize()` before making
the post-step decision, so the guard stops the next step; it cannot preempt a
kernel already executing. PyTorch defines synchronization as waiting for all
kernels in all streams on the device:
<https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html>.

## Cheap hardware result

Artifact:
`artifacts/molt-stream/benchmarks/production-throughput-20260902-075735.json`.

- Workload: 10,776,640 parameters, context 512, batch 16, accumulation 1,
  contiguous mmap packing, BF16, fused CUDA AdamW.
- Backend: `compile-max-autotune-no-cudagraphs`.
- Start temperature: 53 C.
- Warm-up: 10 updates.
- Measurement: 30/30 updates, 245,760 tokens.
- Sustained measured throughput including pauses: 126,920 tokens/s.
- Early/late throughput: 123,219 / 126,720 tokens/s; ratio 1.0284.
- Measured temperature: 68-70 C; range 2 C.
- Allocated VRAM peak: 1,600,279,040 bytes.
- Normal cruise pauses: 15.28-17.50 ms in the displayed samples.
- Thermal abort: false. Thermal throttle reason: false.
- Measured interval: 1.936 seconds and 124.218 GPU-board joules.
- Total setup, compilation, warm-up and measurement: 71.109 seconds.

Therefore the short steady-state target was exceeded, not merely reached. The
cold end-to-end rate is much lower because compilation dominates this short run.
No 500-step claim is made from a 30-step interval.

## Acceptance status

| Gate | Status |
|---|---|
| 10-25 ms smooth normal-cruise mapping | Passed by unit tests |
| Full-speed/cooling/protective/abort UI state | Passed by unit tests |
| Atomic recoverable thermal-abort checkpoint | Passed at mechanism level |
| All automated tests | 63 passed after implementation |
| >=80,000 measured tokens/s | Passed in one short hardware run |
| <=75 C measured interval | Passed in one short hardware run |
| Long-run stable equilibrium | Failed across four 200-update candidates |
| 500-step clean completion | Not tested / not passed |
| 80 C hardware-triggered abort | Not deliberately induced; mechanism tested |

Heating a laptop to the abort boundary solely to test software would be an
unnecessary stress test. Future fault injection should replace the telemetry
sampler with a deterministic fake in an integration harness.

## Long-run falsification and frontier

The long-run promotion gate required 200/200 measured updates, at least 80,000
tokens/s, peak at or below the configured cruise ceiling, no measured-interval
thermal event, and late/early throughput ratio at least 0.97.

| Artifact | Controller | tok/s | Peak C | Late/early | Thermal event | Decision |
|---|---|---:|---:|---:|---|---|
| `075926` | 65-75 C zones, 20 warm-up | 116,183 | 74 | 0.953 | Full-run event; interval not isolated in this schema | reject stability |
| `080046` | 65-75 C zones, 100 warm-up | 101,143 | 77 | 0.871 | interval SW thermal `0x20` | reject |
| `080305` | 65-72 C + hysteretic protection | 80,631 | 76 | 0.812 | interval SW thermal `0x20` | reject |
| `080544` | steady 40 ms + protection | 72,295 | 75 | 0.943 | none | reject throughput/stability |
| `080733` | steady 50 ms + protection | 68,432 | 75 | 0.919 | none | reject throughput/stability |

The requested zone scheme is the fastest of the long probes, but it enters the
driver's software-thermal slowdown state. Proactive steady pacing removes the
throttle bit, but the cooling system still needs enough recovery time to drop
aggregate throughput below 80,000 tokens/s. No result passed all gates, and no
500-step run is justified yet.

Profiles are retained with explicit experimental roles:

- `molt-stream-cruise.json`: exact 65-75 C requested control; not recommended
  for long unattended runs on this laptop.
- `molt-stream-cruise-safe.json`: 72 C hysteretic recovery experiment; still
  observed software-thermal slowdown.
- `molt-stream-cruise-steady.json`: 50 ms proactive no-throttle experiment;
  safest measured candidate, but below the requested throughput.
