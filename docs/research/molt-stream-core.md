# MOLT-Stream Core: research design and feasibility record

## Claim status

MOLT-Stream is an experimental clean-room prototype, not a demonstrated new
training method. It combines established mechanisms behind explicit capability
gates. No 8B fine-tune, 1B streamed pretraining run, 100 GB corpus run, 2x gain,
or original-algorithm claim has been established on this host.

## Prior art and novelty boundary

| Mechanism | Closest prior work | What MOLT currently adds | Status |
|---|---|---|---|
| Frozen layer streaming | Soup CLI streams one decoder layer at a time and reports an 8B NF4 LoRA run at 3.32 GB and 119.6 tokens/s | Independent NF4 payload, LoRA-only autograd, explicit copy/compute stream ablation, telemetry and thermal abort | Mechanism prototype; not novel |
| NF4 LoRA | QLoRA introduced NF4, double quantization and paged optimizers | Clean-room block-NF4 tensor and LoRA gradient integrity tests | Not bitsandbytes-compatible; no double quantization yet |
| Fused kernels | Liger supplies Triton RMSNorm, SwiGLU and fused losses | Explicit dispatch that refuses to call PyTorch fallbacks “Liger” | Liger unavailable on audited Windows runtime |
| Low-rank optimizer state | GaLore projects matrix gradients; Q-GaLore quantizes/adapts projections | Inspectable two-sided GaLore AdamW reference with full-rank equivalence test | Correctness prototype, not performance-ready |
| Memory-mapped data | OS mmap and `torch.from_file` | Deterministic contiguous windows and checkpointed cursor/RNG | Working; 100 GB empirical gate untested |
| Asynchronous copies | CUDA streams and pinned non-blocking copies | Four-layer packed-NF4 bundles with one stream wait per bundle | Median +4.11%; below +10% gate; experimental only |
| CUDA graphs | PyTorch `make_graphed_callables` | Graph-safety probe and explicit rejection of address-changing host restream backward | API works for resident static modules; current stream path is unsafe to capture |
| Windows fusion | PyTorch `torch.compile(max-autotune)` and optional native extensions | Pinned `triton-windows` 3.4 extra, short-path cache, explicit no-silent-fallback benchmark | No-graph mode works; graph mode launcher fails; native MSVC/nvcc route unavailable |

Primary sources:

- Soup repository: <https://github.com/MakazhanAlpamys/Soup>
- QLoRA: <https://arxiv.org/abs/2305.14314>
- GaLore: <https://arxiv.org/abs/2403.03507>
- Q-GaLore: <https://arxiv.org/abs/2407.08296>
- Liger: <https://arxiv.org/abs/2410.10989>
- PyTorch CUDA streams: <https://docs.pytorch.org/docs/main/notes/cuda.html>
- PyTorch `make_graphed_callables`: <https://docs.pytorch.org/docs/main/generated/torch.cuda.make_graphed_callables.html>
- PyTorch `torch.compile`: <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- PyTorch Windows Inductor prerequisites: <https://docs.pytorch.org/tutorials/unstable/inductor_windows.html>
- NVIDIA CUDA Graph constraints: <https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html>
- Triton-Windows compatibility matrix: <https://github.com/triton-lang/triton-windows>
- PyTorch mmap tensors: <https://docs.pytorch.org/docs/stable/generated/torch.from_file.html>

Licenses for external model weights, datasets and optional binary runtimes must
be reviewed at exact pinned revisions before an 8B experiment. MOLT does not
copy Soup, bitsandbytes, GaLore or Liger source code.

## Mathematical feasibility

For `N` frozen parameters and NF4 block size `B`, current host storage is
`M_host = N/2 + 4N/B bytes`. At `B=64`, this is `0.5625N`: about 4.50 GB for 8B
parameters and 7.88 GB for 14B, excluding adapters, Python, page-locking and
metadata. An 8B payload may fit in 16 GB RAM; 14B has little safe headroom.

If the largest layer contains `L` parameters, two BF16 layer buffers alone need
`4L` bytes. Adapters, activations, attention workspaces, embeddings/head, CUDA
context and fragmentation remain. A flat 3.5 GB peak is therefore a measurement
target, not a consequence of streaming.

Backward must re-stream every frozen layer. Ignoring metadata, an 8B update moves
at least `2 × 0.5625 × 8e9 = 9 GB` for forward and backward. At sustained PCIe
bandwidth `R`, transfer alone costs at least `9/R` seconds. Base matrix multiplies
also remain: LoRA reduces trainable state, not frozen-base FLOPs. Thus 35,000 LM
tokens/s is not credible as an 8B-streaming target on this RTX 4060; it is kept
only as a separate small-model gate.

## Stream algorithm

For bundle `j`, keep packed NF4 bytes/scales in pinned host memory, enqueue up to
four payloads for bundle `j+1` on `copy_stream`, issue one compute-stream wait for
bundle `j`, then dequantize and execute its layers in order. The dequantized base
is discarded after forward and re-streamed during backward; only input and LoRA
gradients are calculated. Eight layers therefore use two forward stream waits,
down from eight in the K=1 implementation.

`make_graphed_callables` is not enabled for this path. CUDA Graph replay requires
fixed tensor addresses and captures forward and backward. MOLT's backward reads
a different pinned host NF4 source for every layer. Capturing it would risk
replaying a stale captured address. The engine raises an explicit capability
error instead of silently producing wrong gradients. A graph-safe successor
requires persistent device bundle slots and must count those slots against VRAM.

For `W in R^(m×n)`, `A in R^(r×n)`, `B in R^(m×r)`:

`y = x W^T + (alpha/r) x A^T B^T`.

The custom backward is exact relative to dequantized frozen `W`; `W` receives no
gradient. Tests compare it with the resident dequantized operation.

For GaLore and matrix gradient `G`, MOLT stores Adam moments for `P^T G` when
`m <= n`, or `G P^T` otherwise, reconstructing the full update afterward. Moment
memory changes from `2mn` to `2r max(m,n)` elements. SVD cost and projection bias
remain material, so this is not promoted.

## Measured validation gates

Host: RTX 4060 Laptop 8 GB, 15.63 GiB RAM, Windows 11, PyTorch 2.8.0+cu128.

| Gate | Evidence | Result |
|---|---|---|
| <=3.5 GB for actual 8B QLoRA or 1B training | No licensed 8B model/dependencies or 1B streamed-optimizer run supplied | **Not tested / not passed** |
| <1.5 GB incremental RAM with actual 100 GB dataset | mmap test stayed below 100 MB on a small file; only 29.4 GB disk was free | **Mechanism passed; scale gate not tested** |
| >=35,000 small-model tokens/s | Eager profile: 127,854 tokens/s over 10 updates and 128,520 over 6; compiled no-graph profile: 175,256 for one update before thermal abort | **Short-window throughput passed; sustained reliability not passed** |
| <=70 C acceptance / <=72 C safety | Production probes peaked at 71 C, 72 C, then 74 C after compiler warm-up; no throttle bit; automatic abort executed | **Both gates failed in compiled probe** |
| Monotonic held-out perplexity and exact recovery | 200-step NLL 87.00 -> 0.306; next-update model/optimizer/data/RNG test bit-exact | **Passed for smoke workload** |

The K=4 width-2048 grouped ablation used seeds 11, 22 and 33 for 60 updates.
Relative to paired synchronous runs, throughput changed by -11.77%, +4.11% and
+4.11% (median +4.11%). Energy was mixed. An earlier 30-update probe happened to
show +10.50%, demonstrating why the unreplicated number could not be promoted.
These are residual-linear activation vectors, not language tokens.

The production profile is `configs/molt-stream-production.json`: context 512,
batch 24, contiguous mmap packing, BF16 autocast and fused AdamW. It contains a
10,776,640-parameter SLM, not the immutable 49.88M workload and not a 1B/8B
model. Its brief measured windows exceeded 35k with 98-99% sampled GPU
utilization, allocated 2.474 GB, and used 1.214 GB process RSS, but both attempts
crossed the stricter 70 C acceptance ceiling. Therefore this is not evidence of
consistent thermally safe throughput or of the large-model memory gate.

The Windows fusion artifact measured eager incremental forward/backward memory
at 2,537,330,176 bytes. Installing the optional, PyTorch-2.8-compatible
`triton-windows==3.4.0.post21` enabled real fusion. Steady
`max-autotune-no-cudagraphs` measured 2,172,671,488 incremental bytes, a 14.37%
reduction, and 0.0668 seconds versus eager 0.2896 seconds for one cached
forward/backward. The first compiled update took 11.71 seconds. Requested
`max-autotune` with CUDA Graphs fails in the Windows static launcher with
`OverflowError: Python int too large to convert to C long`. A misleading early
probe that called `.loss` on the wrapper bypassed compiled `forward` and was
discarded. System `cl`, `nvcc`, Ninja and `CUDA_HOME` remain absent, so the
separate clean-room C++/CUDA extension route cannot be compiled or verified.
The 40% memory-reduction gate is **not passed**.

## Next falsification

Reproduce and isolate the Triton-Windows static-launcher overflow, then compare
the supported no-graph compiler backend over a thermally controlled full run.
For a native-extension comparison, install MSVC, matching CUDA Toolkit/nvcc and
Ninja in a separate pinned environment. For streaming, prototype persistent K=4 device slots and
graph only the graph-safe compute section, then compare against synchronous and
ungrouped paths in randomized order. Promotion still requires >=10% repeated
end-to-end gain, correct gradients, and the full memory/thermal gates.

## Exact reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\molt.exe inspect
.\.venv\Scripts\molt.exe stream-tune --width 2048 --layers 8 --sequence 128 --steps 60 --bundle-size 4 --seed 11
.\.venv\Scripts\molt.exe fusion-benchmark --config configs\molt-stream-production.json
.\.venv\Scripts\molt.exe benchmark --config configs\molt-stream-production.json --warmup-steps 10 --steps 30
```

## Thermal duty-cycle falsification

MOLT now supports a run-level thermal target, emergency abort, bounded dynamic
30-50 ms inter-step pauses, enforced-power telemetry, and an explicit NVML power
limit command. Power changes are never implicit. The requested 65 W limit was
inside NVML's reported 5-140 W constraints, but an elevated set request failed
with `NVMLError_NoPermission`; the enforced limit remained 115 W.

The exact `molt-stream-cool.json` cheap kill test used context 512, microbatch
16, accumulation 2, contiguous mmap data and cached no-graph compilation. It
completed five warm-up and 8/30 measured steps before the latest sampled
temperature exceeded the 72 C emergency boundary. Peak temperature was 74 C.
The controller inserted 12 pauses totaling 0.4933 seconds. Measured-interval
throughput was 115,172.5 tokens/s, peak allocated VRAM was 1.644 GB, and the
enforced power limit was 115 W. Thus the <=62 C, >=150,000 tokens/s and
zero-abort gates all failed. A 500-step run was killed before launch under the
preregistered cheap-falsification rule.

This result shows that a fixed tens-of-milliseconds sleep is a scheduler, not a
temperature lock. Temperature depends on cooling capacity, ambient conditions,
firmware power control and thermal inertia. MOLT does not claim it can hold
58-60 C on this hardware.

## Predictive thermal cruise experiment (EXP-1027)

MOLT now has an opt-in `predictive-cruise` controller. This is a MOLT-specific
implementation, **not a novelty claim**. GPU power/temperature scheduling and
DVFS are established research areas. NVIDIA documents separate software thermal,
hardware thermal, power-cap and hardware-slowdown clock reasons, so the sampler
records the reason mask rather than inferring throttling from throughput alone:
<https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html>.
PyTorch officially exposes `max-autotune-no-cudagraphs`; MOLT uses that exact
backend name rather than implying CUDA Graph capture:
<https://docs.pytorch.org/docs/stable/generated/torch.compile.html>.

For sampled temperature `T_k`, filtered temperature `F_k`, filtered positive
heating rate `S_k`, and horizon `h`, the predictive signal is
`T_hat = F_k + h max(0, S_k)`. A bounded pause/compute ratio is updated smoothly
from this signal. When a power budget `P_target` is configured, an immediate
feed-forward lower bound is
`r_power = max(0, (P_active - P_target)/(P_target - P_idle))`, with conservative
`P_idle <= 15 W`. The pause is `min(pause_max, r * compute_seconds)`. Raw
temperature above the abort boundary still wins over the controller. The trainer
also rechecks temperature after sleeping, fixing a race where a delayed final
NVML sample could arrive after the pre-pause safety check.

Pre-registered steady-state gates were: at least nine measured updates, raw
temperature range <=3 C, late-third/early-third throughput >=0.97, no thermal
clock event, no abort, and peak <=70 C. Speed includes controller sleeps. The
fixed model/data/effective batch were 10,776,640 parameters, context 512,
contiguous packing, BF16 autocast, fused CUDA AdamW, and 16,384 tokens/update.

| Artifact | Backend / updates | Sustained tok/s | Late/early | Peak C | Range C | Outcome |
|---|---:|---:|---:|---:|---:|---|
| `203016` | compiled, 2/20 | 164,242 | insufficient | 74 overall | insufficient | safety abort |
| `203128` | compiled, 0/50 | 0 | insufficient | 73 overall | insufficient | safety abort |
| `203347` | eager, 50/50 | 66,661 | 0.881 | 72 | 10 | reject |
| `203618` | eager, 60/60 | 27,643 | 0.798 | 69 | 8 | reject; thermal event |
| `223253` | eager, 60/60 | 103,858 | 0.945 | 64 | 7 | best short safe run; stability miss |
| `223401` | eager, 100/100 | 43,366 | 0.576 | 73 | 11 | reject |
| `223807` | eager + 50 W feed-forward, 100/100 | 45,000 | 0.586 | 72 | 9 | reject |

The compiled burst exceeds 150k tokens/s but is not a valid sustained result.
The strongest safe short interval misses the stability threshold by 2.55
percentage points. The 50 W run began at 51 C and averaged 49.0 W, yet heat-soak
still reached 72 C; average board power is therefore not a universal temperature
set point. The firmware-enforced limit also varied across experiments and the
earlier 65 W set request failed for permission. NVIDIA documents that changing a
power limit requires privileged access and may not persist:
<https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html>.

No configuration passed all gates. `molt-stream-max-throughput.json` remains a
compiled candidate, `molt-stream-max-throughput-eager-control.json` is the
diagnostic control, and `molt-stream-sustainable-35w.json` is unverified after a
bounded run was manually killed. None should be described as a production
thermal solution.

## Zoned cruise successor (EXP-1028)

The simpler `zone-cruise` successor isolates normal pacing from emergency
protection. It uses zero pause below 65 C, a smooth 10-25 ms curve from 65-75 C,
a 100 ms protective recovery pause from 75-80 C, and an atomic checkpoint/stop
at 80 C. The boundary was lowered from the requested 85-90 C because the local
GPU reports a 75 C target and earlier MOLT runs degraded at 72-73 C.

The first 30-update measured interval reached 126,920 tokens/s including pauses,
held 68-70 C, allocated 1.600 GB, had a late/early speed ratio of 1.028, and
reported no throttle or abort. Total cold run time was 71.109 seconds because
compiler setup dominated the 1.936-second measurement. This is a passed cheap
mechanism test, not long-run production acceptance. Full rationale and raw gate
status are in `docs/results/2026-09-02-zone-cruise.md`.

Longer probes rejected promotion. The exact 75 C controller reached 101,143
tokens/s after 100 warm-up updates but peaked at 77 C, recorded measured-interval
software-thermal slowdown (`0x20`), and fell to a 0.871 late/early ratio. A 72 C
hysteretic profile reached 80,631 tokens/s but still peaked at 76 C and recorded
the same slowdown bit. Proactive fixed-duty variants removed the throttle event,
but 40 ms and 50 ms pacing produced only 72,295 and 68,432 aggregate tokens/s,
with stability ratios below 0.97. Thus the requested joint gate was not achieved.
