# Hardware capability report — 2026-09-01

## Observed system

Collected locally on 2026-09-01 (Asia/Colombo).

| Component | Observed value | Evidence/limitation |
|---|---|---|
| OS | Windows 11 Home, 64-bit, 10.0.26200 build 26200 | Windows CIM |
| Machine | ASUS ROG Strix G614JV | Windows CIM |
| CPU | Intel Core i7-13650HX, 14 cores / 20 logical processors | Windows CIM |
| RAM | 15.63 GiB usable; 2×8 GiB SK Hynix DDR5-4800 | Windows CIM |
| Discrete GPU | NVIDIA GeForce RTX 4060 Laptop GPU | `nvidia-smi` |
| VRAM | 8,188 MiB | `nvidia-smi`; preferred over a truncated CIM value |
| GPU compute capability | 8.9 | `nvidia-smi` |
| GPU driver | 610.62; Windows driver 32.0.16.1062 | `nvidia-smi` / CIM |
| Driver CUDA API | 13.3 | This is driver capability, not an installed toolkit |
| GPU default power limit | 100 W | Requested/current limit was unavailable in one query |
| Integrated GPU | Intel UHD Graphics | Not proposed as a training backend initially |
| NPU | None exposed by present-device inspection | Absence is observational, not proof of silicon absence |
| Project drive | D: NTFS, 166.0 GiB total, 51.2 GiB free | Windows volume query |
| OS drive | C: NTFS, 283.5 GiB total, 34.9 GiB free | Windows volume query |

At inspection time, desktop processes occupied about 0.8 GiB VRAM. Baseline
runs must record pre-run free VRAM and reject trials with material background
GPU load rather than silently accepting noisy measurements.

## Software environment

| Tool/runtime | Status |
|---|---|
| Python | Not installed; Windows launcher present but reports no installations |
| `pip` / ML frameworks | Not installed/discoverable |
| `uv` | 0.11.26 |
| Git | 2.54.0.windows.1 |
| Node.js | 26.3.0 |
| .NET | Runtime 8.0.21 and 10.0.10; no SDK |
| CUDA Toolkit / `nvcc` | Not installed |
| CMake / Ninja / C/C++ compilers | Not discoverable |
| WSL | Not installed |

**Interpretation:** use native Windows with a repository-local `uv` environment
and official PyTorch wheels first. Do not require a local CUDA Toolkit for the
baseline. Treat `torch.compile`, custom extensions, bitsandbytes, and
FlashAttention as separate compatibility experiments, not baseline defaults.
PyTorch officially supports Windows with NVIDIA CUDA wheels, while its Windows
Inductor path requires a compiler for relevant backends:
<https://docs.pytorch.org/get-started/locally/> and
<https://docs.pytorch.org/tutorials/unstable/inductor_windows.html>.

## Measurement capabilities

- `nvidia-smi` exposes board power with a documented ±5 W accuracy and a 1 s
  average on Ampere-or-newer GPUs. Sample `power.draw` and utilization at 5–10 Hz
  where the driver permits, integrate by trapezoid rule, and label the result
  **GPU board energy estimate from sampled power**.
- The command-line query does not expose `total_energy_consumption` on this
  system, despite newer NVML APIs defining a total-energy counter. Feature-detect
  rather than assume support.
- RAM can be sampled through Windows performance counters/process APIs; VRAM
  peak should use both framework allocator peaks and NVML device samples.
- No calibrated whole-system wall-power meter is available. CPU energy and total
  system energy must be marked unavailable, not inferred from TDP.

Primary telemetry reference: NVIDIA NVML device queries
<https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html>.

## Resource constraints for experiments

- Keep prepared dataset plus artifacts below 20 GiB initially; preserve at least
  20 GiB free on D: for checkpoints and failure recovery.
- Target under 7.0 GiB steady VRAM to leave headroom for the display and driver.
- Target under 12 GiB process RAM to avoid paging on a 16 GiB system.
- Run plugged in, with a fixed Windows/NVIDIA power mode, stable thermal state,
  and background GPU applications closed; record rather than assume those states.
