# Security policy

MOLT is research software. The current supported security surface is the
versioned Python package and CLI on Windows 11 with Python 3.12.

## Reporting a vulnerability

Do not publish exploitable details in a public issue. Use GitHub's private
security-advisory workflow for this repository. Include the affected version,
reproduction steps, impact, and whether untrusted datasets, configurations or
checkpoints are required.

## Trust boundaries

- Treat model weights, token files and configurations as untrusted input.
- MOLT verifies its own checkpoints with SHA-256 metadata and loads them through
  PyTorch's restricted `weights_only=True` loader.
- SHA-256 integrity is not authenticity. Do not open an externally supplied run
  unless its origin is trusted.
- Optional model ecosystems may execute repository-provided model code when
  their own remote-code flags are enabled. MOLT does not enable remote model
  code implicitly.
- GPU power/clock changes are never implicit. `optimize-gpu` requires
  Administrator access and explicit confirmation, applies only a documented
  graphics-clock range, and resets clocks in a `finally` block. A process kill,
  driver crash, or power loss can bypass cleanup; run `nvidia-smi -rgc` after an
  abnormal termination.

Security support does not imply a production safety certification.
