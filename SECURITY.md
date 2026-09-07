# Security policy

MOLT's supported security surface is the versioned Python package and CLI on
Windows 10/11 with Python 3.12. Alpha releases receive security fixes on the
latest published alpha only.

## Reporting a vulnerability

Do not publish exploitable details in a public issue. Use GitHub's private
security-advisory workflow for this repository. Include the affected version,
reproduction steps, impact, and whether untrusted datasets, configurations or
checkpoints are required.

Please allow the maintainer 7 days to acknowledge a report and 90 days for
coordinated disclosure. Critical issues may require a shorter timeline.

## Trust boundaries

- Treat model weights, token files and configurations as untrusted input.
- MOLT verifies its own checkpoints with SHA-256 metadata and loads them through
  PyTorch's restricted `weights_only=True` loader.
- SHA-256 integrity is not authenticity. Do not open an externally supplied run
  unless its origin is trusted.
- Optional model ecosystems may execute repository-provided model code when
  their own remote-code flags are enabled. MOLT does not enable remote model
  code implicitly.
- Install only from the GitHub release workflow or PyPI project named
  `moltengine`. Release publishing uses short-lived OIDC credentials; reusable
  PyPI tokens are not stored in the repository.
- The repository installer pins its bootstrap URL and verifies the downloaded
  bootstrap script before execution. The lockfile constrains Python dependency
  resolution for source installations.
- GPU power/clock changes are never implicit. `optimize-gpu` requires
  Administrator access and explicit confirmation, applies only a documented
  graphics-clock range, and resets clocks in a `finally` block. A process kill,
  driver crash, or power loss can bypass cleanup; run `nvidia-smi -rgc` after an
  abnormal termination.

Security support does not imply a production safety certification. Operators
remain responsible for model, dataset, dependency, and generated-output risk.
