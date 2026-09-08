# Publishing MOLT

MOLT publishes Python distributions through PyPI Trusted Publishing. GitHub
Actions exchanges a short-lived OpenID Connect identity with PyPI; no reusable
PyPI API token is stored in the repository or in GitHub secrets.

## PyPI trusted-publisher configuration

The `moltengine` project already exists on PyPI. Its trusted publisher must keep
these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `moltengine` |
| GitHub owner | `PraveenNimilka` |
| Repository | `MOLT` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Manage this binding from the PyPI project's publishing settings. Do not add an
API token to the repository or GitHub Actions configuration.

## Release procedure

1. Confirm `main` is clean and CI is green.
2. Update the version in `pyproject.toml`, release notes, and user-facing version
   constants in one release commit.
3. Run `uv run --frozen python -m pytest -q` and `uv build` locally.
4. Create and push an annotated release tag; use a cryptographically signed tag
   when maintainer signing is configured. A `v*` tag starts
   `.github/workflows/publish.yml`.
5. Create a GitHub Release from the verified tag.
6. Verify the GitHub `publish` workflow, PyPI file hashes, and a clean-environment
   installation before announcing the release.

The workflow may instead be started from **Actions → publish → Run workflow**
after the trusted publisher is verified.
PyPI versions are immutable: never reuse a version after it has been uploaded.

## Local distribution verification

```powershell
uv run --frozen python -m pytest -q
uv build
uvx twine check dist/*
```

The base package can be installed directly from PyPI. End users then run
`molt setup`, which installs and validates the supported CUDA PyTorch, QLoRA,
data, monitoring, and optimized-kernel runtime documented in the README.
