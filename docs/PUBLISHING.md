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
   when maintainer signing is configured.
5. Create a GitHub Release from that tag. Publishing the release starts
   `.github/workflows/publish.yml`.
6. Verify the GitHub `publish` workflow, PyPI file hashes, and a clean-environment
   installation before announcing the release.

For `0.11.0a1`, the workflow may instead be started once from
**Actions → publish → Run workflow** after the trusted publisher is verified.
PyPI versions are immutable: never reuse a version after it has been uploaded.

## Local distribution verification

```powershell
uv run --frozen python -m pytest -q
uv build
uvx twine check dist/*
```

CUDA PyTorch uses its own package index. End users should install the supported
CUDA PyTorch wheel first, then install MOLT from PyPI as documented in the README.
