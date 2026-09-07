# Licensing

## Recommended model for the current stage

MOLT uses a dual-license business model:

1. the public source is offered under the
   [PolyForm Shield License 1.0.0](../LICENSE); and
2. uses that are not permitted by Shield, or that require negotiated support,
   warranties, indemnity, OEM terms, or alternative redistribution rights,
   require a separate written commercial agreement from the maintainer.

This matches the research-alpha stage: ordinary internal and non-competing use
can evaluate and adopt the engine, while a competing hosted service or training
product cannot rely on the public grant. A noncommercial-only license would
unnecessarily prevent many companies from using MOLT internally. A permissive
OSI license would maximize adoption but would not preserve the intended
competitive boundary.

## Public source license

PolyForm Shield is source-available, not OSI-approved open source. The complete
license text controls; documentation summaries do not modify its terms. The
repository supplies the required copyright notice and defines the licensor's
line of business in `LICENSE`.

PolyForm Shield 1.0.0 does not currently have an SPDX License List identifier.
Package metadata therefore uses the valid custom SPDX expression
`LicenseRef-PolyForm-Shield-1.0.0`. The same expression applies to the wheel and
source distribution. `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and
`TRADEMARKS.md` are declared as package legal files.

Model weights, datasets, CUDA components, PyTorch, Transformers, PEFT,
bitsandbytes, Triton-related components, and other dependencies are separate
works under their own terms. See the [attribution review](../THIRD_PARTY_NOTICES.md).

## Previously published MIT revisions

Repository revisions through commit `a612b40734258b6ee0f1bea5777c7659a352dbee`
and `moltengine` distributions already published as `0.10.0a1` were offered
under MIT. Existing grants are not revoked by relicensing later work. The
historical MIT license is not a grant for current source.

## Commercial inquiries

Contact the maintainer through the
[official GitHub profile](https://github.com/PraveenNimilka) before using current
MOLT source to provide a potentially competing product or service. A commercial
license exists only when both parties execute a separate written agreement.
Obtain qualified legal advice for a binding interpretation.
