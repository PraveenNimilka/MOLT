class MoltStreamError(RuntimeError):
    """Base error for explicit, non-fallback failures."""


class CapabilityError(MoltStreamError):
    """The requested mechanism is unavailable on the current runtime."""


class IntegrityError(MoltStreamError):
    """Persistent state failed an integrity invariant."""
