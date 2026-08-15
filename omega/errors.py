from __future__ import annotations

"""Structured error taxonomy for the OMEGA engine.

Every failure the engine can produce belongs to one of these categories so
callers can distinguish transient, permanent, and configuration problems and
react appropriately: retry, stop, or fail fast with an actionable message.
"""


class OmegaError(RuntimeError):
    """Base class for all OMEGA engine errors."""

    category = "omega"


class ConfigError(OmegaError):
    """Configuration is missing, malformed, or violates an invariant."""

    category = "config"


class DataError(OmegaError):
    """Input data fails validation or cannot be parsed."""

    category = "data"


class IntegrityError(OmegaError):
    """Storage/provenance invariant broken: immutable conflict, stale lock,
    cross-partition overlap, or a ledger that contradicts manifests."""

    category = "integrity"


class ProviderError(OmegaError):
    """A data provider failed after bounded retries, or returned an invalid
    payload. Transient conditions must be retried before this is raised."""

    category = "provider"


class OperationalError(OmegaError):
    """Runtime infrastructure failure (storage, filesystem, platform)."""

    category = "operational"


class ResourceError(OmegaError):
    """Resource limits exceeded: storage, memory, or runtime budget."""

    category = "resource"


def classify(exc: Exception) -> OmegaError:
    """Wrap an arbitrary exception in the taxonomy, preserving its message."""
    if isinstance(exc, OmegaError):
        return exc
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return OperationalError(str(exc))
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return DataError(str(exc))
    return OperationalError(str(exc))
