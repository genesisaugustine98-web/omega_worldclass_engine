from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapacityReport:
    root: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    managed_bytes: int
    warning: bool
    retention_candidates: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def inspect_capacity(
    root: str | Path,
    minimum_free_bytes: int = 2_000_000_000,
    candidate_prefixes: tuple[str, ...] = ("cache", "temporary", "derived/old"),
) -> CapacityReport:
    """Report storage pressure and removable candidates without deleting anything."""
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    managed_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    candidates = tuple(
        str(path / prefix)
        for prefix in candidate_prefixes
        if (path / prefix).exists()
    )
    return CapacityReport(
        root=str(path),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        managed_bytes=managed_bytes,
        warning=usage.free < minimum_free_bytes,
        retention_candidates=candidates,
    )


# Raw and manifest paths are intentionally never automatic retention candidates.