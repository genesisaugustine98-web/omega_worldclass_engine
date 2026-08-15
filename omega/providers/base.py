from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from ..errors import ProviderError
from ..utils import retry

logger = logging.getLogger("omega.providers")

DEFAULT_TIMEOUT_SECONDS = 60


class ProviderRateLimiter:
    """Minimal pacing so free-tier API quotas are never exceeded.

    ``min_interval_seconds`` spaces requests apart (e.g. 8s for an 8-call/min
    free tier). ``max_per_minute`` additionally throttles bursts. Used only by
    free-tier adapters; pacing is a soft constraint, never a correctness gate.
    """

    def __init__(self, min_interval_seconds: float = 0.0):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_request_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


def provider_get(
    url: str,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = 4,
) -> bytes:
    """Fetch a URL with bounded retry/backoff and structured ProviderError."""

    @retry(
        attempts=attempts,
        base_delay=1.0,
        max_delay=12.0,
        exceptions=(urllib.error.URLError, TimeoutError, ConnectionError),
        on_retry=lambda attempt, delay, exc: logger.warning(
            "Provider transient failure (attempt %d): %s; retrying in %.1fs", attempt, exc, delay
        ),
    )
    def _get() -> bytes:
        message = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(message, timeout=timeout) as response:
            return response.read()

    try:
        return _get()
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"Provider HTTP {exc.code} for {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Provider request failed: {exc.reason}") from exc


def fx_calendar_mask(timestamps: pd.Series) -> pd.Series:
    """Mask of timestamps inside the FX trading calendar.

    The market is closed on Saturday and on Sunday before 20:00 UTC. Some free
    providers (Twelve Data) emit flat, zero-volume bars for the closed period;
    those must not enter a training panel because they are not real trades.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    return ~((ts.dt.dayofweek == 5) | ((ts.dt.dayofweek == 6) & (ts.dt.hour < 20)))


@dataclass(frozen=True)
class PartitionRequest:
    instrument: str
    start: datetime
    end: datetime
    granularity: str = "M30"
    price: str = "MBA"

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Partition boundaries must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("Partition end must be after start")
        if self.granularity != "M30":
            raise ValueError("V1 supports M30 partitions only")

    @property
    def key(self) -> str:
        start = self.start.astimezone(timezone.utc)
        return f"{self.instrument}/{start:%Y/%m}"


class HistoricalDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: PartitionRequest) -> tuple[bytes, pd.DataFrame, dict]:
        """Return immutable raw bytes, normalized bars, and provenance metadata."""