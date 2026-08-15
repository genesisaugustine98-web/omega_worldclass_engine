from __future__ import annotations

import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

from ..errors import ProviderError
from .base import HistoricalDataProvider, PartitionRequest, ProviderRateLimiter, provider_get

logger = logging.getLogger("omega.providers.polygon")

MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class PolygonProvider(HistoricalDataProvider):
    """Polygon.io free-tier FX adapter gated by explicit terms acceptance.

    Polygon.io ``v2/aggs/ticker`` supports FX tickers like ``C:EURUSD`` with
    arbitrary multiplier/timespan granularity. The free "Basic" plan historically
    offered daily aggregates only; hourly/sub-hourly bars require a paid plan.
    This adapter therefore defaults to daily bars and resamples them to M30,
    which yields monthly partitions but a coarser underlying signal. Set
    ``require_intraday=True`` to fail loudly instead of silently downgrading.

    API key: ``OMEGA_POLYGON_API_KEY`` (required at runtime, never committed).
    """

    name = "polygon"

    SYMBOL_MAP = {
        "EUR_USD": "C:EURUSD",
        "USD_JPY": "C:USDJPY",
        "GBP_USD": "C:GBPUSD",
        "USD_CHF": "C:USDCHF",
        "EUR_JPY": "C:EURJPY",
        "GBP_JPY": "C:GBPJPY",
        "AUD_USD": "C:AUDUSD",
        "USD_CAD": "C:USDCAD",
        "EUR_GBP": "C:EURGBP",
    }

    def __init__(
        self,
        terms_accepted: bool,
        api_key: str | None = None,
        pacing_seconds: float = 12.0,
        require_intraday: bool = True,
    ):
        if not terms_accepted:
            raise PermissionError("Set explicit_terms_accepted=true only after reviewing provider terms")
        self.api_key = api_key or os.getenv("OMEGA_POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError("OMEGA_POLYGON_API_KEY is required at runtime and must not be committed")
        self.require_intraday = require_intraday
        self.rate_limiter = ProviderRateLimiter(min_interval_seconds=max(0.0, pacing_seconds))
        self.request_timeout_seconds = 60

    def fetch(self, request: PartitionRequest) -> tuple[bytes, pd.DataFrame, dict]:
        if request.instrument not in self.SYMBOL_MAP:
            raise ProviderError(f"Polygon has no free FX ticker for {request.instrument}")
        if self.require_intraday:
            raise ProviderError(
                "Polygon free tier lacks hourly/sub-hourly FX aggregates; set require_intraday=false to use daily bars"
            )
        start = request.start.astimezone(timezone.utc)
        end = request.end.astimezone(timezone.utc)
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self.api_key,
        }
        url = (
            "https://api.polygon.io/v2/aggs/ticker/"
            f"{urllib.parse.quote(self.SYMBOL_MAP[request.instrument], safe=':')}/range/"
            f"1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}?"
            + urllib.parse.urlencode(params)
        )
        self.rate_limiter.wait()
        try:
            raw = provider_get(
                url,
                headers={"User-Agent": "omega-research/1.0"},
                timeout=self.request_timeout_seconds,
            )
        except ProviderError as exc:
            raise ProviderError(f"Polygon request failed for {request.key}: {exc}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError(
                f"Polygon response for {request.key} exceeds {MAX_RESPONSE_BYTES} bytes; refusing to parse"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Polygon returned invalid JSON for {request.key}") from exc
        if payload.get("status") == "ERROR":
            raise ProviderError(
                f"Polygon API error for {request.key}: {payload.get('error', 'unknown')}"
            )
        try:
            frame = self._normalize(payload)
        except (ValueError, KeyError) as exc:
            raise ProviderError(f"Polygon payload failed normalization for {request.key}: {exc}") from exc
        metadata = {
            "name": self.name,
            "ticker": self.SYMBOL_MAP[request.instrument],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "url_without_credentials": url.split("&apiKey=")[0],
            "license_review_required": True,
            "coverage_not_guaranteed": True,
        }
        return raw, frame, metadata

    @staticmethod
    def _normalize(payload: dict) -> pd.DataFrame:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("Polygon returned no aggregate results")
        records = []
        for row in results:
            try:
                records.append(
                    {
                        "timestamp": datetime.fromtimestamp(row["t"] / 1000.0, tz=timezone.utc),
                        "open": float(row["o"]),
                        "high": float(row["h"]),
                        "low": float(row["l"]),
                        "close": float(row["c"]),
                        "volume": int(row.get("v") or 0),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"malformed aggregate row: {row}") from exc
        frame = pd.DataFrame.from_records(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame = frame.set_index("timestamp").resample("30min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        frame = frame.dropna(subset=["close"]).reset_index()
        if frame.empty:
            raise ValueError("Provider returned no resampled candles")
        return frame