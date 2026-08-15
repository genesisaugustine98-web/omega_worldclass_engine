from __future__ import annotations

import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

from ..errors import ProviderError
from .base import HistoricalDataProvider, PartitionRequest, ProviderRateLimiter, provider_get

logger = logging.getLogger("omega.providers.twelvedata")

MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class TwelveDataProvider(HistoricalDataProvider):
    """Twelve Data free-tier FX adapter gated by explicit terms acceptance.

    Twelve Data free tier exposes M1/M5/M15/M30/H1/H2/H4/D/W/M candles for a
    limited instrument set (EUR/USD, USD/JPY, GBP/USD, USD/CHF and a handful
    more) with around 8 credits/minute. This adapter paces requests to stay
    within a free-tier budget, marks coverage as not guaranteed, and never
    claims deep long-horizon coverage.

    API key: ``OMEGA_TWELVEDATA_API_KEY`` (required at runtime, never committed).
    """

    name = "twelvedata"

    SYMBOL_MAP = {
        "EUR_USD": "EUR/USD",
        "USD_JPY": "USD/JPY",
        "GBP_USD": "GBP/USD",
        "USD_CHF": "USD/CHF",
        "EUR_JPY": "EUR/JPY",
        "GBP_JPY": "GBP/JPY",
        "AUD_USD": "AUD/USD",
        "USD_CAD": "USD/CAD",
        "EUR_GBP": "EUR/GBP",
    }

    def __init__(self, terms_accepted: bool, api_key: str | None = None, pacing_seconds: float = 8.0):
        if not terms_accepted:
            raise PermissionError("Set explicit_terms_accepted=true only after reviewing provider terms")
        self.api_key = api_key or os.getenv("OMEGA_TWELVEDATA_API_KEY")
        if not self.api_key:
            raise RuntimeError("OMEGA_TWELVEDATA_API_KEY is required at runtime and must not be committed")
        self.rate_limiter = ProviderRateLimiter(min_interval_seconds=max(0.0, pacing_seconds))
        self.request_timeout_seconds = 60

    def fetch(self, request: PartitionRequest) -> tuple[bytes, pd.DataFrame, dict]:
        if request.instrument not in self.SYMBOL_MAP:
            raise ProviderError(f"Twelve Data has no free FX symbol for {request.instrument}")
        params = {
            "symbol": self.SYMBOL_MAP[request.instrument],
            "interval": "30min",
            "outputsize": "5000",
            "timezone": "UTC",
            "apikey": self.api_key,
        }
        start = request.start.astimezone(timezone.utc)
        end = request.end.astimezone(timezone.utc)
        url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(params)
        self.rate_limiter.wait()
        try:
            raw = provider_get(
                url,
                headers={"User-Agent": "omega-research/1.0"},
                timeout=self.request_timeout_seconds,
            )
        except ProviderError as exc:
            raise ProviderError(f"Twelve Data request failed for {request.key}: {exc}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError(
                f"Twelve Data response for {request.key} exceeds {MAX_RESPONSE_BYTES} bytes; refusing to parse"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Twelve Data returned invalid JSON for {request.key}") from exc
        if payload.get("status") == "error":
            raise ProviderError(
                f"Twelve Data API error for {request.key}: {payload.get('message', 'unknown')}"
            )
        try:
            frame = self._normalize(payload, start, end)
        except (ValueError, KeyError) as exc:
            raise ProviderError(f"Twelve Data payload failed normalization for {request.key}: {exc}") from exc
        metadata = {
            "name": self.name,
            "symbol": self.SYMBOL_MAP[request.instrument],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "url_without_credentials": url.split("&apikey=")[0],
            "license_review_required": True,
            "coverage_not_guaranteed": True,
        }
        return raw, frame, metadata

    @staticmethod
    def _normalize(payload: dict, start: datetime, end: datetime) -> pd.DataFrame:
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError("Twelve Data returned no time series values")
        records = []
        for row in values:
            try:
                records.append(
                    {
                        "timestamp": row["datetime"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row.get("volume") or 0),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"malformed time series row: {row}") from exc
        frame = pd.DataFrame.from_records(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        frame = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)]
        if frame.empty:
            raise ValueError("Provider returned no candles within the requested partition window")
        return frame.reset_index(drop=True)
