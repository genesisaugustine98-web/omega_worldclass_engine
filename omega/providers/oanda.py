from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from ..errors import ProviderError
from ..utils import retry
from .base import HistoricalDataProvider, PartitionRequest

logger = logging.getLogger("omega.providers.oanda")

MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class OandaProvider(HistoricalDataProvider):
    """OANDA v20 candle adapter gated by explicit terms acceptance.

    OANDA coverage depends on account, instrument, and endpoint behavior. This
    adapter does not claim 20-30 year coverage and must not be used to infer it.
    """

    name = "oanda"

    def __init__(self, terms_accepted: bool, environment: str = "practice", token: str | None = None):
        if not terms_accepted:
            raise PermissionError("Set explicit_terms_accepted=true only after reviewing provider terms")
        if environment not in {"practice", "live"}:
            raise ValueError("OANDA environment must be 'practice' or 'live'")
        self.token = token or os.getenv("OMEGA_OANDA_TOKEN")
        if not self.token:
            raise RuntimeError("OMEGA_OANDA_TOKEN is required at runtime and must not be committed")
        self.environment = environment
        self.request_timeout_seconds = 60

    @property
    def base_url(self) -> str:
        host = "api-fxpractice.oanda.com" if self.environment == "practice" else "api-fxtrade.oanda.com"
        return f"https://{host}/v3"

    def fetch(self, request: PartitionRequest) -> tuple[bytes, pd.DataFrame, dict]:
        params = urllib.parse.urlencode(
            {
                "from": request.start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "to": request.end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "granularity": request.granularity,
                "price": request.price,
            }
        )
        url = f"{self.base_url}/instruments/{request.instrument}/candles?{params}"
        try:
            raw = self._request(url)
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"OANDA HTTP {exc.code} for {request.key}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"OANDA request failed for {request.key}: {exc.reason}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError(
                f"OANDA response for {request.key} exceeds {MAX_RESPONSE_BYTES} bytes; refusing to parse"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"OANDA returned invalid JSON for {request.key}") from exc
        try:
            frame = self._normalize(payload)
        except (ValueError, KeyError) as exc:
            raise ProviderError(f"OANDA payload failed normalization for {request.key}: {exc}") from exc
        metadata = {
            "name": self.name,
            "environment": self.environment,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "url_without_credentials": url,
            "license_review_required": True,
            "coverage_not_guaranteed": True,
        }
        return raw, frame, metadata

    @retry(
        attempts=4,
        base_delay=1.0,
        max_delay=12.0,
        exceptions=(urllib.error.URLError, TimeoutError, ConnectionError),
        on_retry=lambda attempt, delay, exc: logger.warning(
            "OANDA transient failure (attempt %d): %s; retrying in %.1fs", attempt, exc, delay
        ),
    )
    def _request(self, url: str) -> bytes:
        message = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(message, timeout=self.request_timeout_seconds) as response:
            return response.read()

    @staticmethod
    def _normalize(payload: dict) -> pd.DataFrame:
        records = []
        for candle in payload.get("candles", []):
            if not candle.get("complete", False):
                continue
            mid = candle.get("mid", {})
            bid = candle.get("bid", {})
            ask = candle.get("ask", {})
            if not mid:
                raise ValueError("OANDA response omitted midpoint candles")
            spread = None
            if bid and ask:
                spread = float(ask["c"]) - float(bid["c"])
            records.append(
                {
                    "timestamp": candle["time"],
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "spread": spread,
                    "volume": int(candle.get("volume", 0)),
                }
            )
        if not records:
            raise ValueError("Provider returned no complete candles")
        frame = pd.DataFrame.from_records(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        return frame.sort_values("timestamp").reset_index(drop=True)


# CONVERSATION_HOOK: add an approved long-history adapter after terms and actual coverage are verified.