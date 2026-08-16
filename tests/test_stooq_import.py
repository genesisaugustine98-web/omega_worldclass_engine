import pandas as pd
import pytest

from scripts.import_stooq import (
    INSTRUMENT_SYMBOLS,
    fetch_stooq_csv,
    parse_stooq_csv,
    stooq_url,
)

SAMPLE = """Date,Time,Open,High,Low,Close,Volume
2024-01-02,00:00:00,1.10420,1.10440,1.10410,1.10430,0
2024-01-02,00:30:00,1.10430,1.10460,1.10420,1.10450,1000
2024-01-02,01:00:00,1.10450,1.10470,1.10430,1.10460,1500
"""


def test_stooq_url_builds_query_for_interval_and_window():
    url = stooq_url("eurusd", 30, "2024-01-01", "2024-06-01")
    assert "stooq.com" in url
    assert "s=eurusd" in url
    assert "i=30" in url
    assert "d1=20240101" in url
    assert "d2=20240601" in url


def test_stooq_url_rejects_unsupported_interval():
    with pytest.raises(ValueError, match="unsupported"):
        stooq_url("eurusd", 45, "2024-01-01", "2024-06-01")


def test_parse_stooq_csv_combines_date_and_time():
    frame = parse_stooq_csv(SAMPLE)
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert frame["timestamp"].iloc[0] == "2024-01-02 00:00:00"
    assert frame["volume"].iloc[0] == 0
    assert len(frame) == 3


def test_parse_stooq_csv_missing_column_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        parse_stooq_csv("Date,Time,Open,High,Low\n2024-01-02,00:00:00,1.1,1.2,1.0\n")


def test_instrument_symbols_cover_common_majors():
    assert INSTRUMENT_SYMBOLS["EUR_USD"] == "eurusd"
    assert INSTRUMENT_SYMBOLS["USD_JPY"] == "usdjpy"


def test_fetch_detects_bot_wall_html(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return self._payload

    from omega.errors import ProviderError

    html = b"<!DOCTYPE html><html><body>challenge</body></html>"
    calls = []

    def fake_urlopen(request, timeout=30):
        calls.append(request)
        return FakeResponse(html)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.Request", lambda url, headers=None: object())
    with pytest.raises(ProviderError, match="bot-wall"):
        fetch_stooq_csv("https://stooq.com/q/d/l/?s=eurusd&i=30")


def test_fetch_returns_csv_when_not_blocked(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return self._payload

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=30: FakeResponse(SAMPLE.encode()))
    monkeypatch.setattr("urllib.request.Request", lambda url, headers=None: object())
    assert fetch_stooq_csv("https://stooq.com/q/d/l/?s=eurusd&i=30") == SAMPLE


def test_roundtrip_through_import_schema(tmp_path):
    from omega.local_import import ImportSchema, import_history_file

    frame = parse_stooq_csv(SAMPLE)
    path = tmp_path / "stooq.csv"
    frame.to_csv(path, index=False)
    result = import_history_file(
        path,
        data_root=tmp_path / "data",
        source="stooq_test_v1",
        instrument="EUR_USD",
        schema=ImportSchema(
            timestamp="timestamp",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            timezone="UTC",
            timestamp_format="%Y-%m-%d %H:%M:%S",
        ),
    )
    assert result["rows"] == 3
    assert result["partition_count"] == 1
