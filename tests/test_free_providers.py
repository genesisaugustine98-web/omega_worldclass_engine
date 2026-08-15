from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import pytest

from omega.acquisition import load_dataset, refresh_dataset
from omega.cloud_config import build_provider, load_cloud_config
from omega.providers.base import HistoricalDataProvider
from omega.providers.polygon import PolygonProvider
from omega.providers.twelvedata import TwelveDataProvider


def _twelvedata_payload(start="2024-01-01 00:00:00"):
    timestamps = pd.date_range(start, periods=3, freq="30min", tz="UTC")
    return {
        "status": "ok",
        "values": [
            {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": "1.10",
                "high": "1.12",
                "low": "1.09",
                "close": "1.11",
                "volume": "100",
            }
            for ts in timestamps
        ],
    }


def _polygon_payload():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return {
        "status": "OK",
        "resultsCount": 2,
        "results": [
            {
                "t": int((base + pd.Timedelta(days=i)).timestamp() * 1000),
                "o": 1.10 + i / 100,
                "h": 1.12 + i / 100,
                "l": 1.09 + i / 100,
                "c": 1.11 + i / 100,
                "v": 1000 + i,
            }
            for i in range(2)
        ],
    }


def test_twelvedata_normalize_filters_and_parses():
    payload = _twelvedata_payload()
    frame = TwelveDataProvider._normalize(payload, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc))
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(frame) == 2
    assert frame["timestamp"].min().tzinfo is not None
    assert frame["close"].dtype == float


def test_twelvedata_normalize_rejects_empty_window():
    with pytest.raises(ValueError, match="no candles"):
        TwelveDataProvider._normalize(
            _twelvedata_payload(),
            datetime(2024, 2, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 1, tzinfo=timezone.utc),
        )


def test_twelvedata_normalize_drops_closed_session_filler():
    timestamps = pd.date_range("2024-01-05 23:00:00", periods=4, freq="30min", tz="UTC")
    payload = {
        "status": "ok",
        "values": [
            {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": "1.10",
                "high": "1.12",
                "low": "1.09",
                "close": "1.11",
                "volume": "100",
            }
            for ts in timestamps
        ],
    }
    frame = TwelveDataProvider._normalize(
        payload,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 8, tzinfo=timezone.utc),
    )
    assert len(frame) == 2
    assert frame["timestamp"].iloc[0].dayofweek == 4
    assert (frame["timestamp"].dt.dayofweek != 5).all()


def test_twelvedata_requires_terms_and_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OMEGA_TWELVEDATA_API_KEY", raising=False)
    with pytest.raises(PermissionError, match="terms"):
        TwelveDataProvider(terms_accepted=False)
    with pytest.raises(RuntimeError, match="OMEGA_TWELVEDATA_API_KEY"):
        TwelveDataProvider(terms_accepted=True)


def test_twelvedata_rejects_unsupported_instrument(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_TWELVEDATA_API_KEY", "dummy")
    provider = TwelveDataProvider(terms_accepted=True, pacing_seconds=0.0)
    from omega.providers.base import PartitionRequest

    request = PartitionRequest("BTC_USD", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc))
    with pytest.raises(Exception, match="no free FX symbol"):
        provider.fetch(request)


def test_polygon_normalize_resamples_to_m30():
    frame = PolygonProvider._normalize(_polygon_payload())
    assert frame["timestamp"].dt.minute.eq(0).all()
    assert frame["timestamp"].dt.second.eq(0).all()
    assert set(frame.columns) == {"timestamp", "open", "high", "low", "close", "volume"}
    assert len(frame) == 2


def test_polygon_requires_terms_and_key(monkeypatch):
    monkeypatch.delenv("OMEGA_POLYGON_API_KEY", raising=False)
    with pytest.raises(PermissionError, match="terms"):
        PolygonProvider(terms_accepted=False)
    with pytest.raises(RuntimeError, match="OMEGA_POLYGON_API_KEY"):
        PolygonProvider(terms_accepted=True)


def test_polygon_intraday_requirement_fails_loudly(monkeypatch):
    monkeypatch.setenv("OMEGA_POLYGON_API_KEY", "dummy")
    provider = PolygonProvider(terms_accepted=True, pacing_seconds=0.0, require_intraday=True)
    from omega.providers.base import PartitionRequest

    request = PartitionRequest("EUR_USD", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc))
    with pytest.raises(Exception, match="require_intraday=false"):
        provider.fetch(request)


def test_build_provider_wires_twelvedata_and_polygon(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_TWELVEDATA_API_KEY", "dummy")
    monkeypatch.setenv("OMEGA_POLYGON_API_KEY", "dummy")
    for provider_name, cls in [("twelvedata", TwelveDataProvider), ("polygon", PolygonProvider)]:
        config = {
            "data_source": {
                "provider": provider_name,
                "explicit_terms_accepted": True,
                "instrument": "EUR_USD",
                "granularity": "M30",
                "price": "MBA",
                "partition": "month",
                "pacing_seconds": 0.0,
            }
        }
        built = build_provider(config)
        assert isinstance(built, cls)
        assert built.name == provider_name


def test_load_cloud_config_accepts_free_providers(tmp_path):
    path = tmp_path / "cloud.yaml"
    path.write_text(
        "data_source:\n"
        "  provider: twelvedata\n"
        "  explicit_terms_accepted: true\n"
        "  instrument: EUR_USD\n"
        "  granularity: M30\n"
        "  price: MBA\n"
        "  partition: month\n"
    )
    config = load_cloud_config(path)
    assert config["data_source"]["provider"] == "twelvedata"


class FakeSequentialProvider(HistoricalDataProvider):
    name = "fake"

    def __init__(self):
        self.calls = []

    def fetch(self, request):
        self.calls.append(request.key)
        timestamps = pd.date_range(request.start, request.end, freq="30min", tz="UTC", inclusive="left")
        ts = pd.Series(timestamps)
        mask = ~((ts.dt.dayofweek == 5) | ((ts.dt.dayofweek == 6) & (ts.dt.hour < 20)))
        timestamps = ts[mask]
        n = len(timestamps)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [1.10] * n,
                "high": [1.12] * n,
                "low": [1.09] * n,
                "close": [1.11] * n,
                "spread": [0.0001] * n,
            }
        )
        return b'{"fixture":true}', frame, {"name": self.name, "retrieved_at": datetime.now(timezone.utc).isoformat()}


CONFIG = {
    "data_source": {
        "provider": "fake",
        "explicit_terms_accepted": True,
        "environment": "practice",
        "instrument": "EUR_USD",
        "granularity": "M30",
        "price": "MBA",
        "partition": "month",
    }
}


def test_refresh_fetches_only_missing_partitions(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OMEGA_RUN_ROOT", str(tmp_path / "runs"))
    provider = FakeSequentialProvider()
    kwargs = {
        "config": CONFIG,
        "project_root": tmp_path,
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-03-01T00:00:00Z",
        "accept_provider_terms": True,
        "provider": provider,
    }
    first = refresh_dataset(**kwargs)
    assert len(first["fetched_partitions"]) == 2
    assert provider.calls == ["EUR_USD/2024/01", "EUR_USD/2024/02"]

    second = refresh_dataset(**kwargs)
    assert second["fetched_partitions"] == []
    assert len(second["already_present"]) == 2
    assert provider.calls == ["EUR_USD/2024/01", "EUR_USD/2024/02"]


def test_refresh_respects_max_partitions(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OMEGA_RUN_ROOT", str(tmp_path / "runs"))
    result = refresh_dataset(
        config=CONFIG,
        project_root=tmp_path,
        start="2024-01-01T00:00:00Z",
        end="2024-04-01T00:00:00Z",
        accept_provider_terms=True,
        max_partitions=1,
        provider=FakeSequentialProvider(),
    )
    assert len(result["fetched_partitions"]) == 1
    assert len(result["skipped_partitions"]) == 2


def test_load_dataset_concatenates_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OMEGA_RUN_ROOT", str(tmp_path / "runs"))
    refresh_dataset(
        config=CONFIG,
        project_root=tmp_path,
        start="2024-01-01T00:00:00Z",
        end="2024-03-01T00:00:00Z",
        accept_provider_terms=True,
        provider=FakeSequentialProvider(),
    )
    panel = load_dataset(config=CONFIG, project_root=tmp_path, start="2024-01-01T00:00:00Z", end="2024-03-01T00:00:00Z")
    assert panel["timestamp"].is_unique
    assert len(panel) > 6
    assert panel["timestamp"].min() == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


def test_load_dataset_raises_when_partitions_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OMEGA_RUN_ROOT", str(tmp_path / "runs"))
    with pytest.raises(Exception, match="missing or incomplete"):
        load_dataset(config=CONFIG, project_root=tmp_path, start="2024-01-01T00:00:00Z", end="2024-03-01T00:00:00Z")


def test_refresh_requires_double_gate(tmp_path):
    with pytest.raises(PermissionError, match="--accept-provider-terms"):
        refresh_dataset(CONFIG, tmp_path, "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z")
    config = deepcopy(CONFIG)
    config["data_source"]["explicit_terms_accepted"] = False
    with pytest.raises(PermissionError, match="explicit_terms_accepted"):
        refresh_dataset(
            config,
            tmp_path,
            "2024-01-01T00:00:00Z",
            "2024-02-01T00:00:00Z",
            accept_provider_terms=True,
        )
