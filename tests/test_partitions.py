import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from omega.partitions import PartitionOrchestrator, dataset_manifest
from omega.providers.base import HistoricalDataProvider, PartitionRequest
from omega.providers.oanda import OandaProvider
from omega.state import StageLedger
from omega.storage import FileStore


class FakeProvider(HistoricalDataProvider):
    name = "fake"

    def __init__(self):
        self.calls = 0

    def fetch(self, request):
        self.calls += 1
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC"),
                "open": [1.1, 1.1, 1.2, 1.2],
                "high": [1.2, 1.2, 1.3, 1.3],
                "low": [1.0, 1.0, 1.1, 1.1],
                "close": [1.1, 1.2, 1.2, 1.25],
                "spread": [0.0001] * 4,
            }
        )
        raw = json.dumps({"partition": request.key, "rows": len(frame)}).encode("utf-8")
        return raw, frame, {"name": self.name, "retrieved_at": "2024-02-01T00:00:00+00:00"}


def request():
    return PartitionRequest(
        instrument="EUR_USD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )


def test_provider_is_gated_by_terms_and_runtime_secret(monkeypatch):
    monkeypatch.delenv("OMEGA_OANDA_TOKEN", raising=False)
    with pytest.raises(PermissionError, match="terms"):
        OandaProvider(terms_accepted=False)
    with pytest.raises(RuntimeError, match="OMEGA_OANDA_TOKEN"):
        OandaProvider(terms_accepted=True)


def test_oanda_normalization_keeps_only_complete_candles():
    payload = {
        "candles": [
            {
                "complete": True,
                "time": "2024-01-02T00:00:00Z",
                "volume": 7,
                "mid": {"o": "1.10", "h": "1.12", "l": "1.09", "c": "1.11"},
                "bid": {"c": "1.1099"},
                "ask": {"c": "1.1101"},
            },
            {"complete": False, "time": "2024-01-02T00:30:00Z"},
        ]
    }
    frame = OandaProvider._normalize(payload)
    assert len(frame) == 1
    assert frame.loc[0, "spread"] == pytest.approx(0.0002)
    assert str(frame.timestamp.dt.tz) == "UTC"


def test_partition_acquisition_is_manifested_and_resumable(tmp_path):
    store = FileStore(tmp_path / "data")
    ledger = StageLedger(tmp_path / "runs" / "ledger.json", "run-1")
    provider = FakeProvider()
    orchestrator = PartitionOrchestrator(store, ledger, provider)

    first = orchestrator.acquire(request())
    second = orchestrator.acquire(request())

    assert first == second
    assert all(isinstance(check["passed"], bool) for check in second["validation"])
    assert provider.calls == 1
    assert first["raw"]["sha256"]
    assert first["normalized"]["rows"] == 4
    assert all(check["passed"] for check in first["validation"])
    assert store.exists("manifests/fake/EUR_USD/2024/01.json")

    dataset = dataset_manifest(store, "fake", "EUR_USD")
    assert dataset["partition_count"] == 1
    assert dataset["dataset_root_sha256"]


def test_content_addressed_raw_write_allows_identical_retry(tmp_path):
    store = FileStore(tmp_path)
    source = tmp_path / "response.json"
    source.write_text('{"ok": true}', encoding="ascii")
    first = store.put_immutable(source, "raw/hash/response.json")
    second = store.put_immutable(source, "raw/hash/response.json")
    assert first == second

    source.write_text('{"ok": false}', encoding="ascii")
    with pytest.raises(FileExistsError, match="different content"):
        store.put_immutable(source, "raw/hash/response.json")