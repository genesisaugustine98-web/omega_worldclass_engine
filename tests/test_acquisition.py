from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import pytest

from omega.acquisition import plan_acquisition, run_acquisition
from omega.providers.base import HistoricalDataProvider
from omega.validation import DataValidationError, validate


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


class FakeProvider(HistoricalDataProvider):
    name = "fake"

    def fetch(self, request):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [1.10, 1.11, 1.12],
                "high": [1.12, 1.13, 1.14],
                "low": [1.09, 1.10, 1.11],
                "close": [1.11, 1.12, 1.13],
                "spread": [0.0001, 0.0001, 0.0001],
            }
        )
        return b'{"fixture":true}', frame, {"name": self.name, "retrieved_at": datetime.now(timezone.utc).isoformat()}


def test_dry_run_plans_exactly_one_partition(tmp_path):
    result = run_acquisition(
        CONFIG,
        tmp_path,
        "2024-01-01T00:00:00Z",
        "2024-02-01T00:00:00Z",
    )
    assert result["mode"] == "dry_run"
    assert result["plan"]["partition_count"] == 1
    assert result["plan"]["request_keys"] == ("EUR_USD/2024/01",)


def test_plan_rejects_partial_months_and_accidental_scale():
    with pytest.raises(ValueError, match="first day"):
        plan_acquisition(CONFIG, "2024-01-02T00:00:00Z", "2024-02-01T00:00:00Z")
    with pytest.raises(ValueError, match="max_partitions=1"):
        plan_acquisition(CONFIG, "2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z")


def test_execution_is_double_gated(tmp_path):
    with pytest.raises(PermissionError, match="--accept-provider-terms"):
        run_acquisition(CONFIG, tmp_path, "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", execute=True)
    config = deepcopy(CONFIG)
    config["data_source"]["explicit_terms_accepted"] = False
    with pytest.raises(PermissionError, match="explicit_terms_accepted"):
        run_acquisition(
            config,
            tmp_path,
            "2024-01-01T00:00:00Z",
            "2024-02-01T00:00:00Z",
            execute=True,
            accept_provider_terms=True,
        )


def test_fake_execution_writes_manifest_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OMEGA_RUN_ROOT", str(tmp_path / "runs"))
    kwargs = {
        "config": CONFIG,
        "project_root": tmp_path,
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-02-01T00:00:00Z",
        "execute": True,
        "accept_provider_terms": True,
        "provider": FakeProvider(),
    }
    first = run_acquisition(**kwargs)
    second = run_acquisition(**kwargs)
    assert first["dataset"]["dataset_root_sha256"] == second["dataset"]["dataset_root_sha256"]
    assert first["manifests"] == second["manifests"]


def test_fx_calendar_allows_sunday_open_but_rejects_saturday():
    base = {
        "open": [1.1],
        "high": [1.2],
        "low": [1.0],
        "close": [1.1],
        "spread": [0.0001],
    }
    sunday = pd.DataFrame({"timestamp": ["2024-01-07T22:00:00Z"], **base})
    assert all(item["passed"] for item in validate(sunday, require_spread=True))
    saturday = pd.DataFrame({"timestamp": ["2024-01-06T22:00:00Z"], **base})
    with pytest.raises(DataValidationError, match="closed_session_rows=1"):
        validate(saturday, require_spread=True)