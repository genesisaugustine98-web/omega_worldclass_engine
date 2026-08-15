import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omega.config import load_config
from omega.errors import ConfigError, IntegrityError, OmegaError, ProviderError, classify
from omega.local_import import ImportSchema, import_history_file
from omega.partitions import check_dataset_integrity
from omega.pipeline import run_pipeline
from omega.state import StageLedger
from omega.storage import FileStore
from omega.utils import retry


def test_config_schema_rejects_embargo_shorter_than_horizon(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
project:
  seed: 1
data:
  timeframe_minutes: 30
features:
  windows: [8]
labels:
  horizon_bars: 16
  lookback_bars: 48
evaluation:
  train_bars: 100
  test_bars: 50
  step_bars: 50
  embargo_bars: 4
models:
  logistic: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="embargo"):
        load_config(cfg_path)


def test_config_rejects_non_30_minute_bars(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
project:
  seed: 1
data:
  timeframe_minutes: 60
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="30-minute"):
        load_config(cfg_path)


def test_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_env_expansion_supports_plain_and_defaulted_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_TEST_SET", "value-from-env")
    monkeypatch.delenv("OMEGA_TEST_UNSET", raising=False)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
project:
  seed: 1
data:
  timeframe_minutes: 30
  root: ${OMEGA_TEST_SET}/data
  alt: ${OMEGA_TEST_UNSET:-fallback}
features:
  windows: [8]
labels:
  horizon_bars: 4
  lookback_bars: 48
evaluation:
  train_bars: 100
  test_bars: 50
  step_bars: 50
  embargo_bars: 8
models:
  logistic: true
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg["data"]["root"] == "value-from-env/data"
    assert cfg["data"]["alt"] == "fallback"


def test_stale_ledger_lock_is_reclaimed(tmp_path):
    ledger = StageLedger(tmp_path / "run" / "ledger.json", "run-1", stale_after_seconds=1.0)
    ledger.lock_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.lock_path.write_text(json.dumps({"pid": 999999, "host": "crashed"}), encoding="ascii")
    old = time.time() - 300
    os_utime(ledger.lock_path, (old, old))

    with ledger._lock():
        assert ledger.lock_path.exists()
    assert not ledger.lock_path.exists()


def os_utime(path, times):
    import os

    os.utime(path, times)


def test_retry_recovers_from_transient_failures_then_succeeds():
    calls = {"count": 0}

    @retry(attempts=3, base_delay=0.01, max_delay=0.02, exceptions=(ConnectionError,))
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_retry_exhaustion_reraises_last_error():
    @retry(attempts=2, base_delay=0.01, max_delay=0.02, exceptions=(ConnectionError,))
    def always_fails():
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError, match="boom"):
        always_fails()


def test_retry_does_not_catch_unrelated_exceptions():
    calls = {"count": 0}

    @retry(attempts=3, base_delay=0.01, exceptions=(ConnectionError,))
    def fatal():
        calls["count"] += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError, match="not transient"):
        fatal()
    assert calls["count"] == 1


def test_error_taxonomy_classifies_foreign_exceptions():
    assert classify(ValueError("x")).category == "data"
    assert classify(FileNotFoundError("x")).category == "operational"
    assert classify(ConfigError("x")).category == "config"
    assert isinstance(OmegaError("x"), RuntimeError)


def test_cross_partition_duplicate_detected_by_integrity_check(tmp_path):
    store = FileStore(tmp_path / "data")
    jan = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-01-02T00:00:00Z", "2024-02-01T00:00:00Z"], utc=True)}
    )
    feb = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2024-02-01T00:00:00Z", "2024-02-01T00:30:00Z"], utc=True)}
    )
    for month, frame in [("2024/01", jan), ("2024/02", feb)]:
        normalized = store.path(f"normalized/tp/EUR_USD/{month}/bars.parquet")
        normalized.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(normalized, index=False)
        manifest = store.path(f"manifests/tp/EUR_USD/{month}.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="ascii")

    with pytest.raises(IntegrityError, match="more than one"):
        check_dataset_integrity(store, "tp", "EUR_USD")

    report = check_dataset_integrity(store, "tp", "EUR_USD", raise_on_duplicate=False)
    assert report["duplicate_timestamp_count"] == 1
    assert "2024-02-01 00:00:00+00:00" in report["duplicate_timestamps"]


def test_pipeline_resumes_completed_stages_without_refitting(tmp_path, monkeypatch):
    from omega.data import synthetic_fx
    from omega.pipeline import make_model

    fits = {"count": 0}
    original_make_model = make_model

    def counting_make_model(name, seed=42):
        fits["count"] += 1
        return original_make_model(name, seed)

    monkeypatch.setattr("omega.pipeline.make_model", counting_make_model)

    cfg = load_config("config.yaml")
    cfg["models"]["hist_gradient_boosting"] = False
    cfg["evaluation"]["train_bars"] = 3000
    df = synthetic_fx(n=8000, seed=cfg["project"]["seed"])
    artifact_dir = tmp_path / "artifacts"

    first = run_pipeline(df, cfg, artifact_dir)
    first_fits = fits["count"]
    assert first_fits > 0
    assert not first.empty

    second = run_pipeline(df, cfg, artifact_dir)
    assert fits["count"] == first_fits
    assert second.equals(first)


def test_pipeline_fails_loudly_when_no_folds_fit(tmp_path):
    from omega.data import synthetic_fx

    cfg = load_config("config.yaml")
    cfg["evaluation"]["train_bars"] = 50_000
    df = synthetic_fx(n=2000, seed=cfg["project"]["seed"])
    with pytest.raises(ValueError, match="too few"):
        run_pipeline(df, cfg, tmp_path / "artifacts")


def test_import_reports_dataset_integrity(tmp_path):
    rows = [
        {"Date": "2024-01-02 00:00", "O": 1.10, "H": 1.12, "L": 1.09, "C": 1.11, "S": 0.0001},
        {"Date": "2024-01-02 00:30", "O": 1.11, "H": 1.13, "L": 1.10, "C": 1.12, "S": 0.0001},
    ]
    source_file = tmp_path / "fixture.csv"
    pd.DataFrame(rows).to_csv(source_file, index=False)
    schema = ImportSchema(
        timestamp="Date", open="O", high="H", low="L", close="C", spread="S", timezone="UTC"
    )
    result = import_history_file(source_file, tmp_path / "data", "fixture_v1", "EUR_USD", schema)
    assert result["integrity"]["duplicate_timestamp_count"] == 0
    assert result["integrity"]["partition_count"] == 1
    assert result["integrity"]["total_rows"] == 2
