import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omega.backtest import hypothetical_state_response
from omega.config import load_config
from omega.data import synthetic_fx
from omega.errors import ConfigError
from omega.evaluation import walk_forward_splits
from omega.features import build_features
from omega.local_import import ImportSchema, import_history_file
from omega.partitions import overlap_advisory
from omega.providers.oanda import OandaProvider
from omega.state import StageLedger
from omega.validation import DataValidationError, validate


# ---------------------------------------------------------------------------
# Pressure-test regressions: every fix from the adversarial campaign is pinned
# here so a future change that re-introduces the defect fails CI.
# ---------------------------------------------------------------------------


def test_synthetic_fx_supports_tiny_samples():
    for n in (1, 2, 3, 10):
        frame = synthetic_fx(n)
        assert len(frame) == n
        assert frame.timestamp.is_monotonic_increasing


def test_empty_dataframe_is_a_validation_error():
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    with pytest.raises(DataValidationError, match="rows=0"):
        validate(empty)


def test_non_numeric_prices_are_a_validation_error():
    frame = synthetic_fx(50)
    frame.loc[5, "close"] = "not-a-number"
    with pytest.raises(DataValidationError, match="numeric"):
        validate(frame)


def test_zero_window_features_are_rejected():
    with pytest.raises(ValueError, match="positive integers"):
        build_features(synthetic_fx(100), windows=(0, 8))


@pytest.mark.parametrize(
    "evaluation_field,value",
    [
        ("alpha", 1.5),
        ("alpha", -0.1),
        ("abstain_below", 3.0),
        ("calibration_fraction", 1.5),
        ("calibration_fraction", 0.0),
    ],
)
def test_config_rejects_out_of_range_evaluation_values(tmp_path, evaluation_field, value):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
project:
  seed: 1
data:
  timeframe_minutes: 30
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
  {evaluation_field}: {value}
models:
  logistic: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(cfg)


def test_calibration_fraction_of_one_is_rejected():
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        list(walk_forward_splits(6000, 3000, 1000, 1000, 8, 1.0))


def test_train_bars_too_small_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        list(walk_forward_splits(6000, 1, 1000, 1000, 8, 0.2))


def test_backtest_surfaces_missing_returns_instead_of_hiding_them():
    p = np.full(20, 0.9)
    r = np.full(20, 0.001)
    r[3] = np.nan
    _, report = hypothetical_state_response(p, r, holding_bars=4, one_bar_latency=True)
    assert report["missing_returns"] == 1


def test_overlap_advisory_reports_sibling_sources(tmp_path):
    rows = [
        {"Date": "2024-01-02 00:00", "O": 1.10, "H": 1.12, "L": 1.09, "C": 1.11, "S": 0.0001},
        {"Date": "2024-01-02 00:30", "O": 1.11, "H": 1.13, "L": 1.10, "C": 1.12, "S": 0.0001},
    ]
    source_file = tmp_path / "fixture.csv"
    pd.DataFrame(rows).to_csv(source_file, index=False)
    schema = ImportSchema(
        timestamp="Date", open="O", high="H", low="L", close="C", spread="S", timezone="UTC"
    )
    data_root = tmp_path / "data"
    first = import_history_file(source_file, data_root, "src_v1", "EUR_USD", schema)
    second = import_history_file(source_file, data_root, "src_v2", "EUR_USD", schema)
    assert any(item["other_provider"] == "local-src_v1" for item in second["overlap_advisory"])
    assert first["overlap_advisory"] == []


def test_concurrent_ledger_updates_are_serialized(tmp_path):
    ledger = StageLedger(tmp_path / "runs" / "ledger.json", "run-c", stale_after_seconds=5.0)
    failures = []

    def worker(index):
        try:
            with ledger.stage(f"s{index}"):
                time.sleep(0.05)
        except Exception as exc:
            failures.append(type(exc).__name__)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert all(ledger.is_complete(f"s{i}") for i in range(4))


def test_oanda_rejects_empty_and_midless_payloads():
    with pytest.raises(ValueError, match="no complete candles"):
        OandaProvider._normalize({"candles": []})
    with pytest.raises(ValueError, match="midpoint"):
        OandaProvider._normalize({"candles": [{"complete": True, "time": "2024-01-02T00:00:00Z"}]})
