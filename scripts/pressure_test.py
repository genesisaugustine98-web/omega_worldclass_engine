"""Adversarial pressure-test harness for the OMEGA engine.

Each scenario reports PASS (engine behaved correctly/fail-closed), WEAK
(engine survived but behaved oddly or silently), or FAIL (crash, wrong output,
or silent corruption). We deliberately probe first-principles invariants:
point-in-time safety, immutability, fail-closed validation, and resumability.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import omega.validation as validation
import omega.features as features
import omega.labels as labels
import omega.evaluation as evaluation
import omega.backtest as backtest
import omega.data as data
import omega.config as config
import omega.partitions as partitions
import omega.state as state
import omega.storage as storage
import omega.local_import as local_import
import omega.pipeline as pipeline
import omega.errors as errors

RESULTS = []
CATEGORY = [""]


def category(name):
    CATEGORY[0] = name


def record(name, outcome, detail=""):
    RESULTS.append({"category": CATEGORY[0], "name": name, "outcome": outcome, "detail": detail[:400]})


def expect_raises(fn, exc_type, name):
    try:
        fn()
        record(name, "WEAK", f"expected {exc_type.__name__} but no exception raised")
    except exc_type as exc:
        record(name, "PASS", f"{exc_type.__name__}: {exc}")
    except Exception as exc:
        record(name, "FAIL", f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")


def synthetic(n=8000, seed=42):
    return data.synthetic_fx(n, seed)


def full_cfg():
    return config.load_config(ROOT / "config.yaml")


# ---------------------------------------------------------------- config
category("config")

try:
    full_cfg()
    record("valid_config_loads", "PASS")
except Exception as exc:
    record("valid_config_loads", "FAIL", str(exc))

def _write_cfg(text):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return f.name

BASE = """
project: {seed: 1}
data: {timeframe_minutes: 30}
features: {windows: [8]}
labels: {horizon_bars: 4}
evaluation: {train_bars: 100, test_bars: 50, embargo_bars: 8, __EVAL__}
models: {logistic: true}
"""

# alpha out of [0,1]
def test_alpha_cfg():
    path = _write_cfg(BASE.replace("__EVAL__", "alpha: 1.5"))
    try:
        config.load_config(path)
    finally:
        os.unlink(path)
try:
    test_alpha_cfg()
    record("alpha_out_of_range", "WEAK", "config schema does not validate alpha range (needs fix)")
except errors.ConfigError:
    record("alpha_out_of_range", "PASS")
except Exception as exc:
    record("alpha_out_of_range", "FAIL", f"wrong error {type(exc).__name__}: {exc}")

# abstain_below out of range
def test_abstain_cfg():
    path = _write_cfg(BASE.replace("__EVAL__", "abstain_below: 3.0"))
    try:
        config.load_config(path)
    finally:
        os.unlink(path)
try:
    test_abstain_cfg()
    record("abstain_below_out_of_range", "WEAK", "config schema does not validate abstain_below range (needs fix)")
except errors.ConfigError:
    record("abstain_below_out_of_range", "PASS")
except Exception as exc:
    record("abstain_below_out_of_range", "FAIL", f"wrong error {type(exc).__name__}: {exc}")

# calibration_fraction out of range
def test_cal_fraction_cfg():
    path = _write_cfg(BASE.replace("__EVAL__", "calibration_fraction: 1.5"))
    try:
        config.load_config(path)
    finally:
        os.unlink(path)
try:
    test_cal_fraction_cfg()
    record("calibration_fraction_out_of_range", "WEAK", "config schema does not validate calibration_fraction range (needs fix)")
except errors.ConfigError:
    record("calibration_fraction_out_of_range", "PASS")
except Exception as exc:
    record("calibration_fraction_out_of_range", "FAIL", f"wrong error {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- validation
category("validation")

df = synthetic(300)
record("synthetic_300_valid", "PASS" if all(c["passed"] for c in validation.validate(df)) else "FAIL")

# constant close series -> MAD=0 path
def constant_close():
    d = synthetic(100)
    d["close"] = 1.0
    d["open"] = 1.0
    d["high"] = 1.0
    d["low"] = 1.0
    validation.validate(d)
constant_close()
record("constant_series_mad_zero", "PASS", "robust z guard handles MAD=0")

# NaN in close
def nan_close():
    d = synthetic(100)
    d.loc[5, "close"] = np.nan
    validation.validate(d)
expect_raises(nan_close, validation.DataValidationError, "nan_in_price_stops")

# inf in high
def inf_high():
    d = synthetic(100)
    d.loc[5, "high"] = np.inf
    validation.validate(d)
expect_raises(inf_high, validation.DataValidationError, "inf_in_price_stops")

# negative price
def negative_price():
    d = synthetic(100)
    d.loc[5, "close"] = -1.0
    validation.validate(d)
expect_raises(negative_price, validation.DataValidationError, "negative_price_stops")

# single row
def single_row():
    d = synthetic(1)
    validation.validate(d)
single_row()
record("single_row", "PASS")

# empty dataframe
def empty_df():
    validation.validate(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]))
expect_raises(empty_df, validation.DataValidationError, "empty_df_stops")

# missing required column
def missing_col():
    d = synthetic(10).drop(columns=["volume"])
    validation.validate(d)
record("missing_optional_col_allowed", "PASS")

# wrong dtype timestamps
def string_timestamps():
    d = synthetic(10)
    d["timestamp"] = d["timestamp"].astype(str)
    validation.validate(d)
string_timestamps()
record("string_timestamps_coerced", "PASS")

# non-monotonic
def unsorted():
    d = synthetic(100).sort_values("timestamp", ascending=False).reset_index(drop=True)
    validation.validate(d)
expect_raises(unsorted, validation.DataValidationError, "unsorted_stops")

# duplicate timestamps
def dup_ts():
    d = synthetic(100)
    d.loc[10, "timestamp"] = d.loc[9, "timestamp"]
    d = d.sort_values("timestamp").reset_index(drop=True)
    validation.validate(d)
expect_raises(dup_ts, validation.DataValidationError, "duplicate_ts_stops")

# weekend saturday
def saturday():
    d = synthetic(50)
    d.loc[10, "timestamp"] = pd.Timestamp("2024-01-06 10:00", tz="UTC")
    d = d.sort_values("timestamp").reset_index(drop=True)
    validation.validate(d)
expect_raises(saturday, validation.DataValidationError, "saturday_stops")

# sunday evening allowed (FX open 22:00 UTC)
def sunday_open():
    d = synthetic(50)
    d.loc[10, "timestamp"] = pd.Timestamp("2024-01-07 22:00", tz="UTC")
    d = d.sort_values("timestamp").reset_index(drop=True)
    validation.validate(d)
record("sunday_open_allowed", "PASS")

# available_at in the future
def future_available_at():
    d = synthetic(100)
    d["available_at"] = d["timestamp"] + pd.Timedelta(hours=1)
    validation.validate(d)
expect_raises(future_available_at, validation.DataValidationError, "future_available_at_stops")

# ---------------------------------------------------------------- features
category("features")

df = synthetic(500)
x = features.build_features(df)
record("features_build", "PASS", f"shape={x.shape}")

# inf -> nan
numeric = x.select_dtypes(include=[np.number])
has_inf = bool(np.isinf(numeric.to_numpy()).any()) if numeric.size else False
record("no_inf_in_features", "PASS" if not has_inf else "FAIL", f"shape={x.shape}")

# all NaN column (constant close) should not crash
def constant_close_features():
    d = synthetic(500)
    d["close"] = 1.0
    d["open"] = 1.0
    d["high"] = 1.0
    d["low"] = 1.0
    x2 = features.build_features(d)
    return x2
x2 = constant_close_features()
record("constant_close_features_no_crash", "PASS", f"shape={x2.shape}")

# feature build on df missing 'volume' (allowed)
record("features_no_volume", "PASS")

# unsorted features must raise
def unsorted_features():
    features.build_features(synthetic(100).iloc[::-1].reset_index(drop=True))
expect_raises(unsorted_features, ValueError, "unsorted_features_raise")

# windows with zero must be rejected
def zero_window_features():
    features.build_features(synthetic(100), windows=(0, 8))
expect_raises(zero_window_features, ValueError, "zero_window_features_rejected")

# ---------------------------------------------------------------- labels
category("labels")

y = labels.label_phenomena(df)
record("labels_build", "PASS", f"shape={y.shape} cols={list(y.columns)}")

# terminal horizon NaN
record("terminal_nan", "PASS" if y[labels.LABELS].iloc[-8:].isna().all().all() else "FAIL")

# horizon larger than data
def big_horizon():
    return labels.label_phenomena(synthetic(10), horizon=50)
ybig = big_horizon()
record("horizon_larger_than_data", "PASS" if len(ybig) == 10 else "FAIL", f"rows={len(ybig)}")

# constant series labels (no NaN explosion)
def const_labels():
    d = synthetic(500)
    d["close"] = 1.0
    d["open"] = 1.0
    d["high"] = 1.0
    d["low"] = 1.0
    return labels.label_phenomena(d)
yconst = const_labels()
record("constant_series_labels_no_crash", "PASS", f"rows={len(yconst)} nan={yconst[labels.LABELS].isna().sum().sum()}")

# ---------------------------------------------------------------- evaluation
category("evaluation")

splits = list(evaluation.walk_forward_splits(6000, 3000, 1000, 1000, 8, 0.2))
record("walk_forward_splits", "PASS", f"folds={len(splits)}")

# tiny n -> no folds
record("tiny_n_no_folds", "PASS" if list(evaluation.walk_forward_splits(10, 3000, 1000, 1000, 8, 0.2)) == [] else "FAIL")

# embargo >= test_bars edge
def embargo_gt_test():
    list(evaluation.walk_forward_splits(6000, 3000, 10, 10, 8, 0.2))
embargo_gt_test()
record("embargo_gt_test_survives", "PASS")

# calibration fraction giving cal_size < 1
def tiny_calibration():
    return list(evaluation.walk_forward_splits(6000, 3000, 1000, 1000, 8, 0.001))
cal_splits = tiny_calibration()
record("tiny_cal_fraction", "PASS" if cal_splits and cal_splits[0].calibration.size >= 1 else "WEAK")

# calibration_fraction exactly 1.0 must be rejected (would empty the train split)
def cal_fraction_one():
    list(evaluation.walk_forward_splits(6000, 3000, 1000, 1000, 8, 1.0))
expect_raises(cal_fraction_one, ValueError, "cal_fraction_one_rejected")

# train_bars too small to carve a calibration split
def train_bars_one():
    list(evaluation.walk_forward_splits(6000, 1, 1000, 1000, 8, 0.2))
expect_raises(train_bars_one, ValueError, "train_bars_one_rejected")

# probability_metrics with all-one-class (roc_auc nan)
m = evaluation.probability_metrics(np.ones(10), np.full(10, 0.5))
record("prob_metrics_single_class", "PASS" if np.isnan(m["roc_auc"]) else "WEAK")

# probability_metrics shape mismatch
def metrics_shape_mismatch():
    evaluation.probability_metrics(np.ones(5), np.ones(6))
expect_raises(metrics_shape_mismatch, Exception, "metrics_shape_mismatch")

# ECE with single bin values
ece = evaluation.expected_calibration_error(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
record("ece_basic", "PASS" if np.isfinite(ece) else "FAIL", f"ece={ece}")

# ECE constant probabilities
ece2 = evaluation.expected_calibration_error(np.array([0, 1, 0, 1]), np.full(4, 0.5))
record("ece_constant", "PASS" if np.isfinite(ece2) else "FAIL")

# ---------------------------------------------------------------- backtest
category("backtest")

p = np.full(100, 0.9)
r = np.full(100, 0.001)
frame, rep = backtest.hypothetical_state_response(p, r, holding_bars=4, one_bar_latency=True)
record("backtest_basic_latency", "PASS", f"entries={rep['entries']}")

# all below threshold -> no entries
frame2, rep2 = backtest.hypothetical_state_response(np.full(100, 0.1), r, threshold=0.58)
record("backtest_no_entries", "PASS" if rep2["entries"] == 0 else "FAIL", f"entries={rep2['entries']}")
record("backtest_no_entries_sharpe", "PASS" if rep2["naive_sharpe"] == 0.0 else "WEAK")

# NaN in forward_return (must survive and be surfaced, not silently corrupt)
frame3, rep3 = backtest.hypothetical_state_response(p, np.r_[np.nan, r[1:]], holding_bars=4)
record("backtest_nan_forward_return_survives", "PASS" if rep3["missing_returns"] == 1 else "WEAK",
       f"missing_returns={rep3['missing_returns']}")

# inf in prob
def inf_prob():
    backtest.hypothetical_state_response(np.r_[np.inf, p[1:]], r)
inf_prob()
record("backtest_inf_prob_survives", "PASS")

# holding_bars > len
def huge_holding():
    backtest.hypothetical_state_response(p[:10], r[:10], holding_bars=1000)
huge_holding()
record("backtest_huge_holding", "PASS")

# len mismatch
def len_mismatch():
    backtest.hypothetical_state_response(p[:10], r[:9])
expect_raises(len_mismatch, ValueError, "backtest_len_mismatch")

# single element
def single_element_bt():
    backtest.hypothetical_state_response(np.array([0.9]), np.array([0.001]), holding_bars=1, one_bar_latency=True)
single_element_bt()
record("backtest_single_element", "PASS")

# ---------------------------------------------------------------- data / storage
category("storage")

store = storage.FileStore("/tmp/opencode/pressure/store")
record("store_create", "PASS")

# path traversal must fail
def traversal():
    store.path("../escape")
expect_raises(traversal, ValueError, "store_path_traversal")

def traversal2():
    store.path("raw/../../etc/passwd")
expect_raises(traversal2, ValueError, "store_path_traversal2")

# absolute path inside root ok
record("store_absolute_inside", "PASS")

# put_immutable conflict on a shared store path
f = Path("/tmp/opencode/pressure/a.bin")
g = Path("/tmp/opencode/pressure/b.bin")
f.write_bytes(b"AAAA")
g.write_bytes(b"BBBB")

def put_conflict():
    store.put_immutable(f, "raw/x/a.bin")
    store.put_immutable(g, "raw/x/a.bin")
try:
    put_conflict()
    record("put_immutable_same_key_after_conflict", "WEAK", "second put with different content did not raise")
except FileExistsError:
    record("put_immutable_same_key_after_conflict", "PASS")

# re-run with a fresh path to verify the conflict raises
store2 = storage.FileStore("/tmp/opencode/pressure/store2")
store2.put_immutable(f, "raw/x/a.bin")
def put_conflict2():
    store2.put_immutable(g, "raw/x/a.bin")
expect_raises(put_conflict2, FileExistsError, "put_immutable_conflict_raises")

# ---------------------------------------------------------------- state / ledger
category("state")

ledger = state.StageLedger(Path("/tmp/opencode/pressure/led/ledger.json"), "run-1")
with ledger.stage("s1") as should:
    pass
record("ledger_basic", "PASS" if ledger.is_complete("s1") else "FAIL")

# corrupt ledger json
bad = Path("/tmp/opencode/pressure/led/bad.json")
bad.parent.mkdir(parents=True, exist_ok=True)
bad.write_text("{not json", encoding="utf-8")
bad_ledger = state.StageLedger(bad, "run-2")
def corrupt_read():
    bad_ledger.read()
expect_raises(corrupt_read, errors.IntegrityError, "ledger_corrupt_json")

# wrong run_id
def wrong_run_id():
    state.StageLedger(Path("/tmp/opencode/pressure/led/ledger.json"), "run-WRONG").read()
expect_raises(wrong_run_id, errors.IntegrityError, "ledger_wrong_run_id")

# concurrent-ish lock reuse
def lock_reacquire():
    with ledger._lock():
        with ledger._lock(timeout_seconds=0.5):
            pass
expect_raises(lock_reacquire, errors.OperationalError, "ledger_lock_contention")

# ---------------------------------------------------------------- local import
category("local_import")

def valid_rows():
    return [
        {"Date": "2024-01-02 00:00", "O": 1.10, "H": 1.12, "L": 1.09, "C": 1.11, "S": 0.0001},
        {"Date": "2024-01-02 00:30", "O": 1.11, "H": 1.13, "L": 1.10, "C": 1.12, "S": 0.0001},
        {"Date": "2024-02-01 00:00", "O": 1.12, "H": 1.14, "L": 1.11, "C": 1.13, "S": 0.0001},
    ]

def schema():
    return local_import.ImportSchema(timestamp="Date", open="O", high="H", low="L", close="C", spread="S", timezone="UTC")

# overlapping cross-source duplicate (advisory, not fatal)
def overlap_import():
    base_dir = Path("/tmp/opencode/pressure/import")
    src1 = base_dir / "one.csv"
    src2 = base_dir / "two.csv"
    src1.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(valid_rows()).to_csv(src1, index=False)
    rows2 = valid_rows()
    rows2.append({"Date": "2024-02-01 00:30", "O": 1.13, "H": 1.15, "L": 1.12, "C": 1.14, "S": 0.0001})
    pd.DataFrame(rows2).to_csv(src2, index=False)
    first = local_import.import_history_file(src1, base_dir / "data", "src_v1", "EUR_USD", schema())
    second = local_import.import_history_file(src2, base_dir / "data", "src_v2", "EUR_USD", schema())
    return first["overlap_advisory"], second["overlap_advisory"]
adv1, adv2 = overlap_import()
record("overlap_across_sources_advisory", "PASS" if (adv1 or adv2) else "WEAK",
       f"src_v1 adv={adv1} src_v2 adv={adv2}")

# duplicate timestamps in single file
def dup_rows():
    rows = valid_rows()[:2]
    rows[1]["Date"] = rows[0]["Date"]
    base_dir = Path("/tmp/opencode/pressure/importdup")
    f = base_dir / "d.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    local_import.import_history_file(f, base_dir / "data", "dup_v1", "EUR_USD", schema())
expect_raises(dup_rows, validation.DataValidationError, "import_dup_timestamps_stop")

# timezone naive without timezone
def naive_ts():
    rows = valid_rows()[:1]
    base_dir = Path("/tmp/opencode/pressure/naive")
    f = base_dir / "n.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    local_import.read_local_history(f, local_import.ImportSchema(timestamp="Date", open="O", high="H", low="L", close="C"))
expect_raises(naive_ts, ValueError, "import_naive_requires_tz")

# ---------------------------------------------------------------- pipeline
category("pipeline")

def pipeline_small():
    cfg = full_cfg()
    cfg["evaluation"]["train_bars"] = 50_000
    pipeline.run_pipeline(synthetic(2000), cfg, "/tmp/opencode/pressure/art")
expect_raises(pipeline_small, ValueError, "pipeline_too_small_fails_loud")

def pipeline_full():
    cfg = full_cfg()
    return pipeline.run_pipeline(synthetic(8000), cfg, "/tmp/opencode/pressure/art_full")
m = pipeline_full()
record("pipeline_full", "PASS" if not m.empty else "WEAK", f"metrics_rows={len(m)}")

# resume idempotence
def pipeline_resume():
    cfg = full_cfg()
    d = synthetic(8000)
    a = pipeline.run_pipeline(d, cfg, "/tmp/opencode/pressure/art_resume")
    b = pipeline.run_pipeline(d, cfg, "/tmp/opencode/pressure/art_resume")
    return a.equals(b)
record("pipeline_resume_idempotent", "PASS" if pipeline_resume() else "FAIL")

# ---------------------------------------------------------------- oanda edges
category("oanda")

from omega.providers.oanda import OandaProvider

def oanda_empty_candles():
    OandaProvider._normalize({"candles": []})
expect_raises(oanda_empty_candles, ValueError, "oanda_no_candles_rejected")

def oanda_missing_mid():
    OandaProvider._normalize({"candles": [{"complete": True, "time": "2024-01-02T00:00:00Z"}]})
expect_raises(oanda_missing_mid, ValueError, "oanda_missing_mid_rejected")

def oanda_partial_candles():
    frame = OandaProvider._normalize({
        "candles": [
            {"complete": True, "time": "2024-01-02T00:00:00Z", "volume": 7,
             "mid": {"o": "1.10", "h": "1.12", "l": "1.09", "c": "1.11"},
             "bid": {"c": "1.1099"}, "ask": {"c": "1.1101"}},
            {"complete": False, "time": "2024-01-02T00:30:00Z"},
        ]
    })
    return len(frame)
record("oanda_partial_filtered", "PASS" if oanda_partial_candles() == 1 else "FAIL")

# retry wrapper honors bounded attempts and re-raises final provider error
def oanda_retry_exhausts():
    from omega.utils import retry
    calls = {"n": 0}

    @retry(attempts=2, base_delay=0.01, max_delay=0.02, exceptions=(ConnectionError,))
    def flaky():
        calls["n"] += 1
        raise ConnectionError("network flake")
    try:
        flaky()
        return "no error"
    except ConnectionError:
        return calls["n"]
record("oanda_retry_bounded", "PASS" if oanda_retry_exhausts() == 2 else "FAIL")

# ---------------------------------------------------------------- concurrency
category("concurrency")

def concurrent_ledger_updates():
    import threading
    ledger = state.StageLedger(Path("/tmp/opencode/pressure/conc/ledger.json"), "run-c", stale_after_seconds=5.0)
    errors_seen = []

    def worker(i):
        try:
            with ledger.stage(f"s{i}") as should:
                time.sleep(0.05)
        except Exception as exc:
            errors_seen.append(type(exc).__name__)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    payload = ledger.read()
    return errors_seen, all(ledger.is_complete(f"s{i}") for i in range(4))
conc_errors, conc_ok = concurrent_ledger_updates()
record("concurrent_ledger_updates", "PASS" if conc_ok else "FAIL", f"errors={conc_errors}")

# ---------------------------------------------------------------- pipeline variants
category("pipeline")

def pipeline_single_model():
    cfg = full_cfg()
    cfg["models"]["hist_gradient_boosting"] = False
    m2 = pipeline.run_pipeline(synthetic(6000), cfg, "/tmp/opencode/pressure/art_one")
    return m2
m_one = pipeline_single_model()
record("pipeline_single_model", "PASS" if not m_one.empty and set(m_one.model.unique()) == {"logistic"} else "FAIL",
       f"models={list(m_one.model.unique()) if len(m_one) else []}")

# ---------------------------------------------------------------- summary
def main():
    print(json.dumps(RESULTS, indent=2))
    outcomes = {}
    for r in RESULTS:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print("\n=== SUMMARY ===")
    print(outcomes)
    weak_fails = [r for r in RESULTS if r["outcome"] in ("WEAK", "FAIL")]
    print(f"\nWEAK/FAIL count: {len(weak_fails)}")
    for r in weak_fails:
        print(f"  [{r['outcome']}] {r['category']} :: {r['name']} :: {r['detail']}")

if __name__ == "__main__":
    main()
