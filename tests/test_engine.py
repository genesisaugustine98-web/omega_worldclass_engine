import numpy as np
import pandas as pd
import pytest
from omega.data import synthetic_fx
from omega.validation import validate, DataValidationError
from omega.features import build_features
from omega.labels import label_phenomena, LABELS
from omega.evaluation import walk_forward_splits

def test_seven_layers_pass_synthetic():
    checks=validate(synthetic_fx(1000)); assert len(checks)==7; assert all(c["passed"] for c in checks)

def test_bad_ohlc_stops():
    df=synthetic_fx(100); df.loc[10,"high"]=df.loc[10,"low"]-1
    with pytest.raises(DataValidationError): validate(df)

def test_weekend_stops():
    df=synthetic_fx(100); df.loc[10,"timestamp"]=pd.Timestamp("2024-01-06",tz="UTC"); df=df.sort_values("timestamp").reset_index(drop=True)
    with pytest.raises(DataValidationError): validate(df)

def test_feature_and_label_contract():
    df=synthetic_fx(500); x=build_features(df); y=label_phenomena(df)
    assert set(LABELS).issubset(y.columns); assert "timestamp" in x; assert len(x)==len(y)==len(df)
    assert y[LABELS].iloc[-8:].isna().all().all()


def test_features_are_causal_and_input_order_is_explicit():
    df = synthetic_fx(250)
    baseline = build_features(df)
    changed = df.copy()
    changed.loc[200:, "close"] *= 1.25
    changed.loc[200:, "high"] *= 1.25
    changed.loc[200:, "low"] *= 1.25
    changed_features = build_features(changed)
    columns = [column for column in baseline.columns if column != "timestamp"]
    assert np.allclose(
        baseline.loc[:150, columns].to_numpy(),
        changed_features.loc[:150, columns].to_numpy(),
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        build_features(df.iloc[::-1].reset_index(drop=True))


def test_labels_keep_timestamp_alignment_and_unknown_terminal_horizon():
    df = synthetic_fx(300)
    labels = label_phenomena(df, horizon=8)
    assert labels.timestamp.equals(df.timestamp)
    assert labels[LABELS].iloc[:-8].notna().any(axis=None)
    assert labels[LABELS].iloc[-8:].isna().all().all()

def test_walk_forward_embargo():
    splits=list(walk_forward_splits(6000,3000,1000,1000,8,.2)); assert splits
    for s in splits: assert s.test.min()-s.calibration.max() >= 9
