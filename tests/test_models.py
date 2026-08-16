from pathlib import Path

import pandas as pd
import pytest

from omega.errors import ConfigError
from omega.models import (
    MODEL_BUILDERS,
    available_models,
    make_model,
    model_available,
    model_catalog,
)


def test_core_models_always_available():
    for name in ("logistic", "hist_gradient_boosting"):
        assert model_available(name)
        assert name in available_models()


def test_catalog_contains_only_available_models():
    catalog = model_catalog(0)
    assert set(catalog) == set(available_models())
    for name, estimator in catalog.items():
        assert estimator is not None


def test_make_model_unknown_name_is_config_error():
    with pytest.raises(ConfigError, match="unknown model"):
        make_model("not_a_model")


def test_make_model_not_wired_is_config_error():
    with pytest.raises(ConfigError, match="not yet wired"):
        make_model("causal_transformer")


def test_optional_model_availability_tracks_importability():
    import importlib.util
    expected = importlib.util.find_spec("lightgbm") is not None
    assert model_available("lightgbm") is expected
    if expected:
        assert "lightgbm" in available_models()


def test_all_builders_produce_calibrated_models_that_fit():
    import numpy as np
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] + X["f1"] > 0).astype(int)
    for name in available_models():
        model = make_model(name, seed=7)
        model.fit(X.iloc[:300], y.iloc[:300], X.iloc[300:350], y.iloc[300:350], alpha=0.1)
        p = model.predict_proba(X.iloc[350:])
        assert p.shape == (50,)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()


def test_missing_optional_dependency_raises_clear_config_error(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ConfigError, match="lightgbm"):
        make_model("lightgbm")
    assert not model_available("lightgbm")
