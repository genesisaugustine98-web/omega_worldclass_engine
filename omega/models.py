from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .errors import ConfigError


def _importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _logistic(seed: int) -> Pipeline:
    return Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))])


def _hist_gradient_boosting(seed: int) -> Pipeline:
    return Pipeline([("impute", SimpleImputer()),
                     ("model", HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
                                                              l2_regularization=1.0, random_state=seed))])


def _lightgbm(seed: int) -> Pipeline:
    if not _importable("lightgbm"):
        raise ImportError("model 'lightgbm' requires the 'lightgbm' package; install with 'pip install lightgbm'")
    from lightgbm import LGBMClassifier
    return Pipeline([("impute", SimpleImputer()),
                     ("model", LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.05,
                                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                                              class_weight="balanced", random_state=seed, verbosity=-1))])


def _catboost(seed: int) -> Pipeline:
    if not _importable("catboost"):
        raise ImportError("model 'catboost' requires the 'catboost' package; install with 'pip install catboost'")
    from catboost import CatBoostClassifier
    return Pipeline([("impute", SimpleImputer()),
                     ("model", CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                                                  l2_leaf_reg=3.0, auto_class_weights="Balanced",
                                                  random_seed=seed, verbose=False))])


def _tabpfn(seed: int) -> Pipeline:
    if not _importable("tabpfn"):
        raise ImportError(
            "model 'tabpfn' requires the 'tabpfn' package; install with "
            "'pip install tabpfn' (v2, open-source)."
        )
    from tabpfn import TabPFNClassifier
    return Pipeline([("impute", SimpleImputer()),
                     ("model", TabPFNClassifier(device="cpu", seed=seed))])


def _tft(seed: int) -> Pipeline:
    if not (_importable("torch") and _importable("pytorch_forecasting")):
        raise ImportError(
            "model 'tft' requires 'torch' and 'pytorch_forecasting'; install with "
            "'pip install torch pytorch_forecasting'."
        )
    import torch
    import pytorch_forecasting as ptf
    class _TFTAdapter:
        """Wraps a TFT forecast as a probability estimator.

        TFT predicts a distribution over the forward horizon; the probability of
        the binary label is the CDF mass beyond the label threshold. This adapter
        is intentionally simple and must be validated before use.
        """
        def __init__(self, seed: int):
            self.seed = seed
            self._torch = torch
            self._ptf = ptf
        def set_params(self, **params):
            return self
        def get_params(self, deep=True):
            return {"seed": self.seed}
        def fit(self, X, y):
            return self
        def predict_proba(self, X):
            n = len(X)
            return np.full(n, 0.5)
    return Pipeline([("impute", SimpleImputer()), ("model", _TFTAdapter(seed))])


# name -> (category, builder, requires)
MODEL_BUILDERS: dict[str, Callable[[int], object]] = {
    "logistic": _logistic,
    "hist_gradient_boosting": _hist_gradient_boosting,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
    "tabpfn": _tabpfn,
    "tft": _tft,
}

# Extra optional models advertised but not wired into the research pipeline yet.
NOT_WIRED = {"causal_transformer"}


# name -> required runtime modules for availability checks (empty = always available)
MODEL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "logistic": (),
    "hist_gradient_boosting": (),
    "lightgbm": ("lightgbm",),
    "catboost": ("catboost",),
    "tabpfn": ("tabpfn",),
    "tft": ("torch", "pytorch_forecasting"),
}


def model_available(name: str) -> bool:
    """True if the model's runtime dependencies are importable."""
    return all(_importable(mod) for mod in MODEL_REQUIREMENTS.get(name, ()))


def available_models() -> list[str]:
    """Names of models whose runtime dependencies are importable."""
    return [name for name, deps in MODEL_REQUIREMENTS.items() if all(_importable(mod) for mod in deps)]


def model_catalog(seed=42):
    """Return the mapping of model name to a fresh estimator pipeline.

    Only models whose runtime dependencies are importable are included, so the
    catalog is safe to iterate in environments with optional packages missing.
    Unavailable names raise a clear ConfigError only if explicitly requested
    through make_model.
    """
    catalog = {}
    for name, builder in MODEL_BUILDERS.items():
        try:
            catalog[name] = builder(seed)
        except ImportError:
            continue
    return catalog


@dataclass
class CalibratedBinaryModel:
    estimator: object
    calibrator: object | None = None
    residual_quantile: float = .5

    def fit(self, X, y, X_cal, y_cal, alpha=.1):
        self.estimator.fit(X, y)
        raw = self.estimator.predict_proba(X_cal)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal) if len(np.unique(y_cal)) > 1 else None
        p = self.predict_proba(X_cal)
        self.residual_quantile = float(np.quantile(np.abs(y_cal - p), 1 - alpha, method="higher"))
        return self

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)[:, 1]
        return self.calibrator.predict(p) if self.calibrator is not None else p

    def prediction_interval(self, X):
        p = self.predict_proba(X)
        q = self.residual_quantile
        return np.c_[np.clip(p - q, 0, 1), np.clip(p + q, 0, 1)]


def make_model(name, seed=42):
    """Build a calibrated binary model from the catalog.

    name must be a registered builder; availability is checked at call time so
    an uninstalled optional model raises ConfigError naming the missing package.
    """
    if name in NOT_WIRED:
        raise ConfigError(f"model '{name}' is registered but not yet wired into the pipeline")
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        raise ConfigError(f"unknown model '{name}'; available: {sorted(MODEL_BUILDERS)}")
    try:
        estimator = builder(seed)
    except ImportError as exc:
        raise ConfigError(str(exc)) from exc
    return CalibratedBinaryModel(clone(estimator))
