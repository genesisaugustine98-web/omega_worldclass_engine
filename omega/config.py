from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import yaml

from .errors import ConfigError

_ENV = re.compile(r"\$\{([^}]+)\}")


def expand_env(value):
    """Recursively expand ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):

        def replace(match: re.Match) -> str:
            expression = match.group(1)
            if ":-" in expression:
                name, default = expression.split(":-", 1)
                return os.getenv(name, default)
            return os.getenv(expression)

        return _ENV.sub(replace, value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def _require_mapping(config: dict, path: str) -> dict:
    section = config.get(path)
    if not isinstance(section, dict):
        raise ConfigError(f"config[{path}] must be a mapping, got {type(section).__name__}")
    return section


def _require(key: str, value, predicate: Callable[[object], bool], message: str) -> None:
    if not predicate(value):
        raise ConfigError(f"config[{key}] {message} (got {value!r})")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bool(value) -> bool:
    return isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _in_unit_interval(value) -> bool:
    return _is_number(value) and 0.0 < value <= 1.0


def _in_closed_unit_interval(value) -> bool:
    return _is_number(value) and 0.0 <= value <= 1.0


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load and validate the research pipeline configuration.

    Every key the pipeline reads is type-checked and range-checked at load time
    so a typo or impossible value fails fast with an actionable ConfigError
    instead of a deep KeyError or a silently wrong run.
    """
    config = _read_yaml(path)

    project = _require_mapping(config, "project")
    _require("project.seed", project.get("seed"), _is_int, "must be an integer")

    data = _require_mapping(config, "data")
    _require("data.timeframe_minutes", data.get("timeframe_minutes"), lambda v: v == 30,
             "must be 30 (V1 supports 30-minute bars only)")
    _require("data.require_spread", data.get("require_spread", False), _is_bool, "must be a boolean")
    _require("data.max_robust_return_z", data.get("max_robust_return_z", 25.0), _is_number,
             "must be a number")

    features = _require_mapping(config, "features")
    windows = features.get("windows")
    _require("features.windows", windows,
             lambda v: isinstance(v, list) and v and all(_is_int(w) and w > 0 for w in v),
             "must be a non-empty list of positive integers")
    _require("features.include_time_features", features.get("include_time_features", True),
             _is_bool, "must be a boolean")

    labels = _require_mapping(config, "labels")
    _require("labels.horizon_bars", labels.get("horizon_bars"), lambda v: _is_int(v) and v > 0,
             "must be a positive integer")
    _require("labels.lookback_bars", labels.get("lookback_bars"), lambda v: _is_int(v) and v > 0,
             "must be a positive integer")
    label_defaults = {
        "move_atr": 1.25,
        "reversal_atr": 0.75,
        "compression_quantile": 0.20,
        "expansion_quantile": 0.80,
    }
    for key, default in label_defaults.items():
        value = labels.get(key, default)
        labels.setdefault(key, value)
        _require(f"labels.{key}", value, _is_number, "must be a number")
    if not (0.0 <= labels.get("compression_quantile", 0.2) < labels.get("expansion_quantile", 0.8) <= 1.0):
        raise ConfigError(
            "config[labels] requires 0 <= compression_quantile < expansion_quantile <= 1 "
            f"(got compression={labels.get('compression_quantile')}, expansion={labels.get('expansion_quantile')})"
        )

    evaluation = _require_mapping(config, "evaluation")
    for key in ("train_bars", "test_bars", "step_bars"):
        _require(f"evaluation.{key}", evaluation.get(key), lambda v: _is_int(v) and v > 0,
                 "must be a positive integer")
    _require("evaluation.embargo_bars", evaluation.get("embargo_bars"),
             lambda v: _is_int(v) and v >= 0, "must be a non-negative integer")
    _require("evaluation.calibration_fraction", evaluation.get("calibration_fraction", 0.2),
             lambda v: _is_number(v) and 0.0 < v < 1.0, "must be in (0, 1)")
    _require("evaluation.alpha", evaluation.get("alpha", 0.1), _in_unit_interval, "must be in (0, 1]")
    _require("evaluation.abstain_below", evaluation.get("abstain_below", 0.5),
             _in_closed_unit_interval, "must be in [0, 1]")

    if evaluation.get("embargo_bars", 0) < labels.get("horizon_bars", 1):
        raise ConfigError(
            f"config[evaluation].embargo_bars ({evaluation.get('embargo_bars')}) must cover the "
            f"label horizon config[labels].horizon_bars ({labels.get('horizon_bars')})"
        )
    if evaluation.get("train_bars", 1) <= evaluation.get("embargo_bars", 0):
        raise ConfigError("config[evaluation].train_bars must exceed embargo_bars")

    models = _require_mapping(config, "models")
    for model, enabled in models.items():
        _require(f"models.{model}", enabled, _is_bool, "must be a boolean")
    if not any(models.values()):
        raise ConfigError("At least one model must be enabled in config[models]")

    if "backtest" in config:
        backtest = _require_mapping(config, "backtest")
        for key in ("spread_bps", "slippage_bps"):
            _require(f"backtest.{key}", backtest.get(key), lambda v: _is_number(v) and v >= 0,
                     "must be a non-negative number")
        _require("backtest.annualization_bars", backtest.get("annualization_bars"),
                 lambda v: _is_int(v) and v > 0, "must be a positive integer")
        _require("backtest.one_bar_latency", backtest.get("one_bar_latency", True),
                 _is_bool, "must be a boolean")

    return config


def _read_yaml(path: str | Path) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file is not valid YAML: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration file must contain a top-level mapping: {path}")
    return expand_env(loaded)
