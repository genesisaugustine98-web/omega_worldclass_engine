from __future__ import annotations

from pathlib import Path

from .config import _read_yaml, expand_env
from .errors import ConfigError


def load_cloud_config(path: str | Path = "config/cloud_free.yaml") -> dict:
    config = _read_yaml(path)
    source = config.get("data_source")
    if not isinstance(source, dict):
        raise ConfigError("config[data_source] must be a mapping")
    if source.get("granularity") != "M30":
        raise ConfigError("config[data_source].granularity must be M30 (V1 adapter supports M30 only)")
    if source.get("partition") != "month":
        raise ConfigError("config[data_source].partition must be 'month'")
    if not isinstance(source.get("explicit_terms_accepted"), bool):
        raise ConfigError("config[data_source].explicit_terms_accepted must be a boolean")
    if source.get("provider") == "oanda" and source.get("environment") not in {None, "practice", "live"}:
        raise ConfigError("config[data_source].environment must be 'practice' or 'live'")
    if source.get("provider") == "polygon" and not isinstance(source.get("require_intraday", True), bool):
        raise ConfigError("config[data_source].require_intraday must be a boolean")
    return config


def build_provider(config: dict):
    source = config["data_source"]
    if source["provider"] == "oanda":
        from .providers.oanda import OandaProvider

        return OandaProvider(
            terms_accepted=source["explicit_terms_accepted"],
            environment=source.get("environment", "practice"),
        )
    if source["provider"] == "twelvedata":
        from .providers.twelvedata import TwelveDataProvider

        return TwelveDataProvider(
            terms_accepted=source["explicit_terms_accepted"],
            pacing_seconds=float(source.get("pacing_seconds", 8.0)),
        )
    if source["provider"] == "polygon":
        from .providers.polygon import PolygonProvider

        return PolygonProvider(
            terms_accepted=source["explicit_terms_accepted"],
            pacing_seconds=float(source.get("pacing_seconds", 12.0)),
            require_intraday=bool(source.get("require_intraday", True)),
        )
    raise ConfigError(f"Unsupported historical provider: {source['provider']}")
