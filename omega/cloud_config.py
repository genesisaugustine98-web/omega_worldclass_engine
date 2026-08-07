from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


_ENV = re.compile(r"\$\{([^:}]+):-([^}]+)\}")


def _expand(value):
    if isinstance(value, str):
        return _ENV.sub(lambda match: os.getenv(match.group(1), match.group(2)), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_cloud_config(path: str | Path = "config/cloud_free.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = _expand(yaml.safe_load(handle))
    source = config["data_source"]
    if source["granularity"] != "M30":
        raise ValueError("The historical V1 adapter supports M30 only")
    if source["partition"] != "month":
        raise ValueError("The resumable V1 pipeline requires monthly partitions")
    if not isinstance(source.get("explicit_terms_accepted"), bool):
        raise ValueError("explicit_terms_accepted must be a boolean")
    return config


def build_provider(config: dict):
    source = config["data_source"]
    if source["provider"] == "oanda":
        from .providers.oanda import OandaProvider

        return OandaProvider(
            terms_accepted=source["explicit_terms_accepted"],
            environment=source.get("environment", "practice"),
        )
    raise ValueError(f"Unsupported historical provider: {source['provider']}")
