from __future__ import annotations
import os, re, yaml
from pathlib import Path

_ENV = re.compile(r"\$\{([^:}]+):-([^}]+)\}")

def _expand(value):
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.getenv(m.group(1), m.group(2)), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value

def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        cfg = _expand(yaml.safe_load(handle))
    if cfg["data"]["timeframe_minutes"] != 30:
        raise ValueError("V1 supports 30-minute bars only")
    if cfg["labels"]["horizon_bars"] > cfg["evaluation"]["embargo_bars"]:
        raise ValueError("Embargo must cover the label horizon")
    return cfg

# CONVERSATION_HOOK: Replace dictionary validation with a versioned Pydantic schema when config migrations begin.
