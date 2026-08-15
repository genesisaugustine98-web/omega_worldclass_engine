from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
from .utils import sha256_file, atomic_json

REQUIRED = ["timestamp", "open", "high", "low", "close"]

@dataclass(frozen=True)
class Manifest:
    source: str
    symbol: str
    retrieved_at: str
    path: str
    sha256: str
    rows: int
    first_timestamp: str
    last_timestamp: str

def ingest_local(source_path, root, source, symbol) -> Manifest:
    """Copy user-provided data into immutable content-addressed storage."""
    src = Path(source_path)
    digest = sha256_file(src)
    suffix = src.suffix.lower()
    dst = Path(root) / "raw" / source / symbol / f"{digest}{suffix}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): shutil.copy2(src, dst)
    frame = read_market_file(dst)
    m = Manifest(source, symbol, datetime.now(timezone.utc).isoformat(), str(dst), digest,
                 len(frame), str(frame.timestamp.min()), str(frame.timestamp.max()))
    atomic_json(dst.with_suffix(dst.suffix + ".manifest.json"), asdict(m))
    return m

def read_market_file(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.copy(); df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)

def normalize_to_parquet(df, path, source="unknown", symbol="unknown"):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["source"] = source; out["symbol"] = symbol
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    atomic_json(path.with_suffix(".manifest.json"), {"sha256": sha256_file(path), "rows": len(out),
        "source": source, "symbol": symbol, "created_at": datetime.now(timezone.utc).isoformat()})
    return path

def synthetic_fx(n=8000, seed=42, start="2020-01-05 22:00:00+00:00"):
    """Deterministic synthetic fixture; never represented as market history.

    Generates enough candidate bars that the weekday filter always yields at
    least ``n`` rows for any ``n >= 1``, so the fixture never crashes or
    silently returns fewer bars than requested.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=int(n * 2.6) + 20, freq="30min", tz="UTC")
    ts = ts[ts.dayofweek < 5][:n]
    n = len(ts)
    vol = np.where((np.arange(n)//700)%2, 0.00045, 0.00015)
    ret = rng.normal(0, vol) + np.sin(np.arange(n)/150) * 0.00003
    close = 1.10 * np.exp(np.cumsum(ret)); open_ = np.r_[close[0], close[:-1]]
    wiggle = np.abs(rng.normal(0, vol/2))
    high = np.maximum(open_, close) * (1 + wiggle); low = np.minimum(open_, close) * (1 - wiggle)
    spread = np.clip(rng.normal(0.00008, 0.00002, n), 0.00002, None)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "spread": spread, "volume": rng.integers(50, 500, n)})

# CONVERSATION_HOOK: Add vendor adapters only after source terms, credentials, and symbol/date coverage are approved.
