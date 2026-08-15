from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

REGISTRY={}


def _require_causal_input(df):
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature input missing columns: {missing}")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Feature input timestamps must be sorted ascending")
    if timestamps.duplicated().any():
        raise ValueError("Feature input timestamps must be unique")


def register_feature(category, name, rationale):
    def deco(fn):
        REGISTRY[name]={"category":category,"rationale":rationale,"function":fn}; return fn
    return deco

@register_feature("Returns","log_return","Price changes aggregate repricing and flow effects; sign is not causal attribution.")
def log_return(df, w): return np.log(df.close).diff(w)

@register_feature("Volatility","realized_vol","Volatility conditions leverage, liquidity, and the significance of a move.")
def realized_vol(df, w): return np.log(df.close).diff().rolling(w).std()

@register_feature("Range","atr_fraction","Range proxies intrabar uncertainty and liquidity consumption.")
def atr_fraction(df, w):
    prev=df.close.shift(); tr=pd.concat([df.high-df.low,(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(w).mean()/df.close

@register_feature("Trend","efficiency_ratio","Directional efficiency separates persistent displacement from noisy travel.")
def efficiency_ratio(df, w): return df.close.diff(w).abs()/df.close.diff().abs().rolling(w).sum()

@register_feature("Location","close_location","Close location within range is a limited proxy for intrabar pressure.")
def close_location(df, w):
    lo=df.low.rolling(w).min(); hi=df.high.rolling(w).max(); return (df.close-lo)/(hi-lo).replace(0,np.nan)

def build_features(df, windows=(2,4,8,16,48), include_time=True):
    """Build features using data available at each row's timestamp only.

    Rolling operations intentionally have no centering and no backward fill.
    NaNs at the beginning of a warm-up window remain NaN for the trainer to
    remove using the training-period policy.
    """
    _require_causal_input(df)
    windows = tuple(windows)
    if not windows or any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError("windows must be a non-empty tuple/list of positive integers")
    out=pd.DataFrame(index=df.index)
    for name, meta in REGISTRY.items():
        for w in windows: out[f"{name}_{w}"]=meta["function"](df,w)
    if "spread" in df: out["spread_fraction"]=df.spread/df.close
    if "volume" in df:
        out["volume_z_48"]=(df.volume-df.volume.rolling(48).mean())/df.volume.rolling(48).std()
    if include_time:
        ts=pd.to_datetime(df.timestamp,utc=True); out["hour_sin"]=np.sin(2*np.pi*ts.dt.hour/24); out["hour_cos"]=np.cos(2*np.pi*ts.dt.hour/24)
        out["london_ny_overlap"]=ts.dt.hour.between(13,16).astype(int)
    out["timestamp"]=df.timestamp
    return out.replace([np.inf,-np.inf],np.nan)

# CONVERSATION_HOOK: Grow toward 500 features only through pre-registered hypotheses, tests, and ablations.
