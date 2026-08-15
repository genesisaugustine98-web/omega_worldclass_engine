from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

class DataValidationError(RuntimeError): pass

@dataclass
class Check:
    layer: int; name: str; passed: bool; detail: str

    def as_dict(self) -> dict:
        return {
            "layer": int(self.layer),
            "name": str(self.name),
            "passed": bool(self.passed),
            "detail": str(self.detail),
        }

def _robust_z(x):
    x = pd.Series(x).dropna(); med=x.median(); mad=(x-med).abs().median()
    return pd.Series(0.0, index=x.index) if mad == 0 else 0.6745*(x-med)/mad

def validate(df: pd.DataFrame, timeframe_minutes=30, require_spread=False, max_robust_return_z=25.0, raise_on_failure=True):
    checks=[]; required={"timestamp","open","high","low","close"}
    checks.append(Check(1,"schema", required.issubset(df.columns) and len(df) > 0,
                        f"columns={list(df.columns)} rows={len(df)}"))
    if not checks[-1].passed:
        if raise_on_failure: raise DataValidationError(checks[-1].detail)
        return [c.as_dict() for c in checks]
    try:
        prices=df[["open","high","low","close"]].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Price columns must be numeric: {exc}") from exc
    ts=pd.to_datetime(df.timestamp, utc=True, errors="coerce")
    checks.append(Check(2,"timestamp_integrity", ts.notna().all() and ts.is_monotonic_increasing and not ts.duplicated().any(),
                        f"invalid={ts.isna().sum()} duplicates={ts.duplicated().sum()}"))
    invalid_weekend = (ts.dt.dayofweek == 5) | ((ts.dt.dayofweek == 6) & (ts.dt.hour < 20))
    delta=ts.diff().dt.total_seconds().div(60); unexpected=(delta < timeframe_minutes).sum()
    checks.append(Check(3,"calendar_and_gaps", not invalid_weekend.any() and unexpected==0,
                        f"closed_session_rows={invalid_weekend.sum()} too_short_intervals={unexpected}"))
    ohlc=(prices.gt(0).all().all() and np.isfinite(prices).to_numpy().all() and
          (prices.high >= prices[["open","close","low"]].max(axis=1)).all() and
          (prices.low <= prices[["open","close","high"]].min(axis=1)).all())
    z=_robust_z(np.log(prices.close).diff()).abs(); outliers=int((z > max_robust_return_z).sum())
    checks.append(Check(4,"ohlc_and_outliers", ohlc and outliers==0, f"ohlc_valid={ohlc} extreme_returns={outliers}"))
    spread_ok = ("spread" in df and df.spread.notna().all() and (df.spread >= 0).all())
    checks.append(Check(5,"spread_quality", spread_ok or not require_spread, f"required={require_spread} valid={spread_ok}"))
    availability_ok = True
    if "available_at" in df:
        availability_ok = (pd.to_datetime(df.available_at, utc=True) <= ts).all()
    checks.append(Check(6,"point_in_time_alignment", availability_ok, "available_at must not exceed timestamp"))
    hashes_ok = df.astype(str).agg("|".join, axis=1).duplicated().sum() == 0
    checks.append(Check(7,"row_integrity", hashes_ok, "exact duplicate row check"))
    failed=[c for c in checks if not c.passed]
    if failed and raise_on_failure: raise DataValidationError("; ".join(f"L{c.layer}:{c.name}:{c.detail}" for c in failed))
    return [c.as_dict() for c in checks]

# CONVERSATION_HOOK: Add source-to-source quote reconciliation and versioned holiday calendars.
