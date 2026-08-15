from __future__ import annotations
import numpy as np
import pandas as pd

def hypothetical_state_response(prob, forward_return, threshold=.58, spread_bps=1, slippage_bps=.5,
                                annualization_bars=12480, holding_bars=1, one_bar_latency=True):
    """Non-overlapping research response; not a trading recommendation.

    A decision at bar ``t`` executes at bar ``t+1`` when ``one_bar_latency`` is
    true (signal can only be observed after the close), matching the declared
    backtest configuration instead of silently assuming a same-close fill.
    """
    p=np.asarray(prob,float); r=np.asarray(forward_return,float)
    if p.shape != r.shape: raise ValueError("probability and forward_return must have equal shape")
    if holding_bars < 1: raise ValueError("holding_bars must be at least 1")
    missing_returns = int((~np.isfinite(r)).sum())
    latency = 1 if one_bar_latency else 0
    position=np.zeros(len(p)); entries=np.zeros(len(p),dtype=bool)
    next_decision=0
    for i, probability in enumerate(p):
        if i >= next_decision and np.isfinite(probability) and probability >= threshold:
            fill = i + latency
            if fill < len(p):
                position[fill]=1.0; entries[fill]=True
                next_decision=fill+holding_bars
    round_trip_bps=2*(spread_bps+slippage_bps)
    costs=entries.astype(float)*round_trip_bps*1e-4
    pnl=position*np.nan_to_num(r)-costs; equity=np.cumprod(1+pnl); peak=np.maximum.accumulate(equity); dd=equity/peak-1
    periods_per_year=annualization_bars/holding_bars
    std=np.std(pnl[entries]) if entries.any() else 0.0
    sharpe=np.mean(pnl[entries])/std*np.sqrt(periods_per_year) if std else 0.0
    return pd.DataFrame({"probability":p,"position":position,"forward_return":r,"cost":costs,"pnl":pnl,"equity":equity,"drawdown":dd}), {
      "entries":int(entries.sum()),"holding_bars":holding_bars,"one_bar_latency":bool(one_bar_latency),
      "missing_returns":missing_returns,
      "exposure_fraction":float(position.mean()),
      "total_return":float(equity[-1]-1),"max_drawdown":float(dd.min()),"naive_sharpe":float(sharpe)}

def attribution(label, y, p, threshold=.58):
    selected=p>=threshold; n=int(selected.sum()); wins=int(((y==1)&selected).sum())
    return {"phenomenon":label,"selected_windows":n,"observed_positive":wins,"precision":wins/n if n else None,
            "language":"Associational out-of-sample evidence; not proof of trader intent or causality."}
