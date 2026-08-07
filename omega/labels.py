from __future__ import annotations
import numpy as np
import pandas as pd

LABELS=["trend_ignition","liquidity_sweep","volatility_expansion","compression","failed_breakout","mean_reversion"]

def label_phenomena(df, horizon=8, lookback=48, move_atr=1.25, reversal_atr=.75, compression_q=.2, expansion_q=.8):
    """Multi-label future-path definitions. Historical narratives remain hypotheses, never embedded as facts."""
    prev=df.close.shift(); tr=pd.concat([df.high-df.low,(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.rolling(lookback).mean(); future_close=df.close.shift(-horizon); move=(future_close-df.close)/atr
    future_high=pd.concat([df.high.shift(-i) for i in range(1,horizon+1)],axis=1).max(axis=1)
    future_low=pd.concat([df.low.shift(-i) for i in range(1,horizon+1)],axis=1).min(axis=1)
    rv=np.log(df.close).diff().rolling(lookback).std(); future_rv=np.log(df.close).diff().shift(-horizon).rolling(horizon).std()
    prior_hi=df.high.shift(1).rolling(lookback).max(); prior_lo=df.low.shift(1).rolling(lookback).min()
    deviation=(df.close-df.close.rolling(lookback).mean())/atr
    future_dev=(future_close-df.close.rolling(lookback).mean())/atr
    y=pd.DataFrame(index=df.index)
    y["trend_ignition"]=(move.abs() >= move_atr).astype(int)
    swept=((future_high > prior_hi) | (future_low < prior_lo)); returned=(future_close <= prior_hi) & (future_close >= prior_lo)
    y["liquidity_sweep"]=(swept & returned).astype(int)
    y["volatility_expansion"]=(future_rv > rv.rolling(lookback).quantile(expansion_q)).astype(int)
    y["compression"]=(future_rv < rv.rolling(lookback).quantile(compression_q)).astype(int)
    breakout=(df.close > prior_hi) | (df.close < prior_lo)
    y["failed_breakout"]=(breakout & (move.abs() >= reversal_atr) & (np.sign(move)==-np.sign(df.close-df.open))).astype(int)
    y["mean_reversion"]=((deviation.abs()>1) & (future_dev.abs()<deviation.abs())).astype(int)
    y["forward_return"]=future_close/df.close-1; y["timestamp"]=df.timestamp
    y.iloc[-horizon:, y.columns.get_indexer(LABELS+["forward_return"])]=np.nan
    return y

# CONVERSATION_HOOK: Tune definitions only on development periods and version every label specification.
