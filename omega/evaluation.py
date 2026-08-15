from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score

@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray; calibration: np.ndarray; test: np.ndarray

def walk_forward_splits(n, train_bars, test_bars, step_bars, embargo_bars, calibration_fraction=.2):
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_bars < 2:
        raise ValueError("train_bars must be at least 2 so a calibration split can be carved out")
    end=train_bars
    while end + embargo_bars + test_bars <= n:
        cal_size=max(100,int(train_bars*calibration_fraction))
        if cal_size >= train_bars:
            cal_size = train_bars - 1
        train=np.arange(end-train_bars,end-cal_size)
        calibration=np.arange(end-cal_size,end); test=np.arange(end+embargo_bars,end+embargo_bars+test_bars)
        yield TemporalSplit(train,calibration,test); end += step_bars

def probability_metrics(y, p):
    y=np.asarray(y); p=np.clip(np.asarray(p),1e-6,1-1e-6)
    result={"brier":brier_score_loss(y,p),"log_loss":log_loss(y,p,labels=[0,1]),"average_precision":average_precision_score(y,p)}
    result["roc_auc"]=roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan
    return result

def expected_calibration_error(y,p,bins=10):
    frame=pd.DataFrame({"y":y,"p":p}); frame["bin"]=pd.cut(frame.p,np.linspace(0,1,bins+1),include_lowest=True)
    grouped=frame.groupby("bin",observed=False).agg(y=("y","mean"),p=("p","mean"),n=("y","size")).dropna()
    return float(((grouped.y-grouped.p).abs()*grouped.n).sum()/max(grouped.n.sum(),1))

# CONVERSATION_HOOK: Add combinatorial purged CV and probability-of-backtest-overfitting diagnostics.
