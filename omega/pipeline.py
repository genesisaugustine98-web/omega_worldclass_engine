from __future__ import annotations
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from .validation import validate
from .features import build_features
from .labels import label_phenomena, LABELS
from .evaluation import walk_forward_splits, probability_metrics, expected_calibration_error
from .models import make_model
from .backtest import hypothetical_state_response, attribution
from .utils import atomic_json, seed_everything

def run_pipeline(df, cfg, artifact_dir="artifacts/run"):
    seed_everything(cfg["project"]["seed"]); art=Path(artifact_dir); art.mkdir(parents=True,exist_ok=True)
    report=validate(df,cfg["data"]["timeframe_minutes"],cfg["data"]["require_spread"],cfg["data"]["max_robust_return_z"])
    atomic_json(art/"validation.json",{"checks":report})
    X=build_features(df,cfg["features"]["windows"],cfg["features"]["include_time_features"])
    label_cfg = cfg["labels"]
    y=label_phenomena(
        df,
        horizon=label_cfg["horizon_bars"],
        lookback=label_cfg["lookback_bars"],
        move_atr=label_cfg["move_atr"],
        reversal_atr=label_cfg["reversal_atr"],
        compression_q=label_cfg["compression_quantile"],
        expansion_q=label_cfg["expansion_quantile"],
    )
    merged=X.merge(y,on="timestamp").dropna(); feature_cols=[c for c in X if c!="timestamp"]
    e=cfg["evaluation"]; all_results=[]; prediction_frames=[]
    for fold,split in enumerate(walk_forward_splits(len(merged),e["train_bars"],e["test_bars"],e["step_bars"],e["embargo_bars"],e["calibration_fraction"])):
      for label in LABELS:
       for name,enabled in [("logistic",cfg["models"]["logistic"]),("hist_gradient_boosting",cfg["models"]["hist_gradient_boosting"])]:
        if not enabled: continue
        train,cal,test=merged.iloc[split.train],merged.iloc[split.calibration],merged.iloc[split.test]
        if train[label].nunique()<2 or cal[label].nunique()<2: continue
        model=make_model(name,cfg["project"]["seed"]+fold).fit(train[feature_cols],train[label],cal[feature_cols],cal[label],e["alpha"])
        p=model.predict_proba(test[feature_cols]); m=probability_metrics(test[label],p); m["ece"]=expected_calibration_error(test[label],p)
        prevalence=float(train[label].mean()); baseline=np.full(len(test),prevalence)
        baseline_metrics=probability_metrics(test[label],baseline)
        m["train_prevalence"]=prevalence; m["baseline_brier"]=baseline_metrics["brier"]
        m["brier_skill_vs_prevalence"]=1-m["brier"]/m["baseline_brier"] if m["baseline_brier"] else np.nan
        all_results.append({"fold":fold,"label":label,"model":name,**m})
        prediction_frames.append(pd.DataFrame({"timestamp":test.timestamp,"fold":fold,"label":label,"model":name,"y":test[label],"p":p,"forward_return":test.forward_return}))
        joblib.dump(model,art/f"model_{label}_{name}_fold{fold}.joblib")
    metrics=pd.DataFrame(all_results); predictions=pd.concat(prediction_frames,ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics.to_csv(art/"metrics.csv",index=False); predictions.to_parquet(art/"predictions.parquet",index=False)
    attrs=[]
    if not predictions.empty:
      for (label,name),g in predictions.groupby(["label","model"]): attrs.append(attribution(label,g.y.to_numpy(),g.p.to_numpy(),e["abstain_below"])|{"model":name})
      trend=predictions[(predictions.label=="trend_ignition") & (predictions.model=="hist_gradient_boosting")]
      if len(trend):
        _, bt=hypothetical_state_response(trend.p,trend.forward_return,e["abstain_below"],cfg["backtest"]["spread_bps"],cfg["backtest"]["slippage_bps"],cfg["backtest"]["annualization_bars"],label_cfg["horizon_bars"]); atomic_json(art/"hypothetical_backtest.json",bt)
    atomic_json(art/"attribution.json",{"items":attrs}); return metrics

# CONVERSATION_HOOK: Add experiment registry IDs, resumable folds, and elapsed-time checkpoints for full history.
