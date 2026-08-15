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
from .state import StageLedger
from .utils import atomic_json, get_logger, seed_everything

logger = get_logger("omega.pipeline")


def _stage_sort_key(relative_name: str) -> tuple[int, str, str]:
    fold, label, model = relative_name.split("/")
    return (int(fold), label, model)


def run_pipeline(df, cfg, artifact_dir="artifacts/run"):
    """Run the research pipeline with per-fold checkpointing.

    Every fold/label/model stage writes its metrics and predictions atomically
    before the next stage starts, and a run ledger records completion. If the
    process dies mid-run, restarting resumes completed stages instead of
    recomputing them.
    """
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
    if merged.empty:
        raise ValueError(
            "No rows survived the warm-up (features), label horizon (labels), and dropna merge; "
            "the dataset is too short or all values are missing"
        )
    all_nan_features=[c for c in feature_cols if merged[c].isna().all()]
    if all_nan_features:
        logger.warning("Features are entirely NaN in the merged panel and carry no signal: %s", all_nan_features)
    atomic_json(art/"trainable_panel.json",{
        "rows": int(len(merged)),
        "feature_columns": feature_cols,
        "all_nan_features": all_nan_features,
        "label_columns": LABELS,
    })
    e=cfg["evaluation"]; enabled_models=[name for name,flag in [("logistic",cfg["models"]["logistic"]),("hist_gradient_boosting",cfg["models"]["hist_gradient_boosting"])] if flag]
    if not enabled_models:
        raise ValueError("No models enabled; enable logistic or hist_gradient_boosting in config")

    run_id=f"pipeline-{cfg['project']['seed']}"
    ledger=StageLedger(art/"ledger.json",run_id)
    results_dir=art/"results"; results_dir.mkdir(parents=True,exist_ok=True)
    predictions_dir=art/"predictions"; predictions_dir.mkdir(parents=True,exist_ok=True)
    skips_dir=art/"skips"; skips_dir.mkdir(parents=True,exist_ok=True)
    models_dir=art/"models"; models_dir.mkdir(parents=True,exist_ok=True)

    folds=list(walk_forward_splits(len(merged),e["train_bars"],e["test_bars"],e["step_bars"],e["embargo_bars"],e["calibration_fraction"]))
    if not folds:
        raise ValueError(
            f"Dataset has {len(merged)} valid rows, too few for train_bars={e['train_bars']}, "
            f"test_bars={e['test_bars']}, embargo_bars={e['embargo_bars']}; no walk-forward folds fit"
        )

    evaluated=0
    for fold,split in enumerate(folds):
      for label in LABELS:
       for name in enabled_models:
        stage_key=f"{fold}/{label}/{name}"
        result_path=results_dir/f"{stage_key}.json"
        status=ledger.status(stage_key)
        if status=="complete":
            if not result_path.exists():
                raise RuntimeError(f"Ledger says complete but results missing: {result_path}")
            evaluated+=1
            continue
        if status=="skipped":
            continue
        ledger.update(stage_key,"running")
        try:
            train,cal,test=merged.iloc[split.train],merged.iloc[split.calibration],merged.iloc[split.test]
            if train[label].nunique()<2 or cal[label].nunique()<2:
                reason="insufficient_classes"
                logger.warning("fold=%d label=%s model=%s skipped: fewer than 2 classes in train or calibration",fold,label,name)
                ledger.update(stage_key,"skipped",reason=reason)
                atomic_json(skips_dir/f"{stage_key}.json",{"fold":fold,"label":label,"model":name,"reason":reason})
                continue
            model=make_model(name,cfg["project"]["seed"]+fold).fit(train[feature_cols],train[label],cal[feature_cols],cal[label],e["alpha"])
            p=model.predict_proba(test[feature_cols]); m=probability_metrics(test[label],p); m["ece"]=expected_calibration_error(test[label],p)
            prevalence=float(train[label].mean()); baseline=np.full(len(test),prevalence)
            baseline_metrics=probability_metrics(test[label],baseline)
            m["train_prevalence"]=prevalence; m["baseline_brier"]=baseline_metrics["brier"]
            m["brier_skill_vs_prevalence"]=1-m["brier"]/m["baseline_brier"] if m["baseline_brier"] else np.nan
            m["fold"]=fold; m["label"]=label; m["model"]=name
            atomic_json(result_path,m)
            prediction_frame=pd.DataFrame({"timestamp":test.timestamp,"fold":fold,"label":label,"model":name,"y":test[label],"p":p,"forward_return":test.forward_return})
            prediction_path=predictions_dir/f"{stage_key}.parquet"
            prediction_path.parent.mkdir(parents=True,exist_ok=True)
            prediction_frame.to_parquet(prediction_path,index=False)
            joblib.dump(model,models_dir/f"model_{label}_{name}_fold{fold}.joblib")
            ledger.update(stage_key,"complete")
            evaluated+=1
        except Exception as exc:
            ledger.update(stage_key,"failed",error_type=type(exc).__name__,error=str(exc)[:2000])
            raise

    result_files=sorted(results_dir.rglob("*.json"),key=lambda f:_stage_sort_key(f.relative_to(results_dir).as_posix()))
    metrics=pd.DataFrame([json.loads(f.read_text(encoding="utf-8")) for f in result_files]) if result_files else pd.DataFrame()
    prediction_files=sorted(predictions_dir.rglob("*.parquet"),key=lambda f:_stage_sort_key(f.relative_to(predictions_dir).as_posix()))
    predictions=pd.concat([pd.read_parquet(f) for f in prediction_files],ignore_index=True) if prediction_files else pd.DataFrame()
    skip_files=sorted(skips_dir.rglob("*.json"))
    skips=[json.loads(f.read_text(encoding="utf-8")) for f in skip_files]
    if metrics.empty:
        logger.warning("No fold produced metrics; all combinations were skipped (%d skips)",len(skips))
    atomic_json(art/"skips.json",{"items":skips})
    metrics.to_csv(art/"metrics.csv",index=False); predictions.to_parquet(art/"predictions.parquet",index=False)
    logger.info("pipeline complete: %d (fold,label,model) stages evaluated, %d skipped, artifacts in %s",evaluated,len(skips),art)

    attrs=[]
    if not predictions.empty:
      for (label,name),g in predictions.groupby(["label","model"]): attrs.append(attribution(label,g.y.to_numpy(),g.p.to_numpy(),e["abstain_below"])|{"model":name})
      trend=predictions[(predictions.label=="trend_ignition") & (predictions.model=="hist_gradient_boosting")]
      if len(trend):
        bt_cfg=cfg.get("backtest",{})
        _, bt=hypothetical_state_response(trend.p,trend.forward_return,e["abstain_below"],bt_cfg.get("spread_bps",1.0),bt_cfg.get("slippage_bps",0.5),bt_cfg.get("annualization_bars",12480),label_cfg["horizon_bars"],bt_cfg.get("one_bar_latency",True)); atomic_json(art/"hypothetical_backtest.json",bt)
    atomic_json(art/"attribution.json",{"items":attrs}); return metrics
