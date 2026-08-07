from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config import load_config
from omega.data import synthetic_fx, normalize_to_parquet
from omega.pipeline import run_pipeline

if __name__ == "__main__":
    cfg=load_config(ROOT / "config.yaml"); df=synthetic_fx(n=8000,seed=cfg["project"]["seed"])
    normalize_to_parquet(df,ROOT / "artifacts/data/synthetic/EURUSD_30m.parquet","synthetic","EURUSD")
    metrics=run_pipeline(df,cfg,ROOT / "artifacts/demo")
    print(metrics.groupby(["label","model"])[["brier","average_precision","ece"]].mean().round(4))
    print("Synthetic academic demo complete: artifacts/demo")
