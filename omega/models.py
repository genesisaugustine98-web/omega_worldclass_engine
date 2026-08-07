from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

def model_catalog(seed=42):
    return {
      "logistic": Pipeline([("impute",SimpleImputer()),("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed))]),
      "hist_gradient_boosting": Pipeline([("impute",SimpleImputer()),("model",HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed))])}

@dataclass
class CalibratedBinaryModel:
    estimator: object
    calibrator: object | None = None
    residual_quantile: float = .5
    def fit(self,X,y,X_cal,y_cal,alpha=.1):
        self.estimator.fit(X,y); raw=self.estimator.predict_proba(X_cal)[:,1]
        self.calibrator=IsotonicRegression(out_of_bounds="clip").fit(raw,y_cal) if len(np.unique(y_cal))>1 else None
        p=self.predict_proba(X_cal); self.residual_quantile=float(np.quantile(np.abs(y_cal-p),1-alpha,method="higher")); return self
    def predict_proba(self,X):
        p=self.estimator.predict_proba(X)[:,1]; return self.calibrator.predict(p) if self.calibrator is not None else p
    def prediction_interval(self,X):
        p=self.predict_proba(X); q=self.residual_quantile; return np.c_[np.clip(p-q,0,1),np.clip(p+q,0,1)]

def make_model(name, seed=42): return CalibratedBinaryModel(clone(model_catalog(seed)[name]))

# CONVERSATION_HOOK: Add TFT/TabPFN adapters only behind the same temporal split and calibration interface.
