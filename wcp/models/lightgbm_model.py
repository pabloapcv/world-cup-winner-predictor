"""Gradient boosting classifier for match outcomes (sklearn HistGradientBoosting)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

from wcp.config import MODELS_DIR
from wcp.features import FEATURE_COLS


class LightGBMPredictor:
    """HistGradientBoosting-based predictor (no native OpenMP dependency)."""

    def __init__(self):
        self.model: CalibratedClassifierCV | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMPredictor":
        base = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=42,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        self.model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
        self.model.fit(X[FEATURE_COLS], y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> tuple[float, float, float]:
        if self.model is None:
            raise RuntimeError("Model not trained")
        probs = self.model.predict_proba(X[FEATURE_COLS])[0]
        return float(probs[0]), float(probs[1]), float(probs[2])

    def feature_importance(self, X: pd.DataFrame) -> pd.Series:
        """Proxy importance via correlation with predictions (portfolio viz)."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        base = self.model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            return pd.Series(base.feature_importances_, index=FEATURE_COLS)
        return pd.Series(1.0 / len(FEATURE_COLS), index=FEATURE_COLS)

    def save(self, path: Path | None = None) -> None:
        path = path or MODELS_DIR / "lightgbm.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path | None = None) -> "LightGBMPredictor":
        path = path or MODELS_DIR / "lightgbm.joblib"
        self.model = joblib.load(path)
        return self
