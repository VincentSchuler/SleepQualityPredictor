from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_DIRECTIONS,
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    RISK_DESCRIPTIONS,
    RISK_LABELS,
    SHAP_EXPLAINER_PATH,
)
from src.preprocess import prepare_features

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


@dataclass
class PredictionResult:
    raw_score: float
    risk_class: int
    risk_label: str
    risk_description: str
    probabilities_like: Dict[str, float]
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]


class SleepRiskPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.metadata = {}
        self._explainer = None
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Place your exported .joblib model in the models/ folder."
            )
        self.model = joblib.load(self.model_path)
        if self.metadata_path.exists():
            self.metadata = joblib.load(self.metadata_path)
        if SHAP_EXPLAINER_PATH.exists():
            self._explainer = joblib.load(SHAP_EXPLAINER_PATH)

    def predict_dataframe(self, df: pd.DataFrame) -> PredictionResult:
        X = prepare_features(df)
        raw_score = float(np.asarray(self.model.predict(X))[0])
        risk_class = int(np.clip(np.rint(raw_score), 1, 4))
        contributions = self.explain_dataframe(X)
        top_positive = [c for c in contributions if c["impact"] > 0][:5]
        top_negative = [c for c in contributions if c["impact"] < 0][:5]

        return PredictionResult(
            raw_score=raw_score,
            risk_class=risk_class,
            risk_label=RISK_LABELS[risk_class],
            risk_description=RISK_DESCRIPTIONS[risk_class],
            probabilities_like=self.score_bands(raw_score),
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
        )

    def explain_dataframe(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        shap_values = None

        if shap is not None:
            try:
                explainer = self._explainer or shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
            except Exception:
                shap_values = None

        if shap_values is not None:
            values = np.asarray(shap_values)
            if values.ndim == 2:
                row = values[0]
            elif values.ndim == 3:
                row = values[0, :, 0]
            else:
                row = np.ravel(values)[: len(FEATURES)]
            contributions = []
            for feat, value, impact in zip(FEATURES, X.iloc[0].tolist(), row):
                contributions.append(
                    {
                        "feature": feat,
                        "value": value,
                        "impact": float(impact),
                        "direction": "increase" if impact > 0 else "decrease",
                        "method": "shap",
                    }
                )
            contributions.sort(key=lambda d: abs(d["impact"]), reverse=True)
            return contributions

        return self._heuristic_explanation(X)

    def _heuristic_explanation(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        medians = self.metadata.get("numeric_medians", {})
        category_modes = self.metadata.get("category_modes", {})
        contributions = []
        row = X.iloc[0].to_dict()

        for feat in FEATURES:
            value = row[feat]
            if feat in medians:
                baseline = medians[feat]
                delta = float(value) - float(baseline)
                direction_factor = FEATURE_DIRECTIONS.get(feat, 0)
                impact = direction_factor * delta
            elif feat in category_modes:
                impact = 0.8 if value != category_modes[feat] else -0.2
            else:
                impact = 0.0
            contributions.append(
                {
                    "feature": feat,
                    "value": value,
                    "impact": float(impact),
                    "direction": "increase" if impact > 0 else "decrease",
                    "method": "heuristic",
                }
            )

        contributions.sort(key=lambda d: abs(d["impact"]), reverse=True)
        return contributions

    @staticmethod
    def score_bands(raw_score: float) -> Dict[str, float]:
        centers = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
        scores = {k: np.exp(-abs(raw_score - v)) for k, v in centers.items()}
        total = sum(scores.values())
        return {RISK_LABELS[k]: float(v / total) for k, v in scores.items()}


def load_predictor() -> SleepRiskPredictor:
    return SleepRiskPredictor()
