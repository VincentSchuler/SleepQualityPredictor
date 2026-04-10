from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURES, FORM_DEFAULTS


def reduce_nap(x: float) -> str:
    if x == 0:
        return "No nap"
    if x <= 15:
        return "15 minutes"
    if x <= 30:
        return "30 minutes"
    if x <= 45:
        return "45 minutes"
    if x <= 60:
        return "60 minutes"
    return "More than 1 hour"


def reduce_screen_time(x: float) -> str:
    if x <= 30:
        return "Less than 30 minutes"
    if x <= 60:
        return "30-60 minutes"
    if x <= 120:
        return "1-2 hours"
    if x <= 180:
        return "2-3 hours"
    return "More than 3 hours"


def reduce_coffee(x: float) -> int:
    if x <= 40:
        return 0
    if x <= 100:
        return 1
    if x <= 200:
        return 2
    return 3


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "mental_health_condition" in df.columns:
        df["Anxiety"] = df["mental_health_condition"].isin(["Anxiety", "Both"]).astype(int)
        df["Depression"] = df["mental_health_condition"].isin(["Depression", "Both"]).astype(int)

    if "nap_duration_mins" in df.columns and "nap_time" not in df.columns:
        df["nap_time"] = df["nap_duration_mins"].apply(reduce_nap)

    if "screen_time_before_bed_mins" in df.columns and "time_screen_before_sleep" not in df.columns:
        df["time_screen_before_sleep"] = df["screen_time_before_bed_mins"].apply(reduce_screen_time)

    if "caffeine_mg_before_bed" in df.columns and "nb_cafe_before_bed" not in df.columns:
        df["nb_cafe_before_bed"] = df["caffeine_mg_before_bed"].apply(reduce_coffee)

    return df


def _ensure_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = FORM_DEFAULTS[feature]
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engineered_features(df)
    df = _ensure_feature_defaults(df)
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def build_single_input(payload: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    values = FORM_DEFAULTS.copy()
    if payload:
        values.update(payload)
    df = pd.DataFrame([values])
    return prepare_features(df)
