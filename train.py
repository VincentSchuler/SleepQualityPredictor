from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from src.config import METADATA_PATH, MODEL_PATH
from src.preprocess import add_engineered_features, prepare_features


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sleep_disorder_risk" not in df.columns:
        raise ValueError("The training dataset must contain a 'sleep_disorder_risk' column.")

    dict_y = {"Healthy": 1, "Mild": 2, "Moderate": 3, "Severe": 4}
    df["y"] = df["sleep_disorder_risk"].map(dict_y).astype(int)
    return df


def train_and_export(data_path: str | Path, export_dir: str | Path = "models") -> None:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = add_engineered_features(df)
    df = build_target(df)
    X = prepare_features(df)
    y = df["y"]

    fit_params = {
        "eval_metric": "mae",
        "eval_names": ["test"],
        "categorical_feature": "auto",
    }

    model_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 20,
        "n_estimators": 10000,
        "n_jobs": -1,
        "learning_rate": 0.05,
        "reg_alpha": 0.9,
        "reg_lambda": 9,
        "max_depth": 10,
        "colsample_bytree": 0.75,
        "subsample": 0.75,
        "min_child_samples": 50,
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=48)
    iterations = []
    oof = np.zeros(len(df))

    print("Running cross-validation...")
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        xtr, xva = X.iloc[train_idx], X.iloc[valid_idx]
        ytr, yva = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            xtr,
            ytr,
            eval_set=[(xva, yva)],
            **fit_params,
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)],
        )

        pred = model.predict(xva)
        oof[valid_idx] = pred
        iterations.append(model.n_iter_)
        mae = mean_absolute_error(yva, pred)
        cls = np.clip(np.rint(pred), 1, 4).astype(int)
        acc = accuracy_score(yva, cls)
        f1 = f1_score(yva, cls, average="macro")
        print(f"Fold {fold}: MAE={mae:.4f} | Accuracy={acc:.4f} | F1-macro={f1:.4f} | best_iter={model.n_iter_}")

    print("\nOverall CV metrics")
    overall_mae = mean_absolute_error(y, oof)
    overall_cls = np.clip(np.rint(oof), 1, 4).astype(int)
    overall_acc = accuracy_score(y, overall_cls)
    overall_f1 = f1_score(y, overall_cls, average="macro")
    print(f"MAE={overall_mae:.4f} | Accuracy={overall_acc:.4f} | F1-macro={overall_f1:.4f}")

    full_params = model_params.copy()
    full_params["n_estimators"] = int(np.median(iterations))

    final_model = lgb.LGBMRegressor(**full_params)
    final_model.fit(X, y, eval_set=[(X, y)], **fit_params)
    joblib.dump(final_model, export_dir / MODEL_PATH.name)

    metadata = {
        "features": list(X.columns),
        "numeric_medians": X.select_dtypes(include=["number"]).median().to_dict(),
        "category_modes": {
            col: X[col].mode().iloc[0]
            for col in X.select_dtypes(include=["category", "object"]).columns
            if not X[col].mode().empty
        },
        "category_options": {
            col: sorted(X[col].astype(str).dropna().unique().tolist())
            for col in X.select_dtypes(include=["category", "object"]).columns
        },
        "cv": {
            "mae": float(overall_mae),
            "accuracy": float(overall_acc),
            "f1_macro": float(overall_f1),
            "median_best_iteration": int(np.median(iterations)),
        },
    }
    joblib.dump(metadata, export_dir / METADATA_PATH.name)

    try:
        import shap

        explainer = shap.TreeExplainer(final_model)
        joblib.dump(explainer, export_dir / "shap_explainer.joblib")
        print("Saved SHAP explainer.")
    except Exception as exc:
        print(f"SHAP explainer was not saved: {exc}")

    print(f"\nSaved model to: {export_dir / MODEL_PATH.name}")
    print(f"Saved metadata to: {export_dir / METADATA_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the sleep-risk LightGBM model and export artifacts.")
    parser.add_argument("--data", required=True, help="Path to the raw training CSV")
    parser.add_argument("--export-dir", default="models", help="Folder where .joblib artifacts will be saved")
    args = parser.parse_args()
    train_and_export(args.data, args.export_dir)
