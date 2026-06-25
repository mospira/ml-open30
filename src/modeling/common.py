from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = "data/processed/dataset_open30m.parquet"
DEFAULT_XGB_DEVICE = os.getenv("OPEN30_XGB_DEVICE", "cuda")

FEATURE_EXCLUDE_PREFIXES = ("date", "ticker", "side", "entry_", "exit_", "horizon_")
FEATURE_EXCLUDE_SUBSTRINGS = ("y_type_m_", "y_R_m_", "y_hit_minute_m_", "y_ambig_m_")

DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": DEFAULT_XGB_DEVICE,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_labeled_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    path = resolve_project_path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_parquet(path)


def extract_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in df.columns:
        if col.startswith(FEATURE_EXCLUDE_PREFIXES):
            continue
        if any(token in col for token in FEATURE_EXCLUDE_SUBSTRINGS):
            continue
        feature_cols.append(col)
    return feature_cols


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n_rows = len(y)
    n_classes = len(classes)
    weights = {cls: n_rows / (n_classes * count) for cls, count in zip(classes, counts)}
    return np.array([weights[value] for value in y])


def load_xgb_params(model_dir: str = "models/v1") -> dict:
    params_path = resolve_project_path(str(Path(model_dir) / "best_params.json"))
    params = DEFAULT_XGB_PARAMS.copy()
    if params_path.exists():
        with params_path.open("r", encoding="utf-8") as f:
            tuned = json.load(f)
        params.update(tuned)
        print(f"Loaded tuned XGBoost params from {params_path}")
    else:
        print("No best_params.json found. Using DEFAULT_XGB_PARAMS.")
    return params
