from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.common import ensure_dir, load_yaml


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical columns.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def fit_and_save_preprocessor(config_path: str = "src/config/config.yaml") -> Path:
    """
    Fit preprocessor on training data and save it.
    """
    config = load_yaml(config_path)

    raw_dir = Path(config["data"]["raw_dir"])
    artifacts_dir = ensure_dir(config["artifacts"]["model_dir"])

    train_df = pd.read_csv(raw_dir / config["data"]["train_file"])
    target_column = config["data"]["target_column"]

    X_train = train_df.drop(columns=[target_column])

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    preprocessor_path = artifacts_dir / config["artifacts"]["preprocessor_file"]
    joblib.dump(preprocessor, preprocessor_path)

    return preprocessor_path


if __name__ == "__main__":
    path = fit_and_save_preprocessor()
    print(f"Saved preprocessor to: {path}")