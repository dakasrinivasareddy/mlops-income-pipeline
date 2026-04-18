from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.common import ensure_dir, load_yaml


def build_model(config: dict):
    """
    Build model based on config.
    """
    model_type = config["model"]["type"].lower()
    params = config["model"]["params"]

    if model_type == "xgboost":
        model = XGBClassifier(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def encode_target(y: pd.Series) -> pd.Series:
    """
    Convert Adult income labels to numeric classes.
    <=50K -> 0
    >50K  -> 1
    """
    mapping = {
        "<=50K": 0,
        ">50K": 1,
    }
    return y.map(mapping)


def train_model(config_path: str = "src/config/config.yaml") -> Path:
    """
    Train model using saved train data and preprocessor artifact.
    """
    config = load_yaml(config_path)

    raw_dir = Path(config["data"]["raw_dir"])
    artifacts_dir = ensure_dir(config["artifacts"]["model_dir"])
    target_column = config["data"]["target_column"]

    train_df = pd.read_csv(raw_dir / config["data"]["train_file"])
    preprocessor = joblib.load(artifacts_dir / config["artifacts"]["preprocessor_file"])

    X_train = train_df.drop(columns=[target_column])
    y_train = encode_target(train_df[target_column])

    if y_train.isnull().any():
        raise ValueError("Target encoding failed. Found unknown class labels in training data.")

    X_train_transformed = preprocessor.transform(X_train)

    model = build_model(config)
    model.fit(X_train_transformed, y_train)

    model_path = artifacts_dir / config["artifacts"]["model_file"]
    joblib.dump(model, model_path)

    return model_path


if __name__ == "__main__":
    path = train_model()
    print(f"Saved model to: {path}")