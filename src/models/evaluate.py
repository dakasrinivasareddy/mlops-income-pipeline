from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.utils.common import load_yaml, save_json


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


def evaluate_model(config_path: str = "src/config/config.yaml") -> dict:
    """
    Evaluate trained model on test data.
    """
    config = load_yaml(config_path)

    raw_dir = Path(config["data"]["raw_dir"])
    artifacts_dir = Path(config["artifacts"]["model_dir"])
    target_column = config["data"]["target_column"]

    test_df = pd.read_csv(raw_dir / config["data"]["test_file"])

    preprocessor = joblib.load(artifacts_dir / config["artifacts"]["preprocessor_file"])
    model = joblib.load(artifacts_dir / config["artifacts"]["model_file"])

    X_test = test_df.drop(columns=[target_column])
    y_test = encode_target(test_df[target_column])

    if y_test.isnull().any():
        raise ValueError("Target encoding failed. Found unknown class labels in test data.")

    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    metrics_path = artifacts_dir / config["artifacts"]["metrics_file"]
    save_json(metrics, metrics_path)

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if key != "classification_report":
            print(f"{key}: {value}")