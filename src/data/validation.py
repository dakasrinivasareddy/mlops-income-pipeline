from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.common import load_yaml


def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    """
    Basic validation checks for the dataset.
    """
    if df.empty:
        raise ValueError("Dataframe is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    if df[target_column].isnull().any():
        raise ValueError("Target column contains missing values.")


def validate_saved_data(config_path: str = "src/config/config.yaml") -> None:
    """
    Validate train and test CSV files.
    """
    config = load_yaml(config_path)
    raw_dir = Path(config["data"]["raw_dir"])
    target_column = config["data"]["target_column"]

    train_df = pd.read_csv(raw_dir / config["data"]["train_file"])
    test_df = pd.read_csv(raw_dir / config["data"]["test_file"])

    validate_dataframe(train_df, target_column)
    validate_dataframe(test_df, target_column)

    print("Validation passed for train and test datasets.")


if __name__ == "__main__":
    validate_saved_data()