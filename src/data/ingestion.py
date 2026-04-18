from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.utils.common import ensure_dir, load_yaml


def load_raw_data() -> pd.DataFrame:
    """
    Load the Adult dataset from OpenML.
    """
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    df = dataset.frame.copy()

    # Standardize target column name
    df = df.rename(columns={"class": "income"})
    return df


def save_train_test_split(config_path: str = "src/config/config.yaml") -> tuple[Path, Path]:
    """
    Load data, split into train/test, and save to CSV.
    """
    config = load_yaml(config_path)

    raw_dir = ensure_dir(config["data"]["raw_dir"])
    train_path = raw_dir / config["data"]["train_file"]
    test_path = raw_dir / config["data"]["test_file"]

    df = load_raw_data()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config["project"]["random_state"],
        stratify=df[config["data"]["target_column"]],
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


if __name__ == "__main__":
    train_path, test_path = save_train_test_split()
    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to: {test_path}")