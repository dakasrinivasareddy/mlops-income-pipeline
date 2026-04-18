from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.common import load_yaml


class PredictionPipeline:
    def __init__(self, config_path: str = "src/config/config.yaml"):
        self.config = load_yaml(config_path)
        artifacts_dir = Path(self.config["artifacts"]["model_dir"])

        self.preprocessor = joblib.load(artifacts_dir / self.config["artifacts"]["preprocessor_file"])
        self.model = joblib.load(artifacts_dir / self.config["artifacts"]["model_file"])

    def predict(self, data: list[dict[str, Any]]) -> list:
        """
        Predict on input records.
        """
        df = pd.DataFrame(data)

        # Match API field names to training column names
        df = df.rename(columns={
            "marital_status": "marital-status",
            "education_num": "education-num",
            "capital_gain": "capital-gain",
            "capital_loss": "capital-loss",
            "hours_per_week": "hours-per-week",
            "native_country": "native-country",
        })

        transformed = self.preprocessor.transform(df)
        predictions = self.model.predict(transformed)
        return predictions.tolist()