from __future__ import annotations

import mlflow
import mlflow.sklearn

from src.data.ingestion import save_train_test_split
from src.data.validation import validate_saved_data
from src.data.transformation import fit_and_save_preprocessor
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.common import load_yaml


def run_training_pipeline(config_path: str = "src/config/config.yaml") -> None:
    """
    Full training pipeline:
    ingestion -> validation -> transformation -> training -> evaluation -> MLflow logging
    """
    config = load_yaml(config_path)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Data pipeline
        train_path, test_path = save_train_test_split(config_path)
        validate_saved_data(config_path)
        preprocessor_path = fit_and_save_preprocessor(config_path)

        # Model pipeline
        model_path = train_model(config_path)
        metrics = evaluate_model(config_path)

        # Log params
        mlflow.log_param("model_type", config["model"]["type"])
        for key, value in config["model"]["params"].items():
            mlflow.log_param(key, value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            if metric_name != "classification_report":
                mlflow.log_metric(metric_name, metric_value)

        # Log artifacts
        mlflow.log_artifact(str(train_path))
        mlflow.log_artifact(str(test_path))
        mlflow.log_artifact(str(preprocessor_path))
        mlflow.log_artifact(str(model_path))

        print("Training pipeline completed successfully.")
        print("Logged experiment to MLflow.")


if __name__ == "__main__":
    run_training_pipeline()