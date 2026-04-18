import logging
import os
from datetime import datetime, UTC

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from api.schemas import (
    IncomePredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
)
from src.models.predict import PredictionPipeline

load_dotenv()

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Income Prediction API",
    version="2.0",
    description="Production-style ML API for income classification",
)

pipeline = PredictionPipeline()


def map_prediction(prediction: int) -> str:
    return ">50K" if prediction == 1 else "<=50K"


def validate_request_data(age: int, hours_per_week: int, education_num: int) -> None:
    if age < 0:
        raise HTTPException(status_code=400, detail="Invalid age. Age must be non-negative.")
    if hours_per_week < 0:
        raise HTTPException(status_code=400, detail="Invalid hours_per_week. Must be non-negative.")
    if education_num < 0:
        raise HTTPException(status_code=400, detail="Invalid education_num. Must be non-negative.")


@app.get("/")
def root():
    return {
        "message": "Income Prediction API is running",
        "model_version": MODEL_VERSION,
        "endpoints": ["/", "/health", "/predict", "/predict-batch", "/docs"],
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: IncomePredictionRequest):
    validate_request_data(
        age=request.age,
        hours_per_week=request.hours_per_week,
        education_num=request.education_num,
    )

    logger.info(f"Received single prediction request: {request.model_dump()}")

    input_data = [request.model_dump()]
    prediction = pipeline.predict(input_data)[0]

    logger.info(f"Single prediction output: {prediction}")

    return PredictionResponse(
        prediction=int(prediction),
        label=map_prediction(int(prediction)),
        timestamp=datetime.now(UTC).isoformat(),
        model_version=MODEL_VERSION,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one record.")

    for record in request.records:
        validate_request_data(
            age=record.age,
            hours_per_week=record.hours_per_week,
            education_num=record.education_num,
        )

    input_data = [record.model_dump() for record in request.records]

    logger.info(f"Received batch prediction request with {len(input_data)} records")

    predictions = pipeline.predict(input_data)

    results = [
        BatchPredictionItem(
            prediction=int(pred),
            label=map_prediction(int(pred)),
            timestamp=datetime.now(UTC).isoformat(),
            model_version=MODEL_VERSION,
        )
        for pred in predictions
    ]

    logger.info(f"Batch prediction completed for {len(results)} records")

    return BatchPredictionResponse(
        predictions=results,
        total_records=len(results),
    )