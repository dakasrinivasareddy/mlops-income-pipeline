from pydantic import BaseModel
from typing import List


class IncomePredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    timestamp: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: List[IncomePredictionRequest]


class BatchPredictionItem(BaseModel):
    prediction: int
    label: str
    timestamp: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]
    total_records: int