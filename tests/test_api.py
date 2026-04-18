from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert "endpoints" in body
    assert "model_version" in body


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "label" in body
    assert "timestamp" in body
    assert "model_version" in body


def test_predict_batch():
    payload = {
        "records": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            },
            {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlwgt": 83311,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": "United-States"
            }
        ]
    }

    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert "total_records" in body
    assert body["total_records"] == 2


def test_invalid_age():
    payload = {
        "age": -1,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 400