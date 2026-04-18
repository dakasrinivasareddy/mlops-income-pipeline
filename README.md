#  Income Prediction Pipeline

This project is a complete, production-style machine learning system that predicts whether a person earns more than $50K per year based on demographic and work-related attributes.

Instead of just training a model, this project focuses on how machine learning systems are actually built, tracked, and deployed in real-world environments.

---

##  What this project demonstrates

- End-to-end ML pipeline (data → model → deployment)
- Experiment tracking using MLflow
- Model serving with FastAPI
- Batch and real-time prediction APIs
- Docker-based deployment
- Automated testing with pytest
- CI-ready structure

---

##  MLflow Experiment Tracking

### Experiment Runs


::contentReference[oaicite:0]{index=0}


Each run represents a different training execution.  
MLflow tracks metrics like accuracy, precision, recall, and F1-score across runs, making it easy to compare model performance.

---

### Run Details (Metrics + Parameters)


::contentReference[oaicite:1]{index=1}


Inside each run, you can see:
- model configuration (XGBoost parameters)
- evaluation metrics (~0.87–0.88 accuracy)
- artifacts like trained model and preprocessor

This is how real ML systems track reproducibility.

---

## API Interface (FastAPI)


::contentReference[oaicite:2]{index=2}


The model is exposed via a production-style API.

### Available endpoints:

- `GET /` → basic info
- `GET /health` → health check
- `POST /predict` → single prediction
- `POST /predict-batch` → batch predictions

---

## Project Structure

ml-platform/
├── data/
├── artifacts/
├── src/
│   ├── data/
│   ├── models/
│   ├── pipelines/
│   └── utils/
├── api/
├── tests/
├── assets/
├── .github/workflows/
├── Dockerfile
├── requirements.txt
└── README.md

## Pipeline Flow

Raw Data
   ↓
Validation
   ↓
Preprocessing
   ↓
Model Training
   ↓
Evaluation
   ↓
MLflow Tracking
   ↓
FastAPI Deployment
   ↓
Batch / Real-time Predictions