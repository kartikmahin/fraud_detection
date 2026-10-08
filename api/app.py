"""
FastAPI Application for Fraud Detection (PyTorch)
REST API with CORS for React frontend integration
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import json
import pickle
import torch
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import FraudDetector, LSTMFraudDetector


app = FastAPI(
    title="Fraud Detection API (PyTorch)",
    description="Deep Learning API for detecting fraudulent banking transactions",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class Transaction(BaseModel):
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in INR")
    transaction_time: int = Field(..., ge=0, le=86399, description="Seconds since midnight")
    location: str = Field(..., description="City code")
    device_id: str = Field(..., description="Device identifier")
    merchant_category: str = Field(..., description="Category: retail, grocery, restaurant, gas, online")
    account_age_days: int = Field(..., ge=0, description="Age of account in days")
    transaction_count_24h: int = Field(..., ge=0, description="Transactions in last 24 hours")
    avg_transaction_amount: float = Field(..., gt=0, description="Average transaction amount")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    message: str


class BatchPredictionRequest(BaseModel):
    transactions: List[Transaction]
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int


# --- Global state ---
detector = None
lstm_detector = None
training_history = None
training_history_lstm = None
eval_metrics = None
eval_metrics_lstm = None


@app.on_event("startup")
async def load_model():
    """Load PyTorch models (MLP + LSTM) and scaler on startup"""
    global detector, lstm_detector, training_history, training_history_lstm, eval_metrics, eval_metrics_lstm
    
    # --- MLP Model ---
    detector = FraudDetector()
    
    model_path = Path("models/fraud_detector.pt")
    if model_path.exists():
        try:
            detector.load_model()
            detector.load_scaler()
            detector.load_feature_names()
            print("PyTorch MLP model loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading MLP model: {e}")
            detector.model = None
    else:
        print("Warning: No trained MLP model found. Train first via /train endpoint or CLI.")
    
    # --- LSTM Model ---
    lstm_detector = LSTMFraudDetector()
    
    lstm_model_path = Path("models/fraud_detector_lstm.pt")
    if lstm_model_path.exists():
        try:
            lstm_detector.load_model()
            lstm_detector.load_scaler()
            lstm_detector.load_feature_names()
            print("PyTorch LSTM model loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading LSTM model: {e}")
            lstm_detector.model = None
    else:
        print("Warning: No trained LSTM model found. Train via /train-lstm endpoint.")
    
    # Load MLP training history
    history_path = Path("models/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # Load LSTM training history
    lstm_history_path = Path("models/training_history_lstm.json")
    if lstm_history_path.exists():
        with open(lstm_history_path, 'r') as f:
            training_history_lstm = json.load(f)
    
    # Load MLP eval metrics
    metrics_path = Path("models/eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
    
    # Load LSTM eval metrics
    lstm_metrics_path = Path("models/eval_metrics_lstm.json")
    if lstm_metrics_path.exists():
        with open(lstm_metrics_path, 'r') as f:
            eval_metrics_lstm = json.load(f)


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Fraud Detection API (PyTorch)",
        "version": "2.0.0",
        "model_type": "Deep Neural Network",
        "framework": "PyTorch",
        "endpoints": {
            "predict": "/predict",
            "predict_lstm": "/predict-lstm",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "model_info": "/model-info",
            "model_info_lstm": "/model-info-lstm",
            "training_history": "/training-history",
            "metrics": "/metrics",
            "train": "/train",
            "train_lstm": "/train-lstm",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None,
        "lstm_model_loaded": lstm_detector is not None and lstm_detector.model is not None,
        "device": str(detector.device) if detector else "N/A",
        "framework": "PyTorch"
    }


@app.get("/model-info")
async def model_info():
    """Get model architecture and info"""
    if detector is None or detector.model is None:
        return {"message": "No model loaded. Train a model first."}
    
    model = detector.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "FraudDetectorNet (Deep Neural Network)",
        "framework": "PyTorch",
        "device": str(detector.device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "architecture": str(model),
        "features_used": detector.feature_names or [],
    }


@app.get("/training-history")
async def get_training_history():
    """Get training metrics history for visualization"""
    global training_history
    
    history_path = Path("models/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history = json.load(f)
        return training_history
    
    return {"message": "No training history available. Train a model first."}


@app.get("/metrics")
async def get_metrics():
    """Get MLP evaluation metrics"""
    global eval_metrics
    
    metrics_path = Path("models/eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
        return eval_metrics
    
    return {"message": "No evaluation metrics available. Train a model first."}


@app.get("/metrics-lstm")
async def get_lstm_metrics():
    """Get LSTM evaluation metrics"""
    global eval_metrics_lstm
    
    metrics_path = Path("models/eval_metrics_lstm.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            eval_metrics_lstm = json.load(f)
        return eval_metrics_lstm
    
    return {"message": "No LSTM evaluation metrics available. Train the LSTM model first."}


@app.get("/training-history-lstm")
async def get_lstm_training_history():
    """Get LSTM training metrics history for visualization"""
    global training_history_lstm
    
    history_path = Path("models/training_history_lstm.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            training_history_lstm = json.load(f)
        return training_history_lstm
    
    return {"message": "No LSTM training history available. Train the LSTM model first."}


@app.post("/predict-receipt", response_model=PredictionResponse)
async def predict_from_receipt(
    file: UploadFile = File(None),
    transaction_amount: Optional[float] = Form(None),
    transaction_time: Optional[int] = Form(None),
    location: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
    merchant_category: Optional[str] = Form(None),
    account_age_days: Optional[int] = Form(None),
    transaction_count_24h: Optional[int] = Form(None),
    avg_transaction_amount: Optional[float] = Form(None),
    model: str = Form("mlp")
) -> PredictionResponse:
    """Predict fraud from an uploaded receipt (image/pdf) or form fields.
    If the receipt file is provided, a real implementation would run OCR to extract
    the required fields. Here we use a placeholder that simply falls back to the
    provided form values and defaults for any missing feature.
    """
    # Default values for missing features (could be tuned from training stats)
    defaults = {
        "transaction_amount": 100.0,
        "transaction_time": 43200,
        "location": "Mumbai",
        "device_id": "UNKNOWN",
        "merchant_category": "retail",
        "account_age_days": 365,
        "transaction_count_24h": 1,
        "avg_transaction_amount": 100.0,
    }
    # Assemble feature dict from supplied form values, falling back to defaults
    data: Dict[str, Any] = {}
    for key, default in defaults.items():
        val = locals().get(key)
        data[key] = val if val is not None else default

    # TODO: extract from `file` using OCR (e.g., pytesseract) – omitted for brevity

    # Choose model
    if model.lower() == "lstm":
        if lstm_detector is None or lstm_detector.model is None:
            raise HTTPException(status_code=503, detail="LSTM model not loaded.")
        result = lstm_detector.predict(data)
    else:
        if detector is None or detector.model is None:
            raise HTTPException(status_code=503, detail="MLP model not loaded.")
        result = detector.predict(data)

    message = ("Transaction flagged as potentially fraudulent" if result["is_fraud"] else "Transaction appears legitimate")
    return PredictionResponse(
        is_fraud=result["is_fraud"],
        fraud_probability=result["fraud_probability"],
        risk_level=result["risk_level"],
        message=message,
    )
@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    
    try:
        transaction_dict = transaction.model_dump()
        device_id = transaction_dict.pop('device_id')
        
        result = detector.predict(transaction_dict)
        
        message = ("Transaction flagged as potentially fraudulent" 
                   if result['is_fraud'] 
                   else "Transaction appears legitimate")
        
        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_fraud(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    predictions = []
    flagged_count = 0
    
    for transaction in request.transactions:
        transaction_dict = transaction.model_dump()
        transaction_dict.pop('device_id')
        
        result = detector.predict(transaction_dict, request.threshold)
        
        if result['is_fraud']:
            flagged_count += 1
            message = "Transaction flagged as potentially fraudulent"
        else:
            message = "Transaction appears legitimate"
        
        predictions.append(PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            message=message
        ))
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_count=flagged_count
    )


@app.post("/train")
async def train_model():
    """Train the PyTorch fraud detection model"""
    global detector, training_history, eval_metrics
    
    try:
        from src.data_pipeline import DataPipeline
        from src.preprocessing import Preprocessor
        from src.train_model import PyTorchTrainer
        from src.pytorch_model import create_data_loaders
        from src.evaluate import ModelEvaluator
        
        # Data pipeline
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.handle_missing_values()
        pipeline.feature_engineering()
        pipeline.encode_categorical()
        X_train, X_test, y_train, y_test = pipeline.prepare_data()
        
        # Preprocessing
        preprocessor = Preprocessor()
        X_train_scaled, X_test_scaled = preprocessor.fit_transform(X_train, X_test)
        
        os.makedirs('models', exist_ok=True)
        pickle.dump(preprocessor.scaler, open('models/scaler.pkl', 'wb'))
        
        feature_names = list(X_train.columns)
        with open('models/feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        # Create data loaders
        input_dim = X_train_scaled.shape[1]
        train_loader, test_loader = create_data_loaders(
            X_train_scaled, y_train.values,
            X_test_scaled, y_test.values,
            batch_size=512
        )
        
        # Train
        trainer = PyTorchTrainer(input_dim)
        trainer.build_model()
        training_history = trainer.train(
            train_loader, test_loader, y_train,
            epochs=50, learning_rate=0.001, patience=10
        )
        trainer.save_model('models/')
        
        # Evaluate
        predictions, probabilities = trainer.predict(X_test_scaled)
        evaluator = ModelEvaluator(y_test.values, predictions, probabilities, 'PyTorch Neural Network')
        metrics = evaluator.calculate_metrics()
        eval_metrics = {k: float(v) for k, v in metrics.items()}
        
        with open('models/eval_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Reload model
        detector = FraudDetector()
        detector.load_model()
        detector.load_scaler()
        detector.load_feature_names()
        
        return {
            "status": "success",
            "message": "PyTorch model trained and saved successfully",
            "metrics": eval_metrics
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/model-info-lstm")
async def lstm_model_info():
    """Get LSTM model architecture and info"""
    if lstm_detector is None or lstm_detector.model is None:
        return {"message": "No LSTM model loaded. Train a model first via /train-lstm."}
    
    model = lstm_detector.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": "FraudLSTMNet (Bidirectional LSTM)",
        "framework": "PyTorch",
        "device": str(lstm_detector.device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "seq_length": lstm_detector.seq_length,
        "architecture": str(model),
        "features_used": lstm_detector.feature_names or [],
    }


@app.post("/predict-lstm", response_model=PredictionResponse)
async def predict_fraud_lstm(transaction: Transaction):
    """Predict if a transaction is fraudulent using the LSTM model"""
    if lstm_detector is None or lstm_detector.model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded. Train via /train-lstm first.")
    
    try:
        transaction_dict = transaction.model_dump()
        device_id = transaction_dict.pop('device_id')
        
        result = lstm_detector.predict(transaction_dict)
        
        message = ("[LSTM] Transaction flagged as potentially fraudulent" 
                   if result['is_fraud'] 
                   else "[LSTM] Transaction appears legitimate")
        
        return PredictionResponse(
            is_fraud=result['is_fraud'],
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM Prediction error: {str(e)}")


@app.post("/train-lstm")
async def train_lstm_model():
    """Train the LSTM fraud detection model"""
    global lstm_detector
    
    try:
        from src.data_pipeline import DataPipeline
        from src.preprocessing import Preprocessor
        from src.train_model import LSTMTrainer
        from src.pytorch_model import create_sequence_data_loaders
        from src.evaluate import ModelEvaluator
        import pandas as pd
        
        # Data pipeline - keep customer_id & transaction_time for sequencing
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.handle_missing_values()
        pipeline.feature_engineering()
        pipeline.encode_categorical()
        
        # We need to split while retaining customer_id & transaction_time
        # for the sequence builder
        from sklearn.model_selection import train_test_split
        
        df = pipeline.df.copy()
        
        # Remove transaction_id & device_id but keep customer_id
        for col in ['transaction_id', 'device_id']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['is_fraud']
        )
        
        # Determine feature columns (everything except identifiers and target)
        exclude_cols = {'customer_id', 'is_fraud', 'transaction_time'}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        # Scale features in-place
        preprocessor = Preprocessor()
        train_df[feature_cols] = preprocessor.fit_transform(
            train_df[feature_cols]
        )
        test_df[feature_cols] = preprocessor.transform(
            test_df[feature_cols]
        )
        
        os.makedirs('models', exist_ok=True)
        pickle.dump(preprocessor.scaler, open('models/scaler.pkl', 'wb'))
        
        with open('models/feature_names_lstm.json', 'w') as f:
            json.dump(feature_cols, f)
        
        # Create sequence data loaders
        seq_length = 10
        train_loader, test_loader, input_dim = create_sequence_data_loaders(
            train_df, test_df, feature_cols,
            seq_length=seq_length, batch_size=512
        )
        
        # Train LSTM
        trainer = LSTMTrainer(input_dim, seq_length=seq_length)
        trainer.build_model()
        y_train = train_df['is_fraud']
        lstm_history = trainer.train(
            train_loader, test_loader, y_train,
            epochs=50, learning_rate=0.001, patience=10
        )
        trainer.save_model('models/')
        
        # Evaluate
        predictions, probabilities = trainer.predict(test_loader)
        y_test_labels = test_df['is_fraud'].values
        
        # The test_loader re-builds sequences, so label count matches dataset length
        # Use the labels from the test_loader's dataset directly
        test_labels_from_loader = test_loader.dataset.labels.numpy()
        
        evaluator = ModelEvaluator(
            test_labels_from_loader, predictions, probabilities,
            'LSTM Neural Network'
        )
        metrics = evaluator.calculate_metrics()
        lstm_eval_metrics = {k: float(v) for k, v in metrics.items()}
        
        with open('models/eval_metrics_lstm.json', 'w') as f:
            json.dump(lstm_eval_metrics, f, indent=2)
        
        # Reload LSTM model
        lstm_detector = LSTMFraudDetector()
        lstm_detector.load_model()
        lstm_detector.load_scaler()
        lstm_detector.load_feature_names()
        
        return {
            "status": "success",
            "message": "LSTM model trained and saved successfully",
            "metrics": lstm_eval_metrics
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LSTM Training error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
