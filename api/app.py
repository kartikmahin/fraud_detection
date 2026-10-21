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
from pathlib import Path
import sys
import os

try:
    import torch
except ImportError:
    torch = None

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import FraudDetector, LSTMFraudDetector
from src.receipt_parser import GENERAL_FEATURE_DEFAULTS, extract_and_impute_receipt
from src.rule_engine import RuleEngine
from src.fusion import FraudSignalFusion


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
    device_id: str = Field("DEV001", description="Device identifier")
    merchant_category: str = Field(..., description="Category: retail, grocery, restaurant, gas, online")
    account_age_days: int = Field(..., ge=0, description="Age of account in days")
    transaction_count_24h: int = Field(..., ge=0, description="Transactions in last 24 hours")
    avg_transaction_amount: float = Field(..., gt=0, description="Average transaction amount")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    # IP metadata (optional — rules are skipped if absent)
    ip_country: Optional[str] = Field(None, description="Country code of the IP address")
    ip_is_vpn: Optional[bool] = Field(None, description="Whether connection is via VPN")
    ip_is_proxy: Optional[bool] = Field(None, description="Whether connection is via proxy")
    ip_login_count: Optional[int] = Field(None, ge=0, description="Different accounts on this IP in 24h")
    ip_geo_distance_km: Optional[float] = Field(None, ge=0, description="Distance between IP geo and registered city (km)")
    # Small-amount context (optional)
    small_txn_count_6h: Optional[int] = Field(None, ge=0, description="Count of txns < ₹200 in last 6 hours")
    cumulative_amount_6h: Optional[float] = Field(None, ge=0, description="Running total of small amounts in 6h")
    same_merchant_count_1h: Optional[int] = Field(None, ge=0, description="Txns to same merchant in last hour")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    message: str
    extracted_features: Optional[Dict[str, Any]] = None
    imputed_features: Optional[Dict[str, Any]] = None
    missing_features: Optional[List[str]] = None
    final_features: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    model_used: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response for the /analyze endpoint — ML + rules + fusion."""
    is_fraud: bool
    final_risk_score: float
    risk_level: str
    ml_score: float
    rule_score: float
    fraud_type_indicators: List[str]
    signals: List[Dict[str, Any]]
    positive_signals: List[Dict[str, Any]]
    threshold: float
    message: str
    model_used: str


class ReceiptExtractResponse(BaseModel):
    """Response for receipt OCR extraction only (no prediction)"""
    extracted_features: Dict[str, Any]
    imputed_features: Dict[str, Any]
    missing_features: List[str]
    final_features: Dict[str, Any]
    raw_text: Optional[str] = None
    message: str


class ComplaintRequest(BaseModel):
    description: str = Field(..., min_length=10, description="Description of the fraud")
    name: Optional[str] = Field(None, description="Complainant name")
    phone: Optional[str] = Field(None, description="Complainant phone number")
    email: Optional[str] = Field(None, description="Complainant email")
    fraud_amount: Optional[float] = Field(None, ge=0, description="Amount lost to fraud (INR)")
    transaction_date: Optional[str] = Field(None, description="Date of fraudulent transaction")
    bank_name: Optional[str] = Field(None, description="Bank name involved")
    upi_id: Optional[str] = Field(None, description="UPI ID involved")
    merchant_name: Optional[str] = Field(None, description="Merchant or recipient name")
    transaction_id: Optional[str] = Field(None, description="Transaction reference ID")


class ComplaintResponse(BaseModel):
    complaint_id: str
    message: str
    reference_number: str
    next_steps: List[str]
    portal_url: str


class BatchPredictionRequest(BaseModel):
    transactions: List[Transaction]
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int


def _model_to_dict(model_obj: BaseModel) -> Dict[str, Any]:
    """Compatible dict conversion for both Pydantic v1 and v2."""
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


# --- Global state ---
detector: Optional[FraudDetector] = None
lstm_detector: Optional[LSTMFraudDetector] = None
training_history = None
training_history_lstm = None
eval_metrics = None
eval_metrics_lstm = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Rule engine and fusion engine (stateless singletons)
rule_engine = RuleEngine()
fusion_engine = FraudSignalFusion(ml_weight=0.6)


def _ensure_models_loaded():
    """Ensure MLP and LSTM detectors are initialized and loaded if model files exist."""
    global detector, lstm_detector
    if detector is None:
        detector = FraudDetector(model_path=str(MODELS_DIR))
        model_path = MODELS_DIR / "fraud_detector.pt"
        if model_path.exists():
            try:
                detector.load_model()
                detector.load_scaler()
                detector.load_feature_names()
            except Exception as e:
                print(f"Warning: Error loading MLP model: {e}")
                detector.model = None

    if lstm_detector is None:
        lstm_detector = LSTMFraudDetector(model_path=str(MODELS_DIR))
        lstm_model_path = MODELS_DIR / "fraud_detector_lstm.pt"
        if lstm_model_path.exists():
            try:
                lstm_detector.load_model()
                lstm_detector.load_scaler()
                lstm_detector.load_feature_names()
            except Exception as e:
                print(f"Warning: Error loading LSTM model: {e}")
                lstm_detector.model = None


@app.on_event("startup")
async def load_model():
    """Load PyTorch models (MLP + LSTM) and scaler on startup"""
    global detector, lstm_detector, training_history, training_history_lstm, eval_metrics, eval_metrics_lstm

    _ensure_models_loaded()

    # Load MLP training history
    history_path = MODELS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            training_history = json.load(f)

    # Load LSTM training history
    lstm_history_path = MODELS_DIR / "training_history_lstm.json"
    if lstm_history_path.exists():
        with open(lstm_history_path, "r") as f:
            training_history_lstm = json.load(f)

    # Load MLP eval metrics
    metrics_path = MODELS_DIR / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            eval_metrics = json.load(f)

    # Load LSTM eval metrics
    lstm_metrics_path = MODELS_DIR / "eval_metrics_lstm.json"
    if lstm_metrics_path.exists():
        with open(lstm_metrics_path, "r") as f:
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
            "predict_receipt": "/predict-receipt",
            "extract_receipt": "/extract-receipt",
            "file_complaint": "/file-complaint",
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
    _ensure_models_loaded()
    return {
        "status": "healthy",
        "model_loaded": detector is not None and detector.model is not None,
        "lstm_model_loaded": lstm_detector is not None and lstm_detector.model is not None,
        "device": str(detector.device) if detector else "N/A",
        "framework": "PyTorch"
    }


# --- Complaint filing storage ---
import json as _json
from datetime import datetime as _dt
COMPLAINTS_FILE = BASE_DIR / "complaints.json"

def _load_complaints():
    if COMPLAINTS_FILE.exists():
        try:
            with open(COMPLAINTS_FILE, "r") as f:
                return _json.load(f)
        except Exception:
            return []
    return []

def _save_complaints(complaints):
    with open(COMPLAINTS_FILE, "w") as f:
        _json.dump(complaints, f, indent=2)


@app.post("/file-complaint", response_model=ComplaintResponse)
async def file_complaint(complaint: ComplaintRequest) -> ComplaintResponse:
    """
    File a fraud complaint. Generates a reference number and stores the complaint.
    Returns the reference number and next steps for the user to follow.
    """
    import uuid
    complaint_id = f"CCP-{_dt.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    reference_number = f"REF-{uuid.uuid4().hex[:8].upper()}"

    complaint_record = {
        "complaint_id": complaint_id,
        "reference_number": reference_number,
        "description": complaint.description,
        "name": complaint.name,
        "phone": complaint.phone,
        "email": complaint.email,
        "fraud_amount": complaint.fraud_amount,
        "transaction_date": complaint.transaction_date,
        "bank_name": complaint.bank_name,
        "upi_id": complaint.upi_id,
        "merchant_name": complaint.merchant_name,
        "transaction_id": complaint.transaction_id,
        "status": "submitted",
        "created_at": _dt.now().isoformat(),
    }

    complaints = _load_complaints()
    complaints.append(complaint_record)
    _save_complaints(complaints)

    next_steps = [
        "Visit the National Cyber Crime Portal at https://cybercrime.gov.in to complete official filing",
        "Keep your complaint reference number for future correspondence",
        "Save all transaction screenshots, bank statements, and communication records",
        "Contact your bank immediately to report the fraud and request a chargeback if applicable",
        "If the fraud involves UPI, report it to your UPI app and bank",
        "Monitor your bank account for any further unauthorized transactions",
    ]

    return ComplaintResponse(
        complaint_id=complaint_id,
        message="Complaint filed successfully. Please complete the official filing on the National Cyber Crime Portal.",
        reference_number=reference_number,
        next_steps=next_steps,
        portal_url="https://cybercrime.gov.in",
    )


@app.get("/model-info")
async def model_info():
    """Get model architecture and info"""
    _ensure_models_loaded()
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

    history_path = MODELS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            training_history = json.load(f)
        return training_history

    return {"message": "No training history available. Train a model first."}


@app.get("/metrics")
async def get_metrics():
    """Get MLP evaluation metrics"""
    global eval_metrics

    metrics_path = MODELS_DIR / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            eval_metrics = json.load(f)
        return eval_metrics

    return {"message": "No evaluation metrics available. Train a model first."}


@app.get("/metrics-lstm")
async def get_lstm_metrics():
    """Get LSTM evaluation metrics"""
    global eval_metrics_lstm

    metrics_path = MODELS_DIR / "eval_metrics_lstm.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            eval_metrics_lstm = json.load(f)
        return eval_metrics_lstm

    return {"message": "No LSTM evaluation metrics available. Train the LSTM model first."}


@app.get("/training-history-lstm")
async def get_lstm_training_history():
    """Get LSTM training metrics history for visualization"""
    global training_history_lstm

    history_path = MODELS_DIR / "training_history_lstm.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            training_history_lstm = json.load(f)
        return training_history_lstm

    return {"message": "No LSTM training history available. Train the LSTM model first."}


@app.post("/extract-receipt", response_model=ReceiptExtractResponse)
async def extract_receipt_features(
    file: Optional[UploadFile] = File(None),
    transaction_amount: Optional[float] = Form(None),
    transaction_time: Optional[int] = Form(None),
    location: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
    merchant_category: Optional[str] = Form(None),
    account_age_days: Optional[int] = Form(None),
    transaction_count_24h: Optional[int] = Form(None),
    avg_transaction_amount: Optional[float] = Form(None),
) -> ReceiptExtractResponse:
    """
    Extract transaction features from an uploaded receipt/bill (image, PDF, text, JSON, CSV).
    This endpoint ONLY performs OCR extraction and feature imputation - NO prediction is made.
    Use this for live-updating form fields when a user uploads a receipt.
    """
    _ensure_models_loaded()

    file_bytes: Optional[bytes] = None
    filename = ""
    content_type = ""

    if file is not None:
        file_bytes = await file.read()
        filename = file.filename or ""
        content_type = file.content_type or ""

    form_fields = {
        "transaction_amount": transaction_amount,
        "transaction_time": transaction_time,
        "location": location,
        "device_id": device_id,
        "merchant_category": merchant_category,
        "account_age_days": account_age_days,
        "transaction_count_24h": transaction_count_24h,
        "avg_transaction_amount": avg_transaction_amount,
    }
    supplied_overrides = {k: v for k, v in form_fields.items() if v is not None}

    # Use form overrides for missing features
    parsed = extract_and_impute_receipt(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
        form_overrides=supplied_overrides,
    )

    final_features = parsed["final_features"]
    extracted_features = parsed["extracted_features"]
    imputed_features = parsed["imputed_features"]
    missing_features = parsed["missing_features"]
    raw_text = parsed["raw_text"]

    extracted_names = list(extracted_features.keys())
    imputed_names = list(imputed_features.keys())

    if file_bytes:
        message = (
            f"Extracted {len(extracted_names)} feature(s) from receipt"
            f"{': ' + ', '.join(extracted_names) if extracted_names else ''}; "
            f"applied general defaults for {len(imputed_names)} missing feature(s)"
        )
    else:
        message = "No file uploaded - used provided form values with general defaults"

    return ReceiptExtractResponse(
        extracted_features=extracted_features,
        imputed_features=imputed_features,
        missing_features=missing_features,
        final_features=final_features,
        raw_text=raw_text,
        message=message,
    )


@app.post("/predict-receipt", response_model=PredictionResponse)
async def predict_from_receipt(
    file: Optional[UploadFile] = File(None),
    transaction_amount: Optional[float] = Form(None),
    transaction_time: Optional[int] = Form(None),
    location: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
    merchant_category: Optional[str] = Form(None),
    account_age_days: Optional[int] = Form(None),
    transaction_count_24h: Optional[int] = Form(None),
    avg_transaction_amount: Optional[float] = Form(None),
    threshold: float = Form(0.5),
    use_form_overrides: bool = Form(False),
    model: str = Form("mlp"),
) -> PredictionResponse:
    """
    Predict whether a transaction is fraudulent from an uploaded bill/receipt
    (image, PDF, text, JSON, or CSV).
    1. Extracts available input features directly from the uploaded receipt.
    2. Assigns general baseline features (`GENERAL_FEATURE_DEFAULTS`) for any
       missing features not present on the receipt.
    3. Feeds the complete feature vector into the MLP model (default) or LSTM model
       to predict whether the transaction is fraud or legitimate.
    """
    _ensure_models_loaded()

    file_bytes: Optional[bytes] = None
    filename = ""
    content_type = ""

    if file is not None:
        file_bytes = await file.read()
        filename = file.filename or ""
        content_type = file.content_type or ""

    form_fields = {
        "transaction_amount": transaction_amount,
        "transaction_time": transaction_time,
        "location": location,
        "device_id": device_id,
        "merchant_category": merchant_category,
        "account_age_days": account_age_days,
        "transaction_count_24h": transaction_count_24h,
        "avg_transaction_amount": avg_transaction_amount,
    }
    supplied_overrides = {k: v for k, v in form_fields.items() if v is not None}

    # When a receipt file is uploaded, missing features default to GENERAL_FEATURE_DEFAULTS
    # unless the caller explicitly requested use_form_overrides=True.
    # When no file is uploaded, any supplied form fields are used and missing ones get general defaults.
    active_overrides = supplied_overrides if (not file_bytes or use_form_overrides) else None

    parsed = extract_and_impute_receipt(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
        form_overrides=active_overrides,
    )

    final_features = parsed["final_features"]
    extracted_features = parsed["extracted_features"]
    imputed_features = parsed["imputed_features"]
    missing_features = parsed["missing_features"]
    raw_text = parsed["raw_text"]

    model_choice = (model or "mlp").lower().strip()
    if model_choice == "lstm":
        if lstm_detector is None or lstm_detector.model is None:
            raise HTTPException(status_code=503, detail="LSTM model not loaded.")
        result = lstm_detector.predict(final_features, threshold=threshold)
    else:
        model_choice = "mlp"
        if detector is None or detector.model is None:
            raise HTTPException(status_code=503, detail="MLP model not loaded.")
        result = detector.predict(final_features, threshold=threshold)

    extracted_names = list(extracted_features.keys())
    imputed_names = list(imputed_features.keys())
    verdict_str = (
        "Transaction flagged as potentially fraudulent"
        if result["is_fraud"]
        else "Transaction appears legitimate"
    )
    if file_bytes:
        message = (
            f"{verdict_str} (Extracted {len(extracted_names)} feature(s) from receipt"
            f"{': ' + ', '.join(extracted_names) if extracted_names else ''}; "
            f"applied general defaults for {len(imputed_names)} missing feature(s))"
        )
    else:
        message = verdict_str

    return PredictionResponse(
        is_fraud=result["is_fraud"],
        fraud_probability=result["fraud_probability"],
        risk_level=result["risk_level"],
        message=message,
        extracted_features=extracted_features,
        imputed_features=imputed_features,
        missing_features=missing_features,
        final_features=final_features,
        raw_text=raw_text,
        model_used=model_choice,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict if a transaction is fraudulent"""
    _ensure_models_loaded()
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        transaction_dict = _model_to_dict(transaction)
        threshold = transaction_dict.pop("threshold", 0.5) or 0.5
        final_features = dict(transaction_dict)
        transaction_dict.pop("device_id", None)

        result = detector.predict(transaction_dict, threshold=threshold)

        message = (
            "Transaction flagged as potentially fraudulent"
            if result["is_fraud"]
            else "Transaction appears legitimate"
        )

        return PredictionResponse(
            is_fraud=result["is_fraud"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            message=message,
            final_features=final_features,
            model_used="mlp",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_fraud(transaction: Transaction):
    """Analyze a transaction using ML model + rule engine + signal fusion.

    This is the recommended endpoint for multi-type fraud detection.
    It runs the ML model for a base probability, then evaluates IP anomaly,
    small-amount pattern, and velocity rules, and fuses all signals into a
    single composite risk score with a full breakdown.

    IP fields and small-amount context fields are optional — if omitted,
    the corresponding rules are simply skipped and the ML score dominates.
    """
    _ensure_models_loaded()

    # Determine which ML model to use (prefer MLP, fall back to LSTM)
    active_detector = None
    model_label = "mlp"
    if detector is not None and detector.model is not None:
        active_detector = detector
        model_label = "mlp"
    elif lstm_detector is not None and lstm_detector.model is not None:
        active_detector = lstm_detector
        model_label = "lstm"

    if active_detector is None:
        raise HTTPException(status_code=503, detail="No ML model loaded. Train a model first.")

    try:
        transaction_dict = _model_to_dict(transaction)
        threshold = transaction_dict.pop("threshold", 0.5) or 0.5

        # Run ML prediction
        ml_input = {
            k: v for k, v in transaction_dict.items()
            if k not in ("device_id",)
        }
        ml_result = active_detector.predict(ml_input, threshold=threshold)
        ml_probability = ml_result["fraud_probability"]

        # Run rule engine on the full transaction (including IP + pattern fields)
        rule_results = rule_engine.evaluate_all(transaction_dict)

        # Fuse ML + rules
        fused = fusion_engine.fuse(ml_probability, rule_results, threshold=threshold)

        # Build message
        if fused.is_fraud:
            type_str = ", ".join(fused.fraud_type_indicators) if fused.fraud_type_indicators else "combined signals"
            message = f"Transaction flagged as potentially fraudulent ({type_str})"
        else:
            message = "Transaction appears legitimate"

        return AnalysisResponse(
            is_fraud=fused.is_fraud,
            final_risk_score=fused.final_risk_score,
            risk_level=fused.risk_level,
            ml_score=fused.ml_score,
            rule_score=fused.rule_score,
            fraud_type_indicators=fused.fraud_type_indicators,
            signals=fused.signals,
            positive_signals=fused.positive_signals,
            threshold=fused.threshold,
            message=message,
            model_used=model_label,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_fraud(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    _ensure_models_loaded()
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    predictions = []
    flagged_count = 0

    for transaction in request.transactions:
        transaction_dict = _model_to_dict(transaction)
        transaction_dict.pop("device_id", None)

        result = detector.predict(transaction_dict, request.threshold)

        if result["is_fraud"]:
            flagged_count += 1
            message = "Transaction flagged as potentially fraudulent"
        else:
            message = "Transaction appears legitimate"

        predictions.append(PredictionResponse(
            is_fraud=result["is_fraud"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            message=message,
            model_used="mlp",
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

        os.makedirs(MODELS_DIR, exist_ok=True)
        pickle.dump(preprocessor.scaler, open(MODELS_DIR / "scaler.pkl", "wb"))

        feature_names = list(X_train.columns)
        with open(MODELS_DIR / "feature_names.json", "w") as f:
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
        trainer.save_model(str(MODELS_DIR) + "/")

        # Evaluate
        predictions, probabilities = trainer.predict(X_test_scaled)
        evaluator = ModelEvaluator(y_test.values, predictions, probabilities, "PyTorch Neural Network")
        metrics = evaluator.calculate_metrics()
        eval_metrics = {k: float(v) for k, v in metrics.items()}

        with open(MODELS_DIR / "eval_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)

        # Reload model
        detector = FraudDetector(model_path=str(MODELS_DIR))
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
    _ensure_models_loaded()
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
    _ensure_models_loaded()
    if lstm_detector is None or lstm_detector.model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded. Train via /train-lstm first.")

    try:
        transaction_dict = _model_to_dict(transaction)
        final_features = dict(transaction_dict)
        transaction_dict.pop("device_id", None)

        result = lstm_detector.predict(transaction_dict)

        message = (
            "[LSTM] Transaction flagged as potentially fraudulent"
            if result["is_fraud"]
            else "[LSTM] Transaction appears legitimate"
        )

        return PredictionResponse(
            is_fraud=result["is_fraud"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            message=message,
            final_features=final_features,
            model_used="lstm",
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
        from sklearn.model_selection import train_test_split

        # Data pipeline - keep customer_id & transaction_time for sequencing
        pipeline = DataPipeline()
        pipeline.load_data()
        pipeline.handle_missing_values()
        pipeline.feature_engineering()
        pipeline.encode_categorical()

        df = pipeline.df.copy()

        # Remove transaction_id & device_id but keep customer_id
        for col in ["transaction_id", "device_id"]:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["is_fraud"]
        )

        # Determine feature columns (everything except identifiers and target)
        exclude_cols = {"customer_id", "is_fraud", "transaction_time"}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        # Scale features in-place
        preprocessor = Preprocessor()
        train_df[feature_cols] = preprocessor.fit_transform(
            train_df[feature_cols]
        )
        test_df[feature_cols] = preprocessor.transform(
            test_df[feature_cols]
        )

        os.makedirs(MODELS_DIR, exist_ok=True)
        pickle.dump(preprocessor.scaler, open(MODELS_DIR / "scaler.pkl", "wb"))

        with open(MODELS_DIR / "feature_names_lstm.json", "w") as f:
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
        y_train = train_df["is_fraud"]
        trainer.train(
            train_loader, test_loader, y_train,
            epochs=50, learning_rate=0.001, patience=10
        )
        trainer.save_model(str(MODELS_DIR) + "/")

        # Evaluate
        predictions, probabilities = trainer.predict(test_loader)
        test_labels_from_loader = test_loader.dataset.labels.numpy()

        evaluator = ModelEvaluator(
            test_labels_from_loader, predictions, probabilities,
            "LSTM Neural Network"
        )
        metrics = evaluator.calculate_metrics()
        lstm_eval_metrics = {k: float(v) for k, v in metrics.items()}

        with open(MODELS_DIR / "eval_metrics_lstm.json", "w") as f:
            json.dump(lstm_eval_metrics, f, indent=2)

        # Reload LSTM model
        lstm_detector = LSTMFraudDetector(model_path=str(MODELS_DIR))
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