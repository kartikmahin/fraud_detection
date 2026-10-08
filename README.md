# Banking System Fraud Detection using Deep Learning

A complete end-to-end production-ready deep learning project for detecting fraudulent banking transactions using **PyTorch** with dual-model architecture: **MLP (DNN)** and **Bidirectional LSTM**.

## 🌐 Live Deployments
- **Frontend (Vercel)**: [https://fraud-detection-fraud4.vercel.app/](https://fraud-detection-fraud4.vercel.app/)
- **Backend API (Render)**: [https://fraud-detection-yda0.onrender.com](https://fraud-detection-yda0.onrender.com)
- **API Health Check**: [https://fraud-detection-yda0.onrender.com/health](https://fraud-detection-yda0.onrender.com/health)
- **Interactive Swagger Docs**: [https://fraud-detection-yda0.onrender.com/docs](https://fraud-detection-yda0.onrender.com/docs)

## Key Features

- **Instant Zero-Retrain Inference** — Pre-trained deep learning weights are bundled and auto-loaded on backend startup
- **Dual-Model Architecture** — MLP (6-layer DNN) + Bidirectional LSTM for comprehensive fraud detection
- **PyTorch Deep Neural Networks** — FraudDetectorNet (MLP) & FraudLSTMNet (LSTM)
- **FastAPI REST API** with CORS for React frontend integration
- **React + Vite web interface** with dark mode and comparative dashboard
- **Comprehensive test suite** (97 unit tests)
- **Class-weighted BCE loss** for handling imbalanced data
- **RobustScaler** for feature normalization

## Project Overview

This project implements a comprehensive fraud detection system that addresses the challenge of detecting rare fraudulent transactions in highly imbalanced datasets (<1% fraud rate). It employs two complementary deep learning approaches:

1. **MLP (Multi-Layer Perceptron)** — Analyzes each transaction independently using a 6-layer fully-connected network
2. **LSTM (Long Short-Term Memory)** — Captures temporal patterns across sequences of customer transactions using a bidirectional 2-layer LSTM

---

## 1. Problem Understanding

### The Challenge
- **Imbalanced Data**: Fraudulent transactions represent less than 1% of all transactions
- **Evolving Patterns**: Fraudsters constantly adapt their methods
- **False Positives vs False Negatives**: Balancing customer experience with fraud prevention

### Business Context
- Digital banking processes millions of transactions daily
- Traditional rule-based systems generate many false positives
- Financial losses from fraud are significant
- Customer experience matters - too many declined transactions leads to churn

---

## 2. Data Pipeline

### Files: `src/data_pipeline.py`

Complete data processing pipeline:

```
DataPipeline
├── load_data()           # Load from CSV or generate synthetic data
├── explore_data()        # EDA and data understanding
├── handle_missing_values() # Imputation
├── detect_outliers()     # IQR-based outlier detection
├── feature_engineering() # Create new features
├── encode_categorical()  # One-hot encoding
└── prepare_data()        # Train/test split
```

### Feature Engineering
- `hour_of_day`: Time of transaction
- `is_night_transaction`: Late night transactions are higher risk
- `amount_to_avg_ratio`: Unusually high amounts
- `high_amount`: Flag for high-value transactions
- `unusual_merchant`: Online purchases are riskier
- `new_customer`: New accounts have higher fraud risk
- `high_frequency`: Unusual transaction frequency

---

## 3. Handling Imbalanced Data

### Why Imbalanced Data is Problematic
- Model learns to predict majority class
- 98% accuracy can be achieved by predicting all legitimate!
- Misses the important minority class (fraud)

### Techniques Implemented

#### A. SMOTE (Synthetic Minority Over-sampling Technique)
**File: `src/preprocessing.py`**

```
SMOTE Process:
1. Select random minority sample
2. Find k-nearest neighbors (k=5)
3. Randomly select one neighbor
4. Generate synthetic sample between them
```

**Pros:**
- Creates new synthetic samples (not duplicates)
- Helps model learn better decision boundaries
- Doesn't discard valuable majority class data

**Cons:**
- Can create noise when applied incorrectly
- May not work well with high-dimensional data

#### B. Random Under Sampling
```
Process: Randomly remove majority class samples to match minority class
```

**Pros:**
- Simple to implement
- Faster training

**Cons:**
- Loses valuable information
- May create biased sample

#### C. Cost-Sensitive Learning
Instead of changing data, we modify the learning algorithm:

```python
class_weight='balanced'  # In sklearn models
```

This assigns higher penalty to misclassification of minority class.

---

## 4. Model Training

### File: `src/train_model.py`

### Model 1: FraudDetectorNet — MLP (PyTorch)

```
Input (23 features) → 128 → 256 → 128 → 64 → 32 → 1 (Sigmoid)
```

| Component | Details |
|-----------|---------|
| **Layers** | 6 fully connected layers with BatchNorm + Dropout (0.3) |
| **Activation** | ReLU between hidden layers, Sigmoid for output |
| **Loss** | BCE loss with cost-sensitive class weights |
| **Optimizer** | Adam with weight decay (1e-5) |
| **Scheduler** | ReduceLROnPlateau |
| **Regularization** | Dropout 0.3, gradient clipping (max_norm=1.0) |
| **Early Stopping** | Patience=10 epochs |

### Model 2: FraudLSTMNet — Bidirectional LSTM (PyTorch)

```
Input (batch, seq=10, features) → BiLSTM(2 layers, hidden=64) → FC(128→64→32→1) → Sigmoid
```

| Component | Details |
|-----------|---------|
| **Architecture** | 2-layer Bidirectional LSTM + FC head |
| **Sequence Length** | 10 transactions per customer |
| **Hidden Dim** | 64 (bidirectional → 128) |
| **Loss** | BCE loss with cost-sensitive class weights |
| **Optimizer** | Adam with weight decay (1e-5) |
| **Scheduler** | ReduceLROnPlateau |
| **Early Stopping** | Patience=10 epochs |

### Training Code Example
```python
# MLP Training
trainer = PyTorchTrainer(input_dim)
trainer.build_model(dropout_rate=0.3)
training_history = trainer.train(
    train_loader, test_loader, y_train,
    epochs=50, learning_rate=0.001, patience=10
)
trainer.save_model('models/')

# LSTM Training
lstm_trainer = LSTMTrainer(input_dim, seq_length=10)
lstm_trainer.build_model()
lstm_history = lstm_trainer.train(
    train_loader, test_loader, y_train,
    epochs=50, learning_rate=0.001, patience=10
)
lstm_trainer.save_model('models/')
```

---

## 5. Evaluation Metrics

### File: `src/evaluate.py`

### Why NOT Accuracy?
A model predicting ALL transactions as legitimate achieves 98% accuracy with 2% fraud rate!

### Metrics Used

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Precision** | TP/(TP+FP) | Of flagged fraud, how many are actual fraud |
| **Recall** | TP/(TP+FN) | Of actual fraud, how many did we catch |
| **F1 Score** | 2×(P×R)/(P+R) | Balance between precision and recall |
| **ROC-AUC** | Area under ROC | Discrimination ability |
| **PR-AUC** | Area under PR curve | Best for imbalanced data |

### Confusion Matrix Analysis

```
                    Predicted
                  Legit    Fraud
Actual Legit     [TN]     [FP]
Actual Fraud    [FN]     [TP]
```

### Business Impact

**FALSE NEGATIVES (Missed Fraud) - MOST DANGEROUS:**
- Direct financial loss to bank and customer
- Reputational damage
- Regulatory penalties
- Customer loses trust
- Example: $1000 fraud not caught = $1000 loss

**FALSE POSITIVES (False Alarms):**
- Customer frustration
- Transaction declined at checkout
- Customer switches to competitor
- Customer support costs increase
- Example: Legit purchase declined = poor experience

---

## 6. Trained Model Performance

### MLP (FraudDetectorNet) — Test Set Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.55% |
| **Precision** | 82.12% |
| **Recall** | 98.75% |
| **Specificity** | 99.56% |
| **F1 Score** | 89.67% |
| **ROC-AUC** | 99.97% |
| **PR-AUC** | 99.38% |

### LSTM (FraudLSTMNet) — Test Set Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.39% |
| **Precision** | 76.94% |
| **Recall** | 99.25% |
| **Specificity** | 99.39% |
| **F1 Score** | 86.68% |
| **ROC-AUC** | 99.94% |
| **PR-AUC** | 99.34% |

### Model Comparison

| Metric | MLP | LSTM | Winner |
|--------|-----|------|--------|
| **Accuracy** | 99.55% | 99.39% | MLP (+0.16pp) |
| **Precision** | 82.12% | 76.94% | MLP (+5.18pp) |
| **Recall** | 98.75% | 99.25% | LSTM (+0.50pp) |
| **F1 Score** | 89.67% | 86.68% | MLP (+2.99pp) |
| **ROC-AUC** | 99.97% | 99.94% | MLP (+0.03pp) |
| **PR-AUC** | 99.38% | 99.34% | MLP (+0.04pp) |

> **Key Insight**: The MLP model slightly outperforms on most metrics (Precision 82.12% vs 76.94%, F1 Score 89.67% vs 86.68%), while the LSTM excels at Recall (99.25% vs 98.75% — catching more fraud). The LSTM's sequential approach captures temporal patterns across transaction sequences, making it complementary to the MLP.

---

## 7. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FRAUD DETECTION SYSTEM                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Streamlit  │     │   FastAPI    │     │   React      │        │
│  │     App      │     │   Backend    │     │   Frontend   │        │
│  └──────┬───────┘     └──────┬────────┘     └──────┬───────┘        │
│         │                    │                    │                  │
│         ▼                    ▼                    ▼                  │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              src/ (Python ML Pipeline)                 │        │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │        │
│  │  │   Data     │ │  Feature   │ │   MLP DNN          │ │        │
│  │  │  Pipeline  │ │ Engineering│ │   (6-layer)        │ │        │
│  │  └────────────┘ └────────────┘ └────────────────────┘ │        │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │        │
│  │  │   SMOTE    │ │  Scaling   │ │   Bidirectional    │ │        │
│  │  │  (balance) │ │ (Robust)   │ │   LSTM (sequence)  │ │        │
│  │  └────────────┘ └────────────┘ └────────────────────┘ │        │
│  │  ┌────────────────────────────────────────────────────┐│        │
│  │  │         Evaluation & Prediction Module             ││        │
│  │  └────────────────────────────────────────────────────┘│        │
│  └────────────────────────────────────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              models/ (Saved Artifacts)                  │        │
│  │   MLP: fraud_detector.pt   │  LSTM: fraud_detector_lstm.pt     │
│  │   scaler.pkl  │  feature_names.json  │  eval_metrics*.json     │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. Deployment API

### FastAPI Application: `api/app.py`

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check (MLP + LSTM status) |
| `/predict` | POST | MLP single transaction prediction |
| `/predict-lstm` | POST | LSTM single transaction prediction |
| `/batch-predict` | POST | MLP batch prediction |
| `/model-info` | GET | MLP model architecture info |
| `/model-info-lstm` | GET | LSTM model architecture info |
| `/metrics` | GET | MLP evaluation metrics |
| `/metrics-lstm` | GET | LSTM evaluation metrics |
| `/training-history` | GET | MLP training history |
| `/training-history-lstm` | GET | LSTM training history |
| `/train` | POST | Train MLP model |
| `/train-lstm` | POST | Train LSTM model |

#### Example Request (MLP)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 1500.00,
    "transaction_time": 36000,
    "location": "Mumbai",
    "device_id": "DEV001",
    "merchant_category": "retail",
    "account_age_days": 365,
    "transaction_count_24h": 3,
    "avg_transaction_amount": 150.00
  }'
```

#### Example Request (LSTM)
```bash
curl -X POST "http://localhost:8000/predict-lstm" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 1500.00,
    "transaction_time": 36000,
    "location": "Mumbai",
    "device_id": "DEV001",
    "merchant_category": "retail",
    "account_age_days": 365,
    "transaction_count_24h": 3,
    "avg_transaction_amount": 150.00
  }'
```

#### Example Response
```json
{
  "is_fraud": false,
  "fraud_probability": 0.15,
  "risk_level": "LOW",
  "message": "Transaction appears legitimate"
}
```

---

## 9. Streamlit Demo

### File: `streamlit_app.py`

Interactive web interface with:
- Make predictions
- Model evaluation metrics
- Model training
- Documentation

Run with:
```bash
streamlit run streamlit_app.py
```

---

## 10. Real-World Banking Impact

### How Banks Use These Systems

1. **Real-Time Decisioning**: Every transaction is scored in milliseconds
2. **Layered Defense**: ML + Rules + Human review
3. **Continuous Learning**: Models retrained with new fraud patterns
4. **Customer Communication**: Alerts for suspicious activity

### How Companies Reduce Fraud Risk

| Strategy | Implementation |
|----------|---------------|
| **Velocity Checks** | Block if too many transactions in short time |
| **Geolocation** | Flag transactions from unusual locations |
| **Device Fingerprinting** | Identify compromised devices |
| **Behavioral Analytics** | Learn spending patterns |
| **Network Analysis** | Graph-based fraud ring detection |
| **2FA/3D Secure** | Additional authentication |
| **Dynamic CVV** | Generate one-time CVV codes |

### Industry Statistics
- Banks lose $25-30 billion annually to card fraud globally
- Machine learning reduces fraud losses by 50-70%
- False positive reduction improves customer experience significantly

---

## 11. Future Improvements

### Production Readiness
- **Docker containerization** for reproducible deployment
- **CI/CD pipeline** with GitHub Actions
- **Structured logging** (JSON logs for production monitoring)
- **API rate limiting** with slowapi
- **Model versioning** to track deployed model versions
- **Environment-based configuration** via config.yaml or env variables

### Advanced Deep Learning Approaches
- **Autoencoders**: Learn normal transaction patterns, flag anomalies
- **Transformer Models**: Handle sequential transaction data with attention mechanisms
- **Graph Neural Networks (GNN)**: Detect fraud rings and suspicious networks
- **Ensemble Methods**: Combine MLP + LSTM predictions for improved accuracy

### Production Considerations
- Model monitoring and drift detection
- A/B testing for model updates
- Feature store for consistent features
- Model registry for version control
- Real-time feature computation
- Low-latency inference requirements

---

## Project Structure

```
fraud_detection_project/
│
├── data/                      # Data files
│   └── transactions.csv       # Synthetic dataset (100K transactions)
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_pipeline.py       # Data loading & preprocessing
│   ├── preprocessing.py       # Scaling & SMOTE
│   ├── pytorch_model.py       # MLP + LSTM model definitions
│   ├── train_model.py         # Training pipelines (MLP + LSTM)
│   ├── evaluate.py            # Evaluation metrics
│   └── predict.py             # Prediction module (MLP + LSTM)
│
├── models/                    # Saved models & metrics
│   ├── fraud_detector.pt      # Trained MLP model
│   ├── fraud_detector_lstm.pt # Trained LSTM model
│   ├── scaler.pkl             # Fitted RobustScaler
│   ├── feature_names.json     # MLP feature list
│   ├── feature_names_lstm.json # LSTM feature list
│   ├── training_history.json  # MLP epoch-by-epoch metrics
│   ├── training_history_lstm.json # LSTM epoch-by-epoch metrics
│   ├── eval_metrics.json      # MLP final metrics
│   └── eval_metrics_lstm.json # LSTM final metrics
│
├── api/                       # FastAPI app
│   ├── __init__.py
│   └── app.py                 # REST API (MLP + LSTM endpoints)
│
├── frontend/                  # React + Vite web application
│   ├── src/
│   │   ├── App.jsx            # Main app with routing
│   │   ├── api.js             # API client (MLP + LSTM)
│   │   ├── components/
│   │   │   └── Sidebar.jsx    # Navigation + model status
│   │   └── pages/
│   │       ├── Dashboard.jsx  # Dual-model comparison dashboard
│   │       ├── Predict.jsx    # Fraud prediction (MLP/LSTM toggle)
│   │       ├── Training.jsx   # Model training (MLP/LSTM)
│   │       ├── Metrics.jsx    # Side-by-side metrics comparison
│   │       └── About.jsx      # Tech stack & architecture
│   └── package.json
│
├── tests/                     # Test suite (97 tests)
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── test_data_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_evaluate.py
│   └── test_predict.py
│
├── streamlit_app.py           # Streamlit demo
│
├── pytest.ini                 # Test configuration
│
├── requirements.txt           # Dependencies
│
└── README.md                  # This file
```

---

## Quick Start & Running the Project

Follow these step-by-step instructions to set up and run the entire project, including the deep learning pipeline, the REST API, the React web application, and the Streamlit dashboard.

### 1. Prerequisites
- **Python 3.8+**
- **Node.js 18+** & **npm**

---

### 2. Backend Setup & Training

#### Step A: Install Python Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Step B: Train the Deep Learning Models
Run the training pipeline to process the dataset, train both the MLP and LSTM models, and save the model artifacts (saved under `models/`):

**Train MLP model:**
```bash
python -m src.train_model
```

**Train LSTM model (optional — can also be triggered from the web UI):**
Training the LSTM model is done via the FastAPI `/train-lstm` endpoint or from the React frontend's Training page.

#### Step C: Start the FastAPI Backend API
Launch the FastAPI development server. It will load both the MLP and LSTM models and run on `http://localhost:8000`:
```bash
uvicorn api.app:app --reload
```

---

### 3. Frontend Setup (React + Vite)

Open a new terminal window/tab to run the web application.

#### Step A: Navigate to Frontend Directory
```bash
cd frontend
```

#### Step B: Install Node Dependencies
```bash
npm install
```

#### Step C: Start the React App
Start the Vite development server. The React web application will be accessible at `http://localhost:5173/`:
```bash
npm run dev
```

---

### 4. Interactive Dashboards & Testing

#### Streamlit Admin Dashboard
To run the Streamlit monitoring and evaluation dashboard (accessible at `http://localhost:8501/`):
```bash
streamlit run streamlit_app.py
```

#### Running the Test Suite
To run the comprehensive suite of 97 unit tests:
```bash
pytest tests/ -v
```

---

### 5. Cloud Deployment Guide

#### Backend (FastAPI on Render)
1. Link your GitHub repository on [Render](https://render.com).
2. Create a new **Web Service**.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
5. The pre-trained weights (`models/*.pt`, `models/*.pkl`) are automatically loaded upon server startup.

#### Frontend (React on Vercel)
1. Import the `frontend` folder into [Vercel](https://vercel.com).
2. In **Settings -> Environment Variables**, add:
   - `VITE_API_URL` = `https://<your-render-service>.onrender.com` (Select **Config** type)
3. Under **Settings -> Deployment Protection**, ensure **Vercel Authentication** is disabled for public access.
4. Deploy the site.

#### Docker Container
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

---

## License

This project is for educational purposes.
