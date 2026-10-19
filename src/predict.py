"""
Prediction Module for Fraud Detection (PyTorch)
Handles loading PyTorch models, extracting/imputing receipt features,
and making predictions on new transactions.
"""

import io
import json
import pickle
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch
    from .pytorch_model import FraudDetectorNet
    HAS_TORCH = True
except ImportError:
    torch = None
    FraudDetectorNet = None
    HAS_TORCH = False

from .receipt_parser import GENERAL_FEATURE_DEFAULTS, extract_and_impute_receipt


class _NumpyRobustScaler:
    """Fallback RobustScaler unpickler when scikit-learn is not installed in the runtime."""

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def transform(self, X: Any) -> np.ndarray:
        vals = X.values if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        return (vals - self.center_) / self.scale_


def _load_scaler_with_fallback(scaler_file: Path) -> Any:
    """Load scaler.pkl using standard pickle, or fallback unpickler if sklearn is absent."""
    try:
        with open(scaler_file, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        sklearn_mod = types.ModuleType("sklearn")
        prep_mod = types.ModuleType("sklearn.preprocessing")
        data_mod = types.ModuleType("sklearn.preprocessing._data")
        data_mod.RobustScaler = _NumpyRobustScaler
        prep_mod._data = data_mod
        sklearn_mod.preprocessing = prep_mod
        sys.modules.setdefault("sklearn", sklearn_mod)
        sys.modules.setdefault("sklearn.preprocessing", prep_mod)
        sys.modules.setdefault("sklearn.preprocessing._data", data_mod)
        with open(scaler_file, "rb") as f:
            return pickle.load(f)


def _load_pt_checkpoint_numpy(model_file: Path) -> Dict[str, Any]:
    """Load a PyTorch .pt zip checkpoint into numpy arrays when torch is not installed."""
    with zipfile.ZipFile(model_file, "r") as z:
        names = z.namelist()
        prefix = names[0].split("/")[0]

        def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            count = int(np.prod(size)) if len(size) > 0 else 1
            arr = storage[storage_offset : storage_offset + count]
            return arr.reshape(size)

        torch_mod = types.ModuleType("torch")
        utils_mod = types.ModuleType("torch._utils")
        utils_mod._rebuild_tensor_v2 = rebuild_tensor_v2
        torch_mod._utils = utils_mod
        torch_mod.FloatStorage = "FloatStorage"
        torch_mod.LongStorage = "LongStorage"
        sys.modules.setdefault("torch", torch_mod)
        sys.modules.setdefault("torch._utils", utils_mod)

        class _PtUnpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                _, storage_type, key, _, _ = pid
                raw = z.read(f"{prefix}/data/{key}")
                dtype = np.float32 if storage_type == "FloatStorage" else np.int64
                return np.frombuffer(raw, dtype=dtype)

        return _PtUnpickler(io.BytesIO(z.read(f"{prefix}/data.pkl"))).load()


class _NumpyMLPModel:
    """Exact NumPy inference implementation of FraudDetectorNet (6-layer MLP)."""

    def __init__(self, state_dict: Dict[str, np.ndarray], input_dim: int):
        self.state_dict = state_dict
        self.input_dim = input_dim

    def eval(self):
        return self

    def parameters(self):
        class _Param:
            def __init__(self, arr):
                self.arr = arr
                self.requires_grad = True
            def numel(self):
                return int(self.arr.size)
        return [_Param(v) for k, v in self.state_dict.items() if k.endswith(".weight") or k.endswith(".bias")]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sd = self.state_dict
        out = np.asarray(x, dtype=np.float32)
        for lin_idx, bn_idx in [(0, 1), (4, 5), (8, 9), (12, 13)]:
            out = out @ sd[f"network.{lin_idx}.weight"].T + sd[f"network.{lin_idx}.bias"]
            mean = sd[f"network.{bn_idx}.running_mean"]
            var = sd[f"network.{bn_idx}.running_var"]
            w = sd[f"network.{bn_idx}.weight"]
            b = sd[f"network.{bn_idx}.bias"]
            out = (out - mean) / np.sqrt(var + 1e-5) * w + b
            out = np.maximum(0.0, out)
        out = out @ sd["network.16.weight"].T + sd["network.16.bias"]
        out = np.maximum(0.0, out)
        out = out @ sd["network.19.weight"].T + sd["network.19.bias"]
        prob = 1.0 / (1.0 + np.exp(-np.clip(out, -60.0, 60.0)))
        return prob.squeeze(-1)


class _NumpyLSTMModel:
    """Exact NumPy inference implementation of FraudLSTMNet (2-layer BiLSTM)."""

    def __init__(self, state_dict: Dict[str, np.ndarray], input_dim: int, hidden_dim: int = 64):
        self.state_dict = state_dict
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def eval(self):
        return self

    def parameters(self):
        class _Param:
            def __init__(self, arr):
                self.arr = arr
                self.requires_grad = True
            def numel(self):
                return int(self.arr.size)
        return [_Param(v) for k, v in self.state_dict.items() if "weight" in k or "bias" in k]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    def _run_lstm_direction(
        self, x_seq: np.ndarray, w_ih: np.ndarray, w_hh: np.ndarray, b_ih: np.ndarray, b_hh: np.ndarray, reverse: bool = False
    ) -> np.ndarray:
        batch_size, seq_len, _ = x_seq.shape
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        outputs = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float32)
        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in steps:
            gates = x_seq[:, t, :] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            i_gate, f_gate, g_gate, o_gate = np.split(gates, 4, axis=1)
            i_gate = self._sigmoid(i_gate)
            f_gate = self._sigmoid(f_gate)
            g_gate = np.tanh(g_gate)
            o_gate = self._sigmoid(o_gate)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            outputs[:, t, :] = h
        return outputs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sd = self.state_dict
        out = np.asarray(x, dtype=np.float32)
        for layer in range(2):
            fwd = self._run_lstm_direction(
                out,
                sd[f"lstm.weight_ih_l{layer}"],
                sd[f"lstm.weight_hh_l{layer}"],
                sd[f"lstm.bias_ih_l{layer}"],
                sd[f"lstm.bias_hh_l{layer}"],
                reverse=False,
            )
            rev = self._run_lstm_direction(
                out,
                sd[f"lstm.weight_ih_l{layer}_reverse"],
                sd[f"lstm.weight_hh_l{layer}_reverse"],
                sd[f"lstm.bias_ih_l{layer}_reverse"],
                sd[f"lstm.bias_hh_l{layer}_reverse"],
                reverse=True,
            )
            out = np.concatenate([fwd, rev], axis=2)

        last_step = out[:, -1, :]
        fc = last_step @ sd["classifier.0.weight"].T + sd["classifier.0.bias"]
        fc = (fc - sd["classifier.1.running_mean"]) / np.sqrt(sd["classifier.1.running_var"] + 1e-5) * sd["classifier.1.weight"] + sd["classifier.1.bias"]
        fc = np.maximum(0.0, fc)
        fc = fc @ sd["classifier.4.weight"].T + sd["classifier.4.bias"]
        fc = np.maximum(0.0, fc)
        logits = fc @ sd["classifier.7.weight"].T + sd["classifier.7.bias"]
        return logits.squeeze(-1)


class FraudDetector:
    """
    Main prediction class for PyTorch-based MLP fraud detection
    """

    def __init__(self, model_path="models/", device=None):
        if HAS_TORCH:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device or "cpu"
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.feature_names = None

    def load_model(self, model_name="fraud_detector.pt"):
        """Load trained PyTorch model from disk"""
        model_file = Path(self.model_path) / model_name

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if HAS_TORCH:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
            input_dim = checkpoint["input_dim"]
            self.model = FraudDetectorNet(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        else:
            checkpoint = _load_pt_checkpoint_numpy(model_file)
            input_dim = checkpoint["input_dim"]
            self.model = _NumpyMLPModel(checkpoint["model_state_dict"], input_dim)

        print(f"PyTorch model loaded: {model_name} (input_dim={input_dim})")
        return self.model

    def load_scaler(self, scaler_name="scaler.pkl"):
        """Load fitted scaler from disk"""
        scaler_file = Path(self.model_path) / scaler_name

        if scaler_file.exists():
            self.scaler = _load_scaler_with_fallback(scaler_file)
            print(f"Scaler loaded: {scaler_name}")

        return self.scaler

    def load_feature_names(self, filename="feature_names.json"):
        """Load feature names used during training"""
        feature_file = Path(self.model_path) / filename

        if feature_file.exists():
            with open(feature_file, "r") as f:
                self.feature_names = json.load(f)
            print(f"Feature names loaded: {len(self.feature_names)} features")

        return self.feature_names

    @staticmethod
    def impute_missing_features(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill any missing base transaction features with general baseline features
        so partial inputs (such as from an uploaded receipt) have a complete feature set.
        """
        completed = dict(GENERAL_FEATURE_DEFAULTS)
        for k, v in transaction_data.items():
            if v is not None and v != "":
                completed[k] = v
        return completed

    def preprocess_transaction(self, transaction_data, impute_missing: bool = False):
        """
        Preprocess a single transaction for prediction

        Args:
            transaction_data: dict or DataFrame with transaction features
            impute_missing: whether to fill missing base fields with general defaults

        Returns:
            Preprocessed features DataFrame ready for model
        """
        if isinstance(transaction_data, dict):
            tx_dict = dict(transaction_data)
            # Automatically impute missing base features when using the full trained 22-feature model
            # or when explicitly requested via impute_missing=True
            if impute_missing or (self.feature_names is not None and len(self.feature_names) >= 10):
                tx_dict = self.impute_missing_features(tx_dict)
            df = pd.DataFrame([tx_dict])
        else:
            df = transaction_data.copy()
            if impute_missing or (self.feature_names is not None and len(self.feature_names) >= 10):
                for key, default_val in GENERAL_FEATURE_DEFAULTS.items():
                    if key not in df.columns:
                        df[key] = default_val
                    else:
                        df[key] = df[key].fillna(default_val)

        # Feature engineering (same as training pipeline)
        if "transaction_time" in df.columns:
            df["hour_of_day"] = df["transaction_time"] // 3600
            df["is_night_transaction"] = (
                (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)
            ).astype(int)

        if "transaction_amount" in df.columns and "avg_transaction_amount" in df.columns:
            df["amount_to_avg_ratio"] = df["transaction_amount"] / (
                df["avg_transaction_amount"] + 1
            )
            df["high_amount"] = (df["transaction_amount"] > 500).astype(int)

        if "merchant_category" in df.columns:
            df["unusual_merchant"] = (df["merchant_category"] == "online").astype(int)

        if "account_age_days" in df.columns:
            df["new_customer"] = (df["account_age_days"] < 90).astype(int)

        if "transaction_count_24h" in df.columns:
            df["high_frequency"] = (df["transaction_count_24h"] > 10).astype(int)

        # Encode categorical variables
        # Note: For single-row DataFrames, pd.get_dummies(..., drop_first=True) drops the single
        # category column! When aligning with self.feature_names, we set one-hot columns directly
        # so the active city/merchant category is preserved as 1.
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in list(categorical_cols):
            if col not in ["transaction_id", "customer_id", "device_id"]:
                if len(df) == 1 and self.feature_names is not None:
                    val = str(df[col].iloc[0])
                    target_col = f"{col}_{val}"
                    if target_col in self.feature_names:
                        df[target_col] = 1
                else:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)

        # Remove non-feature columns
        drop_cols = ["transaction_id", "customer_id", "device_id", "is_fraud", "transaction_time"]
        for col in drop_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        # Align with training features
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        return df

    def predict(self, transaction_data, threshold=0.5):
        """
        Predict if transaction is fraudulent using PyTorch MLP model

        Args:
            transaction_data: dict or DataFrame with transaction features
            threshold: classification threshold (default 0.5)

        Returns:
            dict with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        features = self.preprocess_transaction(transaction_data)

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values

        # Convert to tensor and predict
        self.model.eval()
        if HAS_TORCH and hasattr(self.model, "network"):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(features_scaled).to(self.device)
                fraud_probability = self.model(X_tensor).cpu().numpy()
        else:
            fraud_probability = self.model(features_scaled)

        if np.ndim(fraud_probability) == 0:
            fraud_probability = float(fraud_probability)
        else:
            fraud_probability = float(np.asarray(fraud_probability).flat[0])

        is_fraud = fraud_probability >= threshold

        result = {
            "is_fraud": bool(is_fraud),
            "fraud_probability": fraud_probability,
            "threshold": threshold,
            "risk_level": self._get_risk_level(fraud_probability),
        }

        return result

    def predict_from_receipt(
        self,
        file_bytes: Optional[bytes] = None,
        filename: str = "",
        content_type: str = "",
        raw_text: Optional[str] = None,
        form_overrides: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Extract transaction features from an uploaded receipt/bill, fill any missing
        features with general baseline defaults, and run the MLP fraud prediction.
        """
        parsed = extract_and_impute_receipt(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            raw_text=raw_text,
            form_overrides=form_overrides,
        )
        final_features = parsed["final_features"]
        prediction = self.predict(final_features, threshold=threshold)
        prediction.update({
            "extracted_features": parsed["extracted_features"],
            "imputed_features": parsed["imputed_features"],
            "missing_features": parsed["missing_features"],
            "final_features": final_features,
            "raw_text": parsed["raw_text"],
        })
        return prediction

    def _get_risk_level(self, probability):
        """Categorize risk level based on fraud probability"""
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_batch(self, transactions, threshold=0.5):
        """Predict fraud for multiple transactions"""
        results = []
        for transaction in transactions:
            result = self.predict(transaction, threshold)
            results.append(result)
        return results


def create_sample_transaction():
    """Create a sample transaction for testing"""
    return {
        "transaction_amount": 250.00,
        "transaction_time": 36000,
        "location": "Mumbai",
        "device_id": "DEV001",
        "merchant_category": "retail",
        "account_age_days": 365,
        "transaction_count_24h": 3,
        "avg_transaction_amount": 150.00,
    }


def test_predictions():
    """Test the fraud detection system"""
    detector = FraudDetector()

    try:
        detector.load_model()
        detector.load_scaler()
        detector.load_feature_names()
    except FileNotFoundError:
        print("No trained model found. Please train a model first.")
        return

    sample = create_sample_transaction()
    result = detector.predict(sample)

    print("\n--- Test Prediction ---")
    print(f"Transaction: {sample}")
    print(f"Result: {result}")


class LSTMFraudDetector:
    """
    Prediction class for the LSTM-based fraud detection model.

    For single-transaction prediction, builds a zero-padded sequence of
    length `seq_length` with the transaction as the last time step.
    """

    def __init__(self, model_path="models/", device=None):
        if HAS_TORCH:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device or "cpu"
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.feature_names = None
        self.seq_length = 10  # default, overridden on load

    def load_model(self, model_name="fraud_detector_lstm.pt"):
        """Load trained LSTM model from disk"""
        model_file = Path(self.model_path) / model_name

        if not model_file.exists():
            raise FileNotFoundError(f"LSTM model file not found: {model_file}")

        if HAS_TORCH:
            from .pytorch_model import FraudLSTMNet
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=True)
            input_dim = checkpoint["input_dim"]
            self.seq_length = checkpoint.get("seq_length", 10)
            self.model = FraudLSTMNet(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        else:
            checkpoint = _load_pt_checkpoint_numpy(model_file)
            input_dim = checkpoint["input_dim"]
            self.seq_length = checkpoint.get("seq_length", 10)
            self.model = _NumpyLSTMModel(checkpoint["model_state_dict"], input_dim)

        print(f"LSTM model loaded: {model_name} (input_dim={input_dim}, seq_length={self.seq_length})")
        return self.model

    def load_scaler(self, scaler_name="scaler.pkl"):
        """Load fitted scaler from disk"""
        scaler_file = Path(self.model_path) / scaler_name

        if scaler_file.exists():
            self.scaler = _load_scaler_with_fallback(scaler_file)
            print(f"Scaler loaded: {scaler_name}")

        return self.scaler

    def load_feature_names(self, filename="feature_names_lstm.json"):
        """Load feature names used during training"""
        feature_file = Path(self.model_path) / filename
        if not feature_file.exists():
            feature_file = Path(self.model_path) / "feature_names.json"

        if feature_file.exists():
            with open(feature_file, "r") as f:
                self.feature_names = json.load(f)
            print(f"Feature names loaded: {len(self.feature_names)} features")

        return self.feature_names

    def preprocess_transaction(self, transaction_data, impute_missing: bool = False):
        """
        Preprocess a single transaction for LSTM prediction.
        Reuses the same feature engineering as FraudDetector.
        """
        helper = FraudDetector(model_path=self.model_path, device=self.device)
        helper.feature_names = self.feature_names
        return helper.preprocess_transaction(transaction_data, impute_missing=impute_missing)

    def predict(self, transaction_data, threshold=0.5):
        """
        Predict if a transaction is fraudulent using the LSTM model.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        features = self.preprocess_transaction(transaction_data)

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values

        # Build padded sequence: (1, seq_length, n_features)
        n_features = features_scaled.shape[1]
        seq = np.zeros((self.seq_length, n_features), dtype=np.float32)
        seq[-1, :] = features_scaled[0]  # place transaction at last step

        self.model.eval()
        if HAS_TORCH and hasattr(self.model, "lstm"):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                logits = self.model(X_tensor)
                fraud_probability = torch.sigmoid(logits).cpu().numpy()
        else:
            logits = self.model(np.expand_dims(seq, axis=0))
            fraud_probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

        if np.ndim(fraud_probability) == 0:
            fraud_probability = float(fraud_probability)
        else:
            fraud_probability = float(np.asarray(fraud_probability).flat[0])

        is_fraud = fraud_probability >= threshold

        result = {
            "is_fraud": bool(is_fraud),
            "fraud_probability": fraud_probability,
            "threshold": threshold,
            "risk_level": self._get_risk_level(fraud_probability),
        }

        return result

    def predict_from_receipt(
        self,
        file_bytes: Optional[bytes] = None,
        filename: str = "",
        content_type: str = "",
        raw_text: Optional[str] = None,
        form_overrides: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Extract features from receipt, impute missing features, and run LSTM prediction."""
        parsed = extract_and_impute_receipt(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            raw_text=raw_text,
            form_overrides=form_overrides,
        )
        final_features = parsed["final_features"]
        prediction = self.predict(final_features, threshold=threshold)
        prediction.update({
            "extracted_features": parsed["extracted_features"],
            "imputed_features": parsed["imputed_features"],
            "missing_features": parsed["missing_features"],
            "final_features": final_features,
            "raw_text": parsed["raw_text"],
        })
        return prediction

    def _get_risk_level(self, probability):
        """Categorize risk level based on fraud probability"""
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"


if __name__ == "__main__":
    test_predictions()