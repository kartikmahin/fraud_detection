"""
Data Pipeline Module for Fraud Detection
Handles data loading, cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPipeline:
    """
    Complete data pipeline for banking fraud detection
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        
    def load_data(self, path=None):
        """
        Load transaction data from CSV or generate synthetic data
        """
        data_path = path or self.data_path
        
        # Try default dataset path if none specified
        if not data_path:
            import os
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'transactions.csv')
            if os.path.exists(default_path):
                data_path = default_path
        
        if data_path:
            self.df = pd.read_csv(data_path)
            print(f"Loaded data from {data_path}")
        else:
            self.df = self._generate_synthetic_data()
            print("Generated synthetic fraud detection dataset")
            
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def _generate_synthetic_data(self, n_samples=100000, fraud_rate=0.02):
        """
        Generate synthetic banking transaction data
        Real fraud datasets are extremely imbalanced (~1% fraud)
        """
        np.random.seed(42)
        
        n_fraud = int(n_samples * fraud_rate)
        n_legitimate = n_samples - n_fraud
        
        legitimate_data = {
            'transaction_id': [f'TXN{i:08d}' for i in range(n_legitimate)],
            'customer_id': np.random.randint(1000, 5000, n_legitimate),
            'transaction_amount': np.random.exponential(150, n_legitimate).round(2),
            'transaction_time': np.random.randint(0, 86400, n_legitimate),
            'location': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'], n_legitimate),
            'device_id': np.random.choice([f'DEV{i:03d}' for i in range(100)], n_legitimate),
            'merchant_category': np.random.choice(['retail', 'grocery', 'restaurant', 'gas', 'online'], n_legitimate),
            'account_age_days': np.random.randint(30, 3650, n_legitimate),
            'transaction_count_24h': np.random.randint(1, 20, n_legitimate),
            'avg_transaction_amount': np.random.exponential(150, n_legitimate).round(2),
            'is_fraud': 0
        }
        
        fraud_data = {
            'transaction_id': [f'TXN{i:08d}' for i in range(n_legitimate, n_samples)],
            'customer_id': np.random.randint(1000, 5000, n_fraud),
            'transaction_amount': np.random.exponential(500, n_fraud).round(2),
            'transaction_time': np.random.randint(0, 86400, n_fraud),
            'location': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'], n_fraud),
            'device_id': np.random.choice([f'DEV{i:03d}' for i in range(100)], n_fraud),
            'merchant_category': np.random.choice(['retail', 'grocery', 'restaurant', 'gas', 'online'], n_fraud),
            'account_age_days': np.random.randint(30, 3650, n_fraud),
            'transaction_count_24h': np.random.randint(1, 50, n_fraud),
            'avg_transaction_amount': np.random.exponential(150, n_fraud).round(2),
            'is_fraud': 1
        }
        
        df_legit = pd.DataFrame(legitimate_data)
        df_fraud = pd.DataFrame(fraud_data)
        
        df = pd.concat([df_legit, df_fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\n--- Dataset Info ---")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        print("\n--- First 5 Rows ---")
        print(self.df.head())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        print("\n--- Missing Values ---")
        print(self.df.isnull().sum())
        
        print("\n--- Target Distribution ---")
        fraud_dist = self.df['is_fraud'].value_counts()
        fraud_pct = self.df['is_fraud'].value_counts(normalize=True) * 100
        print(f"Legitimate (0): {fraud_dist[0]} ({fraud_pct[0]:.2f}%)")
        print(f"Fraud (1): {fraud_dist[1]} ({fraud_pct[1]:.2f}%)")
        print(f"\nImbalance Ratio: {fraud_dist[0]/fraud_dist[1]:.1f}:1")
        
        return self.df
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("\n--- Handling Missing Values ---")
        
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() == 0:
            print("No missing values found")
            return self.df
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        print("Missing values imputed")
        return self.df
    
    def detect_outliers(self, columns=None):
        """
        Detect outliers using IQR method
        """
        if columns is None:
            columns = ['transaction_amount', 'transaction_count_24h']
        
        print("\n--- Outlier Detection (IQR Method) ---")
        
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
                print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return self.df
    
    def feature_engineering(self):
        """
        Create new features for better fraud detection
        """
        print("\n--- Feature Engineering ---")
        
        original_cols = len(self.df.columns)
        
        self.df['hour_of_day'] = self.df['transaction_time'] // 3600
        
        self.df['is_night_transaction'] = ((self.df['hour_of_day'] >= 22) | 
                                            (self.df['hour_of_day'] <= 5)).astype(int)
        
        self.df['amount_to_avg_ratio'] = (
            self.df['transaction_amount'] / (self.df['avg_transaction_amount'] + 1)
        )
        
        self.df['high_amount'] = (self.df['transaction_amount'] > 500).astype(int)
        
        self.df['unusual_merchant'] = (
            self.df['merchant_category'] == 'online'
        ).astype(int)
        
        self.df['new_customer'] = (self.df['account_age_days'] < 90).astype(int)
        
        self.df['high_frequency'] = (
            self.df['transaction_count_24h'] > 10
        ).astype(int)
        
        new_cols = len(self.df.columns) - original_cols
        print(f"Created {new_cols} new features")
        print(f"New features: hour_of_day, is_night_transaction, amount_to_avg_ratio, "
              f"high_amount, unusual_merchant, new_customer, high_frequency")
        
        return self.df
    
    def encode_categorical(self):
        """
        Encode categorical variables
        """
        print("\n--- Encoding Categorical Variables ---")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['transaction_id', 'device_id']:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                print(f"Encoded {col}")
        
        return self.df
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare train/test split
        """
        print("\n--- Preparing Train/Test Split ---")
        
        if 'transaction_id' in self.df.columns:
            self.df.drop('transaction_id', axis=1, inplace=True)
        if 'customer_id' in self.df.columns:
            self.df.drop('customer_id', axis=1, inplace=True)
        if 'device_id' in self.df.columns:
            self.df.drop('device_id', axis=1, inplace=True)
        if 'transaction_time' in self.df.columns:
            self.df.drop('transaction_time', axis=1, inplace=True)
        
        X = self.df.drop('is_fraud', axis=1)
        y = self.df['is_fraud']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Training fraud rate: {self.y_train.mean()*100:.2f}%")
        print(f"Test fraud rate: {self.y_test.mean()*100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_feature_names(self):
        """
        Get feature names after preprocessing
        """
        return list(self.X_train.columns)


def main():
    """
    Main function to run data pipeline
    """
    pipeline = DataPipeline()
    
    pipeline.load_data()
    pipeline.explore_data()
    pipeline.handle_missing_values()
    pipeline.detect_outliers()
    pipeline.feature_engineering()
    pipeline.encode_categorical()
    X_train, X_test, y_train, y_test = pipeline.prepare_data()
    
    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pipeline, X_train, X_test, y_train, y_test = main()
