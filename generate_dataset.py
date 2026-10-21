"""
Generate a realistic synthetic fraud detection dataset
Enhanced with multi-type fraud patterns: IP anomalies, small-amount fraud, and velocity abuse.
Saves to data/transactions.csv
"""

import pandas as pd
import numpy as np
import os


def generate_dataset(n_samples=100000, fraud_rate=0.02, seed=42):
    """
    Generate a realistic synthetic banking transaction dataset for fraud detection.

    Features:
    - transaction_id: Unique ID per transaction
    - customer_id: Customer identifier
    - transaction_amount: Amount in INR
    - transaction_time: Seconds since midnight (0–86399)
    - location: Indian city
    - device_id: Device identifier
    - merchant_category: retail, grocery, restaurant, gas, online
    - account_age_days: Age of the account in days
    - transaction_count_24h: Number of transactions in the last 24 hours
    - avg_transaction_amount: Customer's average transaction amount

    IP metadata:
    - ip_country: Country code of the IP
    - ip_is_vpn: Whether the IP is a VPN
    - ip_is_proxy: Whether the IP is a proxy
    - ip_login_count: Number of different accounts on this IP in 24h
    - ip_geo_distance_km: Distance between IP geo and registered city

    Small-amount context:
    - small_txn_count_6h: Count of txns < ₹200 in last 6 hours
    - cumulative_amount_6h: Running total of small amounts in 6h window
    - same_merchant_count_1h: Txns to same merchant in last hour

    Labels:
    - is_fraud: Target label (0 = legitimate, 1 = fraud)
    - fraud_type: none, large_amount, small_amount_series, ip_anomaly, account_takeover
    """
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai',
              'Kolkata', 'Pune', 'Ahmedabad']
    merchants = ['retail', 'grocery', 'restaurant', 'gas', 'online']
    devices = [f'DEV{i:03d}' for i in range(200)]

    # ---- Legitimate transactions ----
    legit_amounts = np.clip(np.random.exponential(150, n_legit), 10, 5000).round(2)
    # Legitimate transactions lean towards daytime (bell curve around 2 PM)
    hour_weights = np.exp(-0.5 * ((np.arange(24) - 14) / 5) ** 2)
    hour_probs = hour_weights / hour_weights.sum()
    legit_hours = np.random.choice(24, n_legit, p=hour_probs)
    legit_times = legit_hours * 3600 + np.random.randint(0, 3600, n_legit)
    legit_avg_amounts = np.clip(np.random.exponential(150, n_legit), 20, 3000).round(2)
    legit_acct_ages = np.random.randint(90, 3650, n_legit)
    legit_txn_counts = np.random.randint(1, 10, n_legit)

    # IP metadata for legitimate transactions
    legit_ip_country = np.where(
        np.random.random(n_legit) < 0.97, 'IN', 
        np.random.choice(['US', 'GB', 'SG', 'AE', 'DE'], n_legit)
    )
    legit_ip_vpn = (np.random.random(n_legit) < 0.05).astype(int)     # 5% VPN usage
    legit_ip_proxy = (np.random.random(n_legit) < 0.02).astype(int)   # 2% proxy
    legit_ip_login_count = np.random.choice([1, 1, 1, 1, 2, 2, 3], n_legit)  # mostly 1
    legit_ip_geo_dist = np.clip(np.random.exponential(50, n_legit), 0, 300).round(1)

    # Small-amount context for legitimate transactions
    legit_small_count_6h = np.random.choice([0, 0, 0, 1, 1, 2], n_legit)
    legit_cum_amount_6h = np.clip(np.random.exponential(100, n_legit), 0, 500).round(2)
    legit_same_merchant_1h = np.random.choice([0, 0, 0, 1, 1], n_legit)

    legit_df = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(n_legit)],
        'customer_id': np.random.randint(1000, 10000, n_legit),
        'transaction_amount': legit_amounts,
        'transaction_time': legit_times,
        'location': np.random.choice(cities, n_legit, p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]),
        'device_id': np.random.choice(devices, n_legit),
        'merchant_category': np.random.choice(merchants, n_legit, p=[0.30, 0.25, 0.20, 0.10, 0.15]),
        'account_age_days': legit_acct_ages,
        'transaction_count_24h': legit_txn_counts,
        'avg_transaction_amount': legit_avg_amounts,
        # IP metadata
        'ip_country': legit_ip_country,
        'ip_is_vpn': legit_ip_vpn,
        'ip_is_proxy': legit_ip_proxy,
        'ip_login_count': legit_ip_login_count,
        'ip_geo_distance_km': legit_ip_geo_dist,
        # Small-amount context
        'small_txn_count_6h': legit_small_count_6h,
        'cumulative_amount_6h': legit_cum_amount_6h,
        'same_merchant_count_1h': legit_same_merchant_1h,
        # Labels
        'is_fraud': 0,
        'fraud_type': 'none',
    })

    # ---- Fraudulent transactions — split into subtypes ----
    # Distribution of fraud subtypes:
    #   40% large_amount  (existing pattern)
    #   25% small_amount_series  (card testing)
    #   20% ip_anomaly  (suspicious IP patterns)
    #   15% account_takeover  (combined signals)
    fraud_subtypes = np.random.choice(
        ['large_amount', 'small_amount_series', 'ip_anomaly', 'account_takeover'],
        n_fraud,
        p=[0.40, 0.25, 0.20, 0.15]
    )

    # Base fraud generation
    fraud_hour_weights = np.array([
        0.06, 0.07, 0.08, 0.08, 0.07, 0.05,  # 0-5 AM (high)
        0.03, 0.02, 0.02, 0.02, 0.02, 0.02,  # 6-11 AM (low)
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # 12-5 PM (low)
        0.03, 0.04, 0.05, 0.06, 0.07, 0.07   # 6-11 PM (rising)
    ])
    fraud_hour_probs = fraud_hour_weights / fraud_hour_weights.sum()
    fraud_hours = np.random.choice(24, n_fraud, p=fraud_hour_probs)
    fraud_times = fraud_hours * 3600 + np.random.randint(0, 3600, n_fraud)

    fraud_acct_ages = np.random.choice(
        np.arange(1, 3650), n_fraud,
        p=np.exp(-np.arange(1, 3650) / 300) / np.exp(-np.arange(1, 3650) / 300).sum()
    )  # Exponentially favor newer accounts

    # Generate per-subtype values
    fraud_amounts = np.zeros(n_fraud)
    fraud_avg_amounts = np.zeros(n_fraud)
    fraud_txn_counts = np.zeros(n_fraud, dtype=int)
    fraud_merchants = np.empty(n_fraud, dtype=object)
    fraud_ip_country = np.empty(n_fraud, dtype=object)
    fraud_ip_vpn = np.zeros(n_fraud, dtype=int)
    fraud_ip_proxy = np.zeros(n_fraud, dtype=int)
    fraud_ip_login_count = np.zeros(n_fraud, dtype=int)
    fraud_ip_geo_dist = np.zeros(n_fraud)
    fraud_small_count_6h = np.zeros(n_fraud, dtype=int)
    fraud_cum_amount_6h = np.zeros(n_fraud)
    fraud_same_merchant_1h = np.zeros(n_fraud, dtype=int)

    for i in range(n_fraud):
        subtype = fraud_subtypes[i]

        if subtype == 'large_amount':
            # Classic high-amount fraud
            fraud_amounts[i] = np.clip(np.random.exponential(500), 50, 50000).round(2)
            fraud_avg_amounts[i] = np.clip(np.random.exponential(120), 20, 2000).round(2)
            fraud_txn_counts[i] = np.clip(np.random.poisson(15), 1, 50)
            fraud_merchants[i] = np.random.choice(merchants, p=[0.10, 0.05, 0.05, 0.10, 0.70])
            fraud_ip_country[i] = np.random.choice(['IN', 'IN', 'IN', 'US', 'GB', 'SG'])
            fraud_ip_vpn[i] = int(np.random.random() < 0.25)
            fraud_ip_proxy[i] = int(np.random.random() < 0.10)
            fraud_ip_login_count[i] = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.1])
            fraud_ip_geo_dist[i] = np.clip(np.random.exponential(200), 0, 3000).round(1)
            fraud_small_count_6h[i] = np.random.choice([0, 1, 2])
            fraud_cum_amount_6h[i] = np.clip(np.random.exponential(150), 0, 800).round(2)
            fraud_same_merchant_1h[i] = np.random.choice([0, 1])

        elif subtype == 'small_amount_series':
            # Card testing: many small amounts in rapid succession
            fraud_amounts[i] = np.clip(np.random.exponential(30), 1, 200).round(2)
            fraud_avg_amounts[i] = np.clip(np.random.exponential(100), 20, 1000).round(2)
            fraud_txn_counts[i] = np.random.randint(8, 30)
            fraud_merchants[i] = np.random.choice(merchants, p=[0.15, 0.10, 0.15, 0.10, 0.50])
            fraud_ip_country[i] = np.random.choice(['IN', 'IN', 'US', 'GB'])
            fraud_ip_vpn[i] = int(np.random.random() < 0.35)
            fraud_ip_proxy[i] = int(np.random.random() < 0.15)
            fraud_ip_login_count[i] = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.10])
            fraud_ip_geo_dist[i] = np.clip(np.random.exponential(300), 0, 2000).round(1)
            # Key signals: high small-txn counts and merchant repetition
            fraud_small_count_6h[i] = np.random.randint(5, 20)
            fraud_cum_amount_6h[i] = np.clip(np.random.exponential(600), 200, 3000).round(2)
            fraud_same_merchant_1h[i] = np.random.randint(3, 10)

        elif subtype == 'ip_anomaly':
            # IP-based suspicious activity
            fraud_amounts[i] = np.clip(np.random.exponential(300), 20, 10000).round(2)
            fraud_avg_amounts[i] = np.clip(np.random.exponential(150), 20, 2000).round(2)
            fraud_txn_counts[i] = np.clip(np.random.poisson(8), 1, 30)
            fraud_merchants[i] = np.random.choice(merchants, p=[0.15, 0.10, 0.10, 0.10, 0.55])
            # Key: foreign IPs, VPN, large geo distance, multi-account
            fraud_ip_country[i] = np.random.choice(['US', 'GB', 'SG', 'AE', 'RU', 'CN', 'IN'],
                                                     p=[0.20, 0.15, 0.10, 0.10, 0.15, 0.10, 0.20])
            fraud_ip_vpn[i] = int(np.random.random() < 0.65)
            fraud_ip_proxy[i] = int(np.random.random() < 0.30)
            fraud_ip_login_count[i] = np.random.choice([3, 4, 5, 6, 7, 8], p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.10])
            fraud_ip_geo_dist[i] = np.clip(np.random.exponential(1500), 500, 5000).round(1)
            fraud_small_count_6h[i] = np.random.choice([0, 1, 2, 3])
            fraud_cum_amount_6h[i] = np.clip(np.random.exponential(200), 0, 1000).round(2)
            fraud_same_merchant_1h[i] = np.random.choice([0, 1, 2])

        elif subtype == 'account_takeover':
            # Combined signals: IP anomaly + unusual amounts + high velocity
            fraud_amounts[i] = np.clip(np.random.exponential(800), 100, 30000).round(2)
            fraud_avg_amounts[i] = np.clip(np.random.exponential(100), 20, 500).round(2)
            fraud_txn_counts[i] = np.random.randint(10, 40)
            fraud_merchants[i] = np.random.choice(merchants, p=[0.05, 0.05, 0.05, 0.05, 0.80])
            fraud_ip_country[i] = np.random.choice(['US', 'GB', 'RU', 'CN', 'IN'],
                                                     p=[0.25, 0.20, 0.20, 0.15, 0.20])
            fraud_ip_vpn[i] = int(np.random.random() < 0.70)
            fraud_ip_proxy[i] = int(np.random.random() < 0.35)
            fraud_ip_login_count[i] = np.random.choice([4, 5, 6, 7, 8, 10])
            fraud_ip_geo_dist[i] = np.clip(np.random.exponential(2000), 800, 6000).round(1)
            fraud_small_count_6h[i] = np.random.randint(3, 15)
            fraud_cum_amount_6h[i] = np.clip(np.random.exponential(800), 300, 4000).round(2)
            fraud_same_merchant_1h[i] = np.random.randint(2, 8)

    fraud_df = pd.DataFrame({
        'transaction_id': [f'TXN{i:08d}' for i in range(n_legit, n_samples)],
        'customer_id': np.random.randint(1000, 10000, n_fraud),
        'transaction_amount': fraud_amounts,
        'transaction_time': fraud_times,
        'location': np.random.choice(cities, n_fraud),
        'device_id': np.random.choice(devices, n_fraud),
        'merchant_category': fraud_merchants,
        'account_age_days': fraud_acct_ages,
        'transaction_count_24h': fraud_txn_counts,
        'avg_transaction_amount': fraud_avg_amounts,
        # IP metadata
        'ip_country': fraud_ip_country,
        'ip_is_vpn': fraud_ip_vpn,
        'ip_is_proxy': fraud_ip_proxy,
        'ip_login_count': fraud_ip_login_count,
        'ip_geo_distance_km': fraud_ip_geo_dist,
        # Small-amount context
        'small_txn_count_6h': fraud_small_count_6h,
        'cumulative_amount_6h': fraud_cum_amount_6h,
        'same_merchant_count_1h': fraud_same_merchant_1h,
        # Labels
        'is_fraud': 1,
        'fraud_type': fraud_subtypes,
    })

    # Combine & shuffle
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject a small number of missing values to make it realistic (~0.1%)
    n_missing = int(n_samples * 0.001)
    for col in ['transaction_amount', 'account_age_days', 'avg_transaction_amount']:
        missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == '__main__':
    print("Generating synthetic fraud detection dataset (multi-type)...")
    df = generate_dataset(n_samples=100000, fraud_rate=0.02)

    os.makedirs('data', exist_ok=True)
    output_path = 'data/transactions.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n--- Class Distribution ---")
    fraud_counts = df['is_fraud'].value_counts()
    print(f"   Legitimate (0): {fraud_counts[0]:,}  ({fraud_counts[0]/len(df)*100:.2f}%)")
    print(f"   Fraud      (1): {fraud_counts[1]:,}  ({fraud_counts[1]/len(df)*100:.2f}%)")
    print(f"   Imbalance Ratio: {fraud_counts[0]/fraud_counts[1]:.0f}:1")

    print(f"\n--- Fraud Type Distribution ---")
    type_counts = df['fraud_type'].value_counts()
    for ftype, count in type_counts.items():
        print(f"   {ftype}: {count:,}  ({count/len(df)*100:.2f}%)")

    print(f"\n--- Sample Rows ---")
    print(df.head(10).to_string(index=False))
    print(f"\n--- Missing Values ---")
    print(df.isnull().sum().to_string())
