"""
Preprocessor
------------
Loads seo_traffic.csv, scales features, creates sequences for the
neural network, and produces train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle, os

FEATURE_COLS = [
    'Organic_Traffic', 'Unique_Visits', 'First_Time_Visits',
    'Returning_Visits', 'Impressions', 'CTR', 'Avg_Position',
    'Bounce_Rate', 'Day_of_Week', 'Is_Weekend', 'Month',
    'Traffic_Lag7', 'Traffic_Lag14', 'Rolling_Mean_7',
    'Rolling_Std_7', 'Algorithm_Flag'
]
TARGET_COL   = 'Organic_Traffic'
SEQUENCE_LEN = 30   # 30-day lookback window for neural network
TEST_SPLIT   = 0.20 # last 20% as test set


class Preprocessor:
    def __init__(self, csv_path):
        self.csv_path      = csv_path
        self.scaler_X      = MinMaxScaler()
        self.scaler_y      = MinMaxScaler()
        self.df            = None
        self.feature_cols  = FEATURE_COLS
        self.target_col    = TARGET_COL
        self.sequence_len  = SEQUENCE_LEN

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self):
        df = pd.read_csv(self.csv_path, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.dropna()
        self.df = df
        return df

    # ── Train / Test split ────────────────────────────────────────────────────
    def split(self):
        n       = len(self.df)
        split_i = int(n * (1 - TEST_SPLIT))
        train   = self.df.iloc[:split_i].copy()
        test    = self.df.iloc[split_i:].copy()
        return train, test

    # ── Scale features ────────────────────────────────────────────────────────
    def fit_transform(self, train_df):
        X_train = self.scaler_X.fit_transform(train_df[self.feature_cols])
        y_train = self.scaler_y.fit_transform(
            train_df[[self.target_col]]
        ).ravel()
        return X_train, y_train

    def transform(self, df):
        X = self.scaler_X.transform(df[self.feature_cols])
        y = self.scaler_y.transform(df[[self.target_col]]).ravel()
        return X, y

    def inverse_y(self, y_scaled):
        return self.scaler_y.inverse_transform(
            np.array(y_scaled).reshape(-1, 1)
        ).ravel()

    # ── Sequence builder (for MLP sliding window) ─────────────────────────────
    def make_sequences(self, X, y, seq_len=None):
        if seq_len is None:
            seq_len = self.sequence_len
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i].ravel())   # flatten window
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    # ── Get raw target series (unscaled) ──────────────────────────────────────
    def get_traffic_series(self):
        return self.df[self.target_col].values.astype(float)

    def get_dates(self):
        return self.df['Date'].values

    def save_scalers(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'X': self.scaler_X, 'y': self.scaler_y}, f)

    def load_scalers(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.scaler_X = d['X']
        self.scaler_y = d['y']
