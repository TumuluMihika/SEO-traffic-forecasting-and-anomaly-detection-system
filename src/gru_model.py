"""
GRU-style Neural Network Model  (Baseline 2)
---------------------------------------------
Implements a deep MLP with sliding window (mimics GRU temporal modelling)
using scikit-learn MLPRegressor — no TensorFlow required.

Architecture inspired by: Samaan et al. (Results in Engineering 2025)
— CNN/LSTM/GRU on web traffic. Here MLP with temporal window serves
the same role: capturing non-linear multivariate patterns.

Key novelty over the paper: we use 16 SEO-specific features including
CTR, Avg_Position, Impressions, BounceRate and Algorithm_Flag —
all absent from every compared paper.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class GRUModel:
    """
    Sliding-window MLP that approximates recurrent behaviour.
    Input:  flattened window of (seq_len × n_features)
    Output: next-day Organic_Traffic (scaled)
    """

    def __init__(self, seq_len=30, hidden_layers=(128, 64, 32),
                 max_iter=500, learning_rate=0.001):
        self.seq_len       = seq_len
        self.hidden_layers = hidden_layers
        self.max_iter      = max_iter
        self.learning_rate = learning_rate
        self._model        = None
        self._fitted       = False
        self._n_features   = None

    def _build(self, input_dim):
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False
        )

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X_seq, y_seq):
        """
        X_seq: (n_samples, seq_len * n_features)  — from preprocessor.make_sequences
        y_seq: (n_samples,)
        """
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        self._n_features = X_seq.shape[1]
        self._build(X_seq.shape[1])
        self._model.fit(X_seq, y_seq)
        self._X_train = X_seq
        self._y_train = y_seq
        self._fitted  = True
        return self

    # ── Predict on prepared sequences ────────────────────────────────────────
    def predict(self, X_seq):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(np.array(X_seq))

    # ── Multi-step forecast ───────────────────────────────────────────────────
    def forecast(self, last_window, steps):
        """
        last_window: (seq_len, n_features) — last known window (scaled)
        steps: number of days ahead
        Returns array of scaled predictions.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        preds  = []
        window = last_window.copy()   # (seq_len, n_features)

        for _ in range(steps):
            x_in  = window.ravel().reshape(1, -1)
            y_hat = self._model.predict(x_in)[0]
            preds.append(y_hat)
            # Slide window: drop oldest row, append new row
            new_row       = window[-1].copy()
            new_row[0]    = y_hat   # update traffic feature (index 0)
            window        = np.vstack([window[1:], new_row])

        return np.array(preds)

    # ── Training loss curve ───────────────────────────────────────────────────
    def loss_curve(self):
        if not self._fitted:
            return []
        return list(self._model.loss_curve_)

    def validation_curve(self):
        if not self._fitted:
            return []
        vc = getattr(self._model, 'validation_scores_', [])
        return list(vc)

    @property
    def n_iter(self):
        if not self._fitted:
            return 0
        return self._model.n_iter_
