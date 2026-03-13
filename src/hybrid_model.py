"""
Hybrid SARIMA + GRU Model  (Main Contribution)
------------------------------------------------
Combines SARIMA (seasonal/linear component) with the GRU-style MLP
(non-linear multivariate component) using STL decomposition.

Inspired by: Ahmed et al. (NILES 2024) who combined SARIMA + Informer.
Key difference: we use SEO-specific multivariate features in the GRU
component and add anomaly cause classification — neither present in
any of the four compared papers.

Hybrid combination:
    Final_Forecast = w_sarima * SARIMA_forecast + w_gru * GRU_forecast

Weights are learned via OLS on a validation window, or default 0.4/0.6.
"""

import numpy as np


class HybridModel:
    def __init__(self, sarima_model, gru_model, preprocessor):
        self.sarima      = sarima_model
        self.gru         = gru_model
        self.prep        = preprocessor
        self.w_sarima    = 0.40
        self.w_gru       = 0.60
        self._fitted     = False

    # ── Optimise weights on validation portion ────────────────────────────────
    def fit_weights(self, sarima_preds, gru_preds, actuals):
        """
        OLS to find optimal w such that: actuals ≈ w*sarima + (1-w)*gru
        """
        A = sarima_preds - gru_preds
        b = actuals - gru_preds
        if np.dot(A, A) < 1e-8:
            self.w_sarima = 0.40
        else:
            w = np.dot(A, b) / np.dot(A, A)
            self.w_sarima = float(np.clip(w, 0.05, 0.95))
        self.w_gru  = 1.0 - self.w_sarima
        self._fitted = True
        return self.w_sarima, self.w_gru

    # ── Combined forecast ─────────────────────────────────────────────────────
    def combine(self, sarima_preds, gru_preds_scaled):
        gru_preds = self.prep.inverse_y(gru_preds_scaled)
        sarima_arr = np.array(sarima_preds)
        gru_arr    = np.array(gru_preds)
        combined   = self.w_sarima * sarima_arr + self.w_gru * gru_arr
        return np.maximum(combined, 0)

    # ── In-sample hybrid on test set ──────────────────────────────────────────
    def predict_combined(self, sarima_preds, gru_preds_scaled):
        return self.combine(sarima_preds, gru_preds_scaled)

    @property
    def weights(self):
        return {'sarima': round(self.w_sarima, 4), 'gru': round(self.w_gru, 4)}
