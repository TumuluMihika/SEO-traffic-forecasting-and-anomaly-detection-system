"""
SARIMA Model  (Baseline 1)
--------------------------
Implements Seasonal ARIMA using STL decomposition + ARIMA-style
autoregressive fitting via numpy/scipy (no statsmodels required).

Architecture inspired by: Ahmed et al. (NILES 2024) — SARIMA+Informer hybrid.
Here SARIMA acts as the seasonal/residual forecaster (same role as in that paper).

STL Decomposition:
  - Trend:    extracted via centred moving average (window=7)
  - Seasonal: weekly pattern averaged across all observed weeks
  - Residual: original – trend – seasonal
"""

import numpy as np
from scipy.signal import medfilt


class SARIMAModel:
    def __init__(self, seasonal_period=7, ar_lags=14):
        self.seasonal_period = seasonal_period
        self.ar_lags         = ar_lags
        self.seasonal_pattern = None   # shape (7,)
        self.ar_coeffs        = None
        self.ar_intercept     = None
        self.trend_slope      = None
        self.trend_intercept  = None
        self._fitted          = False

    # ── STL decomposition ─────────────────────────────────────────────────────
    def _decompose(self, series):
        n   = len(series)
        w   = self.seasonal_period
        # Trend: centred moving average
        trend = np.convolve(series, np.ones(w) / w, mode='same')
        # Edge correction — replicate boundary
        half = w // 2
        for i in range(half):
            trend[i]        = np.mean(series[:i + half + 1])
            trend[n - 1 - i] = np.mean(series[n - i - half - 1:])

        detrended = series - trend

        # Seasonal: average for each position in the cycle
        seasonal = np.zeros(n)
        period_avgs = np.zeros(w)
        for k in range(w):
            indices = np.arange(k, n, w)
            period_avgs[k] = np.mean(detrended[indices])
        # Remove mean so seasonal sums to zero
        period_avgs -= period_avgs.mean()
        for i in range(n):
            seasonal[i] = period_avgs[i % w]

        residual = series - trend - seasonal
        return trend, seasonal, residual, period_avgs

    # ── AR model on residuals ─────────────────────────────────────────────────
    def _fit_ar(self, residuals):
        lags = self.ar_lags
        n    = len(residuals)
        if n <= lags:
            lags = max(1, n // 3)
            self.ar_lags = lags

        X, y = [], []
        for i in range(lags, n):
            X.append(residuals[i - lags:i])
            y.append(residuals[i])
        X = np.array(X)
        y = np.array(y)

        # OLS: coefficients via pseudo-inverse
        X_b = np.hstack([np.ones((len(X), 1)), X])
        try:
            coeffs = np.linalg.lstsq(X_b, y, rcond=None)[0]
        except Exception:
            coeffs = np.zeros(lags + 1)
        self.ar_intercept = coeffs[0]
        self.ar_coeffs    = coeffs[1:]
        return residuals   # return for reference

    # ── Trend projection (linear) ─────────────────────────────────────────────
    def _fit_trend(self, trend):
        t = np.arange(len(trend))
        coeffs = np.polyfit(t, trend, 1)
        self.trend_slope     = coeffs[0]
        self.trend_intercept = coeffs[1]

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, series):
        series = np.array(series, dtype=float)
        trend, seasonal, residual, period_avgs = self._decompose(series)
        self.seasonal_pattern = period_avgs
        self._fit_trend(trend)
        self._fit_ar(residual)
        self._train_series = series
        self._train_residuals = residual
        self._fitted = True
        return self

    # ── Forecast ──────────────────────────────────────────────────────────────
    def predict(self, steps, last_index=None):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        n = len(self._train_series)
        if last_index is None:
            last_index = n - 1

        # Project trend
        t_vals   = np.arange(last_index + 1, last_index + 1 + steps)
        trend_fc = self.trend_slope * t_vals + self.trend_intercept

        # Project seasonal
        seasonal_fc = np.array([
            self.seasonal_pattern[(last_index + 1 + i) % self.seasonal_period]
            for i in range(steps)
        ])

        # AR forecast on residuals
        residual_buf = list(self._train_residuals[-self.ar_lags:])
        residual_fc  = []
        for _ in range(steps):
            x_in  = np.array(residual_buf[-self.ar_lags:])
            r_hat = self.ar_intercept + np.dot(self.ar_coeffs, x_in)
            residual_fc.append(r_hat)
            residual_buf.append(r_hat)

        forecast = trend_fc + seasonal_fc + np.array(residual_fc)
        return np.maximum(forecast, 0)

    # ── In-sample fitted values ───────────────────────────────────────────────
    def fitted_values(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        n    = len(self._train_series)
        lags = self.ar_lags
        _, seasonal, residual, _ = self._decompose(self._train_series)
        trend_vals = (self.trend_slope * np.arange(n) + self.trend_intercept)

        fitted_res = np.zeros(n)
        for i in range(lags, n):
            x_in = residual[i - lags:i]
            fitted_res[i] = self.ar_intercept + np.dot(self.ar_coeffs, x_in)

        fitted = trend_vals + seasonal + fitted_res
        return np.maximum(fitted, 0)

    def get_decomposition(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        t, s, r, _ = self._decompose(self._train_series)
        return {'trend': t, 'seasonal': s, 'residual': r}
