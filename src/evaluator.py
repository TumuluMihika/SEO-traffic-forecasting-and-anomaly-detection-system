"""
Evaluator
---------
Computes RMSE, MAE, MAPE, SMAPE and R² for each model.
These five metrics match those used across all four compared papers,
enabling direct quantitative comparison.

Paper coverage:
  RMSE  — Ahmed et al. (2024), Samaan et al. (2025)
  MAE   — Ahmed et al. (2024), Samaan et al. (2025)
  MAPE  — Sikka & Kumar (2023), Ahmed et al. (2024)
  SMAPE — Ahmed et al. (2024) [primary metric]
  R²    — Samaan et al. (2025), Sikka & Kumar (2023)
"""

import numpy as np


class Evaluator:
    def __init__(self):
        self.results = {}

    def _rmse(self, y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _mae(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    def _mape(self, y_true, y_pred):
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def _smape(self, y_true, y_pred):
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask  = denom != 0
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

    def _r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-8:
            return 1.0
        return float(1 - ss_res / ss_tot)

    def evaluate(self, model_name, y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        metrics = {
            'RMSE':  round(self._rmse(y_true, y_pred), 4),
            'MAE':   round(self._mae(y_true, y_pred), 4),
            'MAPE':  round(self._mape(y_true, y_pred), 4),
            'SMAPE': round(self._smape(y_true, y_pred), 4),
            'R2':    round(self._r2(y_true, y_pred), 6),
        }
        self.results[model_name] = metrics
        return metrics

    def compare_table(self):
        """Returns list of dicts for all models — used by Flask API."""
        rows = []
        for model, m in self.results.items():
            rows.append({'model': model, **m})
        return rows

    def best_model(self):
        if not self.results:
            return None
        return min(self.results, key=lambda k: self.results[k]['RMSE'])
