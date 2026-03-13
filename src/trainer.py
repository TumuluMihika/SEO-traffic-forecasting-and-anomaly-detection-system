"""
Model Trainer
-------------
Orchestrates the full training pipeline:
  1. Load + preprocess data
  2. Train SARIMA baseline
  3. Train GRU-style MLP baseline
  4. Train hybrid combiner
  5. Detect + classify anomalies
  6. Evaluate all three models
  7. Cache results for Flask API
"""

import numpy as np
import json, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

from preprocessor    import Preprocessor
from sarima_model    import SARIMAModel
from gru_model       import GRUModel
from hybrid_model    import HybridModel
from anomaly_detector import AnomalyDetector
from evaluator       import Evaluator


DATA_PATH  = os.path.join(BASE, 'data', 'seo_traffic.csv')
CACHE_PATH = os.path.join(BASE, 'data', 'results_cache.json')


def train_all(csv_path=None, progress_cb=None):
    def log(msg):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    log("Loading and preprocessing data...")
    data_path = csv_path or DATA_PATH
    prep = Preprocessor(data_path)
    prep.load()
    train_df, test_df = prep.split()

    X_train_raw, y_train_raw = prep.fit_transform(train_df)
    X_test_raw,  y_test_raw  = prep.transform(test_df)

    # Build sequences for GRU
    X_train_seq, y_train_seq = prep.make_sequences(X_train_raw, y_train_raw)
    X_test_seq,  y_test_seq  = prep.make_sequences(X_test_raw,  y_test_raw)

    traffic_full   = prep.get_traffic_series()
    n_train        = len(train_df)
    traffic_train  = traffic_full[:n_train]
    traffic_test   = traffic_full[n_train:]

    # Dates
    dates_full  = prep.get_dates()
    dates_test  = dates_full[n_train:]

    # ── SARIMA ────────────────────────────────────────────────────────────────
    log("Training SARIMA model...")
    sarima = SARIMAModel(seasonal_period=7, ar_lags=14)
    sarima.fit(traffic_train)
    sarima_test_pred = sarima.predict(steps=len(traffic_test))

    # ── GRU ───────────────────────────────────────────────────────────────────
    log("Training GRU model (this takes ~30s)...")
    gru = GRUModel(seq_len=30, hidden_layers=(128, 64, 32), max_iter=300)
    gru.fit(X_train_seq, y_train_seq)
    gru_test_pred_scaled = gru.predict(X_test_seq)
    gru_test_pred = prep.inverse_y(gru_test_pred_scaled)

    # Align lengths — sequences start seq_len steps in
    seq_len  = prep.sequence_len
    sarima_aligned = sarima_test_pred[seq_len:]
    actuals_aligned = traffic_test[seq_len:]
    dates_aligned   = dates_test[seq_len:]
    test_df_aligned = test_df.iloc[seq_len:].reset_index(drop=True)

    # ── Hybrid ────────────────────────────────────────────────────────────────
    log("Fitting hybrid weights...")
    hybrid = HybridModel(sarima, gru, prep)
    hybrid.fit_weights(sarima_aligned, gru_test_pred, actuals_aligned)
    hybrid_pred = hybrid.predict_combined(sarima_aligned, gru_test_pred_scaled)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log("Computing evaluation metrics...")
    ev = Evaluator()
    sarima_metrics = ev.evaluate('SARIMA',           actuals_aligned, sarima_aligned)
    gru_metrics    = ev.evaluate('GRU (Multivariate)', actuals_aligned, gru_test_pred)
    hybrid_metrics = ev.evaluate('SARIMA+GRU Hybrid', actuals_aligned, hybrid_pred)

    log("Detecting and classifying anomalies...")
    detector = AnomalyDetector(threshold_sigma=2.0, window=30)
    anomaly_df = detector.detect(actuals_aligned, hybrid_pred, test_df_aligned)
    anomaly_summary = detector.summarise(anomaly_df)

    # ── Format time series for dashboard ─────────────────────────────────────
    dates_str  = [str(d)[:10] for d in dates_aligned]

    results = {
        'dates':        dates_str,
        'actuals':      actuals_aligned.tolist(),
        'sarima':       np.maximum(sarima_aligned, 0).tolist(),
        'gru':          np.maximum(gru_test_pred, 0).tolist(),
        'hybrid':       np.maximum(hybrid_pred, 0).tolist(),
        'metrics':      ev.compare_table(),
        'anomalies':    anomaly_df.to_dict('records') if not anomaly_df.empty else [],
        'anomaly_summary': anomaly_summary,
        'hybrid_weights':  hybrid.weights,
        'dataset_info': {
            'total_rows':    len(prep.df),
            'train_rows':    n_train,
            'test_rows':     len(test_df),
            'date_start':    str(prep.df['Date'].min())[:10],
            'date_end':      str(prep.df['Date'].max())[:10],
            'n_features':    len(prep.feature_cols),
            'algo_updates':  int(prep.df['Algorithm_Flag'].sum()),
        },
        'training_complete': True
    }

    with open(CACHE_PATH, 'w') as f:
        json.dump(results, f, default=str)

    log(f"Training complete. Best model: {ev.best_model()}")
    log(f"SARIMA  → RMSE: {sarima_metrics['RMSE']:.2f}  R²: {sarima_metrics['R2']:.4f}")
    log(f"GRU     → RMSE: {gru_metrics['RMSE']:.2f}  R²: {gru_metrics['R2']:.4f}")
    log(f"Hybrid  → RMSE: {hybrid_metrics['RMSE']:.2f}  R²: {hybrid_metrics['R2']:.4f}")
    log(f"Anomalies found: {len(anomaly_df)}")

    return results


if __name__ == '__main__':
    train_all()
