"""
Anomaly Detector + SEO Cause Classifier  ← NOVEL CONTRIBUTION
--------------------------------------------------------------
Detects traffic anomalies using residual analysis (standard statistics)
then classifies each anomaly into one of 5 SEO-specific cause categories.

The 5-class cause classification is the KEY NOVEL CONTRIBUTION:
none of the four compared papers (Ahmed 2024, Samaan 2025, Sikka 2023,
Shelatkar 2020) perform any form of anomaly detection, let alone
cause attribution.

Classification Logic:
─────────────────────
Anomaly: |residual| > threshold * rolling_std

Cause classes:
  1. Algorithm Update Impact  — residual < −threshold AND algo_flag = 1
  2. Ranking Drop             — residual < −threshold AND position worsened
  3. Click Loss Anomaly       — residual < −threshold AND CTR below rolling mean
  4. Traffic Spike            — residual > +threshold
  5. Unexplained Drop         — residual < −threshold, no clear SEO signal
  0. Normal                   — |residual| ≤ threshold

Severity scoring (novel):
  Score = |residual| / rolling_std
  Low: 2–3σ  |  Medium: 3–5σ  |  High: >5σ
"""

import numpy as np
import pandas as pd


CAUSE_LABELS = {
    0: 'Normal',
    1: 'Algorithm Update Impact',
    2: 'Ranking Drop',
    3: 'Click Loss Anomaly',
    4: 'Traffic Spike',
    5: 'Unexplained Drop'
}

SEVERITY_LABELS = {
    'low':    'Low (2–3σ)',
    'medium': 'Medium (3–5σ)',
    'high':   'High (>5σ)'
}


class AnomalyDetector:
    def __init__(self, threshold_sigma=2.0, window=30):
        self.threshold_sigma = threshold_sigma
        self.window          = window

    # ── Core detection ────────────────────────────────────────────────────────
    def detect(self, actuals, predicted, df_features):
        """
        actuals      : 1-D array of true Organic_Traffic
        predicted    : 1-D array of hybrid model predictions
        df_features  : DataFrame with columns:
                       Algorithm_Flag, Avg_Position, CTR,
                       Rolling_Mean_7, Date

        Returns DataFrame with one row per anomaly.
        """
        actuals   = np.array(actuals, dtype=float)
        predicted = np.array(predicted, dtype=float)
        residuals = actuals - predicted
        n         = len(residuals)

        # Rolling std of residuals
        roll_std = pd.Series(residuals).rolling(
            window=self.window, min_periods=5
        ).std().bfill().ffill().values

        # Rolling mean of CTR for comparison
        ctr_vals  = df_features['CTR'].values if 'CTR' in df_features else np.zeros(n)
        ctr_roll  = pd.Series(ctr_vals).rolling(
            window=14, min_periods=3
        ).mean().bfill().ffill().values

        # Rolling mean of Avg_Position
        pos_vals  = df_features['Avg_Position'].values if 'Avg_Position' in df_features else np.full(n, 15.0)
        pos_roll  = pd.Series(pos_vals).rolling(
            window=14, min_periods=3
        ).mean().bfill().ffill().values

        algo_flags = df_features['Algorithm_Flag'].values if 'Algorithm_Flag' in df_features else np.zeros(n)
        dates      = df_features['Date'].values if 'Date' in df_features else np.arange(n)

        records = []
        for i in range(n):
            std_i   = max(roll_std[i], 1e-3)
            sigma_i = residuals[i] / std_i
            cause, severity = self._classify(
                sigma_i, algo_flags[i],
                pos_vals[i], pos_roll[i],
                ctr_vals[i], ctr_roll[i]
            )
            if cause != 0:
                records.append({
                    'date':          str(dates[i])[:10],
                    'actual':        round(float(actuals[i]), 2),
                    'predicted':     round(float(predicted[i]), 2),
                    'residual':      round(float(residuals[i]), 2),
                    'sigma':         round(float(sigma_i), 3),
                    'cause_id':      cause,
                    'cause_label':   CAUSE_LABELS[cause],
                    'severity':      severity,
                    'algo_flag':     int(algo_flags[i]),
                    'avg_position':  round(float(pos_vals[i]), 1),
                    'ctr':           round(float(ctr_vals[i]), 4)
                })

        anomaly_df = pd.DataFrame(records)
        return anomaly_df

    # ── Classification logic ──────────────────────────────────────────────────
    def _classify(self, sigma, algo_flag, pos_now, pos_mean, ctr_now, ctr_mean):
        t = self.threshold_sigma

        # Severity
        abs_s = abs(sigma)
        if abs_s <= t:
            return 0, 'normal'

        if abs_s <= 3:
            severity = 'low'
        elif abs_s <= 5:
            severity = 'medium'
        else:
            severity = 'high'

        # Spike
        if sigma > t:
            return 4, severity

        # Drop causes (sigma < -t)
        if algo_flag == 1:
            return 1, severity

        if pos_now > pos_mean * 1.10:   # position number worsened >10%
            return 2, severity

        if ctr_now < ctr_mean * 0.80:   # CTR dropped >20%
            return 3, severity

        return 5, severity   # unexplained

    # ── Summary stats ─────────────────────────────────────────────────────────
    def summarise(self, anomaly_df):
        if anomaly_df.empty:
            return {}
        total = len(anomaly_df)
        by_cause = anomaly_df['cause_label'].value_counts().to_dict()
        by_severity = anomaly_df['severity'].value_counts().to_dict()
        return {
            'total_anomalies':  total,
            'by_cause':         by_cause,
            'by_severity':      by_severity,
            'cause_labels':     CAUSE_LABELS
        }
