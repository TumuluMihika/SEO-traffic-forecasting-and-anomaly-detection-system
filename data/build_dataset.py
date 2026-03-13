"""
Dataset Builder
---------------
Combines real daily traffic data (real.csv) with SEO feature distributions
derived from a real e-commerce SEO audit dataset (data.csv).

The traffic time series is 100% real (2167 days, Sep 2014 - Aug 2020).
SEO features (CTR, Position, Impressions, BounceRate) are synthesized using
real statistical distributions from data.csv, correlated with actual traffic
patterns to ensure ecological validity.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

def build(real_csv_path, seo_csv_path, output_path):
    # ── Load real traffic data ────────────────────────────────────────────────
    df = pd.read_csv(real_csv_path)
    for col in ['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        )
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # ── Load real SEO audit data for distributions ────────────────────────────
    seo = pd.read_csv(seo_csv_path, sep=';')
    seo['Position']    = pd.to_numeric(seo['Position'].astype(str).str.replace(',', '.'), errors='coerce')
    seo['BounceRate']  = pd.to_numeric(seo['BounceRate'].astype(str).str.replace(',', '.'), errors='coerce')
    seo['Impressions'] = pd.to_numeric(seo['Impressions'], errors='coerce')
    seo['Clicks']      = pd.to_numeric(seo['Clicks'], errors='coerce')
    seo = seo.dropna(subset=['Position', 'BounceRate', 'Impressions', 'Clicks'])

    n = len(df)
    traffic       = df['Page.Loads'].values.astype(float)
    traffic_norm  = (traffic - traffic.min()) / (traffic.max() - traffic.min())  # 0–1

    # ── Synthesize Avg_Position (correlated: high traffic → low position number) ─
    pos_mean  = seo['Position'].mean()     # 12.9
    pos_std   = seo['Position'].std()      # 7.6
    base_pos  = pos_mean - 4 * traffic_norm   # range ~8.9–12.9
    noise_pos = np.random.normal(0, pos_std * 0.4, n)
    avg_position = np.clip(base_pos + noise_pos, 1.0, 56.0).round(1)

    # ── Synthesize Impressions (correlated with traffic) ──────────────────────
    imp_scale = traffic_norm * 4500 + 800
    noise_imp = np.random.normal(0, 400, n)
    impressions = np.clip(imp_scale + noise_imp, 50, 42000).astype(int)

    # ── Synthesize CTR (correlated: better position → higher CTR) ─────────────
    # CTR roughly = 30% / position^0.9  (industry approximation)
    base_ctr = 0.30 / (avg_position ** 0.9)
    noise_ctr = np.random.normal(0, 0.005, n)
    ctr = np.clip(base_ctr + noise_ctr, 0.001, 0.35).round(4)

    # ── Synthesize BounceRate (anti-correlated with traffic depth) ────────────
    br_mean  = seo['BounceRate'].mean()   # 0.317
    br_std   = seo['BounceRate'].std()    # 0.136
    base_br  = br_mean + 0.08 * (1 - traffic_norm)
    noise_br = np.random.normal(0, br_std * 0.3, n)
    bounce_rate = np.clip(base_br + noise_br, 0.0, 1.0).round(4)

    # ── Add engineered time features ──────────────────────────────────────────
    df['Day_of_Week']  = df['ds'].dt.dayofweek          # 0=Mon 6=Sun
    df['Is_Weekend']   = (df['Day_of_Week'] >= 5).astype(int)
    df['Month']        = df['ds'].dt.month

    # ── Lag features (temporal dependencies) ─────────────────────────────────
    df['Traffic_Lag7']      = df['Page.Loads'].shift(7)
    df['Traffic_Lag14']     = df['Page.Loads'].shift(14)
    df['Rolling_Mean_7']    = df['Page.Loads'].shift(1).rolling(7).mean()
    df['Rolling_Std_7']     = df['Page.Loads'].shift(1).rolling(7).std()

    # ── Google Algorithm Update Flags (known major updates) ──────────────────
    # Sources: Moz Google Algorithm Change History
    algo_dates = [
        '2015-02-04',  # Google Phantom
        '2015-04-21',  # Mobilegeddon
        '2015-07-17',  # Panda 4.2
        '2016-01-08',  # Core Quality Update
        '2016-09-27',  # Penguin 4.0
        '2017-03-07',  # Fred Update
        '2017-07-09',  # Intrusive Interstitials
        '2018-03-09',  # Broad Core Update
        '2018-08-01',  # Medic Update
        '2019-03-12',  # Core Update March
        '2019-06-03',  # Core Update June
        '2019-09-24',  # Core Update September
        '2020-01-13',  # Core Update January
        '2020-05-04',  # Core Update May
    ]
    algo_dates_dt = pd.to_datetime(algo_dates)
    df['Algorithm_Flag'] = df['ds'].isin(algo_dates_dt).astype(int)

    # ── Attach synthesized SEO columns ────────────────────────────────────────
    df['Avg_Position']  = avg_position
    df['Impressions']   = impressions
    df['CTR']           = ctr
    df['Bounce_Rate']   = bounce_rate

    # ── Rename for clarity ────────────────────────────────────────────────────
    df = df.rename(columns={
        'ds':               'Date',
        'Page.Loads':       'Organic_Traffic',
        'Unique.Visits':    'Unique_Visits',
        'First.Time.Visits':'First_Time_Visits',
        'Returning.Visits': 'Returning_Visits',
    })

    # ── Drop rows with NaN from lag computation ───────────────────────────────
    df = df.dropna().reset_index(drop=True)

    # ── Select final columns ──────────────────────────────────────────────────
    final_cols = [
        'Date', 'Organic_Traffic', 'Unique_Visits', 'First_Time_Visits',
        'Returning_Visits', 'Impressions', 'CTR', 'Avg_Position',
        'Bounce_Rate', 'Day_of_Week', 'Is_Weekend', 'Month',
        'Traffic_Lag7', 'Traffic_Lag14', 'Rolling_Mean_7', 'Rolling_Std_7',
        'Algorithm_Flag'
    ]
    df = df[final_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Algorithm update days: {df['Algorithm_Flag'].sum()}")
    return df


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(
        real_csv_path=os.path.join(base, 'data', 'real.csv'),
        seo_csv_path=os.path.join(base, 'data', 'data.csv'),
        output_path=os.path.join(base, 'data', 'seo_traffic.csv')
    )
