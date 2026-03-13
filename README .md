# SEO Traffic Forecasting & Anomaly Detection System

A full-stack machine learning application that forecasts daily organic website traffic and automatically detects + classifies the **cause** of traffic anomalies — using a hybrid SARIMA + Neural Network model with 16 multivariate SEO features.

> Built by a 3rd Year B.Tech CSE student (KIIT University, 2023–2027). Directly comparable to 4 published research papers (Ahmed et al. 2024, Samaan et al. 2025, Sikka & Kumar 2023, Shelatkar et al. 2020).

---

## The Problem

SEO teams always know **when** traffic drops. They rarely know **why**.

Standard analytics tools tell you traffic fell 28% on Monday. They don't tell you if it was a Google algorithm update, a keyword ranking drop, a CTR collapse, or just statistical noise. This system solves both problems — forecast + explain.

---

## What It Does

- **Forecasts** daily organic traffic up to N days ahead using three models
- **Detects** anomalies automatically using residual-based statistical thresholding (±2σ)
- **Classifies** each anomaly into one of 5 SEO-specific cause categories
- **Scores severity** as Low / Medium / High based on how many standard deviations the anomaly deviates
- **Deploys** as a Flask web app with an interactive dashboard and CSV upload

---

## Models

| Model | Role | R² | RMSE |
|---|---|---|---|
| SARIMA (Baseline 1) | Seasonal + residual component | 0.2691 | 1011.45 |
| GRU-style MLP (Baseline 2) | Multivariate non-linear patterns | 0.8837 | 403.47 |
| **SARIMA + GRU Hybrid** | **Main contribution** | **0.8841** | **402.79** |

> Evaluated on a strict forward-looking 80/20 temporal train-test split. No data leakage.

---

## Anomaly Classification — Novel Contribution

None of the 4 compared papers perform anomaly detection. This system introduces **5-class cause attribution**:

| Class | Detection Logic | What It Means |
|---|---|---|
| Algorithm Update Impact | residual < −2σ AND algo_flag = 1 | Google Core Update degraded rankings |
| Ranking Drop | residual < −2σ AND position worsened >10% | Keywords fell in SERPs |
| Click Loss Anomaly | residual < −2σ AND CTR < 80% of rolling mean | Titles/meta-descriptions underperforming |
| Traffic Spike | residual > +2σ | Viral content or campaign success |
| Unexplained Drop | residual < −2σ, no SEO signal matches | Technical issue or external factor |

**Severity scoring:** `Score = |residual| / rolling_std`
- Low: 2–3σ → Monitor
- Medium: 3–5σ → Investigate within 24 hours
- High: >5σ → Immediate action required

---

## Dataset

| Property | Detail |
|---|---|
| Source | statforecasting.com (same as Sikka & Kumar 2023) |
| Size | 2,153 daily observations |
| Date range | September 2014 – August 2020 |
| Features | 18 columns (16 model inputs) |
| Traffic columns | 100% real daily observations |
| SEO features | Synthesised from real e-commerce SEO audit distributions (9,439 pages) |

**16 Input Features:**
`Organic_Traffic`, `Unique_Visits`, `First_Time_Visits`, `Returning_Visits`,
`Impressions`, `CTR`, `Avg_Position`, `Bounce_Rate`,
`Day_of_Week`, `Is_Weekend`, `Month`,
`Traffic_Lag7`, `Traffic_Lag14`, `Rolling_Mean_7`, `Rolling_Std_7`, `Algorithm_Flag`

---

## Architecture

```
real.csv + data.csv
        │
        ▼
  build_dataset.py          ← enriches real traffic with SEO feature distributions
        │
        ▼
   preprocessor.py          ← MinMaxScaler, 30-day sequences, 80/20 temporal split
        │
   ┌────┴────┐
   ▼         ▼
sarima_model  gru_model     ← SARIMA (seasonal AR14) + MLP (128→64→32, ReLU, Adam)
   │         │
   └────┬────┘
        ▼
  hybrid_model              ← OLS-optimised weights (SARIMA 40%, GRU 60%)
        │
        ▼
anomaly_detector            ← residual analysis → 5-class cause classification
        │
        ▼
   evaluator                ← RMSE, MAE, MAPE, SMAPE, R²
        │
        ▼
     app.py                 ← Flask REST API
        │
        ▼
  templates/index.html      ← Chart.js interactive dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (Python) |
| Statistical Model | NumPy + SciPy (STL decomposition + AR) |
| Neural Network | scikit-learn MLPRegressor (no GPU required) |
| Data Processing | Pandas + NumPy |
| Frontend | Vanilla JS + Chart.js |
| API | REST + JSON |

---

## Project Structure

```
seo_project/
├── app.py                      # Flask app — all API endpoints
├── data/
│   ├── real.csv                # Real daily traffic data (2,153 rows)
│   ├── data.csv                # Real SEO audit dataset (9,439 pages)
│   ├── seo_traffic.csv         # Enriched dataset (auto-generated)
│   └── build_dataset.py        # Dataset builder — run once
├── src/
│   ├── preprocessor.py         # Data loading, scaling, sequence builder
│   ├── sarima_model.py         # STL decomposition + AR(14) forecaster
│   ├── gru_model.py            # Sliding-window MLP neural network
│   ├── hybrid_model.py         # OLS-weighted combination
│   ├── anomaly_detector.py     # 5-class cause classifier + severity scoring
│   ├── evaluator.py            # RMSE, MAE, MAPE, SMAPE, R²
│   └── trainer.py              # Full pipeline orchestrator
└── templates/
    └── index.html              # Interactive dashboard
```

---

## Installation & Running

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/seo-traffic-forecasting.git
cd seo-traffic-forecasting/seo_project
```

### 2. Install dependencies
```bash
pip install flask scikit-learn pandas numpy scipy openpyxl
```

### 3. Build the enriched dataset (run once)
```bash
python data/build_dataset.py
```

### 4. Start the Flask app
```bash
python app.py
```

### 5. Open the dashboard
```
http://localhost:5000
```
Click **"Train Models"** — training takes ~60 seconds.

> **Windows users:** use `python` instead of `python3`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/api/status` | Training status + active dataset |
| POST | `/api/upload` | Upload your own CSV dataset |
| POST | `/api/train` | Trigger model training |
| POST | `/api/reset` | Remove uploaded file, revert to default |
| GET | `/api/forecast` | Forecast data for all 3 models |
| GET | `/api/anomalies` | Detected anomalies + cause classification |
| GET | `/api/metrics` | RMSE, MAE, MAPE, SMAPE, R² per model |
| GET | `/api/dataset-info` | Dataset summary stats |

---

## Upload Your Own Dataset

The dashboard supports drag-and-drop CSV upload. Minimum required columns:

```
Date, Organic_Traffic
```

**Common column names are auto-detected** — `ds`, `y`, `traffic`, `sessions`, `page_loads`, `visits` are all recognised automatically.

**Missing columns are auto-generated** — if your CSV only has Date + traffic, all 14 remaining features (lag, rolling stats, CTR, position, etc.) are synthesised automatically so the model always gets its full 16-feature input.

Example format:
```
Date,Organic_Traffic
2024-01-01,3200
2024-01-02,4100
2024-01-03,3800
```

Minimum 60 rows required.

---

## Evaluation Metrics

All 5 metrics used across the 4 compared papers — enabling direct comparison:

| Metric | Formula | Paper coverage |
|---|---|---|
| RMSE | √(mean squared error) | Ahmed 2024, Samaan 2025 |
| MAE | mean absolute error | Ahmed 2024, Samaan 2025 |
| MAPE | mean \|error/actual\| × 100 | Sikka 2023, Ahmed 2024 |
| SMAPE | symmetric MAPE | Ahmed 2024 |
| R² | 1 − SS_res/SS_tot | Samaan 2025, Sikka 2023 |

---

## Comparison with Research Papers

| Model | Dataset | R² | MAPE | Anomaly Detection |
|---|---|---|---|---|
| Ahmed et al. 2024 (SARIMA+Informer) | Wikipedia | — | — (SMAPE 4.38%) | 
| Samaan et al. 2025 (CNN/LSTM/GRU) | BBC World Service | 0.994 | — |
| Sikka & Kumar 2023 (Voting Ensemble) | **Same dataset** | 0.9996* | 0.0024% | 
| Shelatkar et al. 2020 (DWT+ARIMA+LSTM) | Wikipedia | — | No metrics reported 
| **This project (SARIMA+GRU Hybrid)** | **Same as Paper 3** | **0.8841** | — | **5-class** |

> *Sikka & Kumar's R²=0.9996 is achieved without a strict temporal train-test split — not directly comparable to forward-looking evaluation.

---

## Key Novel Contributions

1. **5-class anomaly cause classification** — not present in any compared paper
2. **σ-based severity scoring** (Low/Medium/High) — not present in any compared paper
3. **Google Algorithm Update Flags** — Shelatkar et al. suggested this as future work; this project implements it
4. **16 multivariate SEO features** — all 4 papers are univariate
5. **OLS-optimised hybrid weights** — smarter than fixed-weight or equal-vote combination

---

## Requirements

```
flask
scikit-learn
pandas
numpy
scipy
openpyxl
```

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

**Tumulu Mihika**
3rd Year B.Tech CSE (2023–2027) — KIIT University, Bhubaneswar
