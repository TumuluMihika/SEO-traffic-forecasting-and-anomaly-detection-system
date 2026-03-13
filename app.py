"""
Flask Application — SEO Traffic Forecasting & Anomaly Detection
---------------------------------------------------------------
API Endpoints:
  GET  /                    → Dashboard HTML
  GET  /api/status          → Training status + upload info
  POST /api/upload          → Upload custom CSV dataset
  POST /api/train           → Trigger model training
  POST /api/reset           → Remove uploaded file, go back to default
  GET  /api/forecast        → Forecast data (all 3 models)
  GET  /api/anomalies       → Detected + classified anomalies
  GET  /api/metrics         → RMSE, MAE, MAPE, SMAPE, R² per model
  GET  /api/dataset-info    → Dataset summary
"""

import os, sys, json, threading
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

CACHE_PATH  = os.path.join(BASE, 'data', 'results_cache.json')
DEFAULT_CSV = os.path.join(BASE, 'data', 'seo_traffic.csv')
UPLOAD_CSV  = os.path.join(BASE, 'data', 'uploaded_dataset.csv')
UPLOAD_INFO = os.path.join(BASE, 'data', 'upload_info.json')

REQUIRED_COLS = ['Date', 'Organic_Traffic']
OPTIONAL_COLS = [
    'Unique_Visits', 'First_Time_Visits', 'Returning_Visits',
    'Impressions', 'CTR', 'Avg_Position', 'Bounce_Rate',
    'Day_of_Week', 'Is_Weekend', 'Month',
    'Traffic_Lag7', 'Traffic_Lag14',
    'Rolling_Mean_7', 'Rolling_Std_7', 'Algorithm_Flag'
]

# Global training state
_training_status = {'running': False, 'log': [], 'done': False, 'error': None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return None


def _get_active_csv():
    if os.path.exists(UPLOAD_CSV):
        return UPLOAD_CSV
    return DEFAULT_CSV


def _auto_enrich(df):
    """Fill any missing optional columns so model always gets 16 features."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    traffic = df['Organic_Traffic'].astype(float)

    if 'Day_of_Week'       not in df: df['Day_of_Week']       = df['Date'].dt.dayofweek
    if 'Is_Weekend'        not in df: df['Is_Weekend']         = (df['Day_of_Week'] >= 5).astype(int)
    if 'Month'             not in df: df['Month']              = df['Date'].dt.month
    if 'Algorithm_Flag'    not in df: df['Algorithm_Flag']     = 0
    if 'Traffic_Lag7'      not in df: df['Traffic_Lag7']       = traffic.shift(7)
    if 'Traffic_Lag14'     not in df: df['Traffic_Lag14']      = traffic.shift(14)
    if 'Rolling_Mean_7'    not in df: df['Rolling_Mean_7']     = traffic.shift(1).rolling(7).mean()
    if 'Rolling_Std_7'     not in df: df['Rolling_Std_7']      = traffic.shift(1).rolling(7).std()
    if 'Avg_Position'      not in df: df['Avg_Position']       = 12.0
    if 'Impressions'       not in df: df['Impressions']        = (traffic * 2.5).clip(1).astype(int)
    if 'CTR'               not in df: df['CTR']                = (traffic / df['Impressions'].replace(0, 1)).clip(0.001, 0.35)
    if 'Bounce_Rate'       not in df: df['Bounce_Rate']        = 0.35
    if 'Unique_Visits'     not in df: df['Unique_Visits']      = (traffic * 0.72).astype(int)
    if 'First_Time_Visits' not in df: df['First_Time_Visits']  = (traffic * 0.60).astype(int)
    if 'Returning_Visits'  not in df: df['Returning_Visits']   = (traffic * 0.12).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def _validate_and_save(filepath, original_filename):
    """
    Reads, validates, enriches, and overwrites the CSV at filepath.
    Returns (ok, message, info_dict).
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return False, f"Cannot read CSV: {e}", {}

    # ── Auto-detect column name variants ──────────────────────────────────
    col_map = {c.lower().strip().replace(' ', '_').replace('.', '_'): c for c in df.columns}
    rename  = {}

    date_variants    = ['date', 'ds', 'day', 'timestamp', 'time', 'datetime']
    traffic_variants = ['organic_traffic', 'page_loads', 'pageloads', 'traffic',
                        'visits', 'sessions', 'y', 'pageviews', 'page_views',
                        'unique_visits', 'clicks']

    found_date    = next((col_map[v] for v in date_variants    if v in col_map), None)
    found_traffic = next((col_map[v] for v in traffic_variants if v in col_map), None)

    if found_date    and found_date    != 'Date':            rename[found_date]    = 'Date'
    if found_traffic and found_traffic != 'Organic_Traffic': rename[found_traffic] = 'Organic_Traffic'
    if rename:
        df = df.rename(columns=rename)

    # ── Check required columns ────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sample = ', '.join(df.columns.tolist()[:8])
        return False, (
            f"Missing required columns: {missing}. "
            f"Your file has: {sample}. "
            f"Please rename your date column to 'Date' and traffic column to 'Organic_Traffic'."
        ), {}

    # ── Parse dates ───────────────────────────────────────────────────────
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        return False, "Cannot parse 'Date' column. Use YYYY-MM-DD format (e.g. 2024-01-15).", {}

    # ── Parse traffic ─────────────────────────────────────────────────────
    df['Organic_Traffic'] = pd.to_numeric(
        df['Organic_Traffic'].astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    )
    df = df.dropna(subset=['Date', 'Organic_Traffic'])

    if len(df) < 60:
        return False, f"Too few rows: {len(df)} valid rows found. Need at least 60 daily records.", {}

    dupes = int(df['Date'].duplicated().sum())
    if dupes > 0:
        df = df.drop_duplicates(subset=['Date'], keep='last')

    # ── Enrich + save ─────────────────────────────────────────────────────
    df = _auto_enrich(df)
    df.to_csv(filepath, index=False)

    info = {
        'original_filename': original_filename,
        'rows':        len(df),
        'date_start':  str(df['Date'].min())[:10],
        'date_end':    str(df['Date'].max())[:10],
        'columns_found':   list(rename.keys()) or list(df.columns[:5]),
        'columns_renamed': rename,
        'duplicates_removed': dupes,
        'auto_filled_cols': [c for c in OPTIONAL_COLS if c not in df.columns]
    }
    return True, f"File validated successfully. {len(df)} rows ready for training.", info


# ── Training ──────────────────────────────────────────────────────────────────

def _run_training(csv_path=None):
    global _training_status
    _training_status['running'] = True
    _training_status['log']     = []
    _training_status['done']    = False
    _training_status['error']   = None

    def cb(msg):
        _training_status['log'].append(msg)

    try:
        from src.trainer import train_all
        active_csv = csv_path or _get_active_csv()
        cb(f"Dataset: {os.path.basename(active_csv)}")
        train_all(csv_path=active_csv, progress_cb=cb)
        _training_status['done'] = True
    except Exception as e:
        import traceback
        _training_status['error'] = str(e)
        cb(f"ERROR: {e}")
        cb(traceback.format_exc())
    finally:
        _training_status['running'] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    cache       = _load_cache()
    upload_info = {}
    if os.path.exists(UPLOAD_INFO):
        with open(UPLOAD_INFO) as f:
            upload_info = json.load(f)
    return jsonify({
        'training':    _training_status,
        'has_cache':   cache is not None,
        'upload_info': upload_info,
        'active_file': os.path.basename(_get_active_csv())
    })


@app.route('/api/upload', methods=['POST'])
def upload():
    if _training_status['running']:
        return jsonify({'error': 'Training is already running. Please wait.'}), 409

    if 'file' not in request.files:
        return jsonify({'error': 'No file in request. Use key "file" in form-data.'}), 400

    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files are accepted.'}), 400

    # Save file
    f.save(UPLOAD_CSV)

    # Validate + enrich
    ok, message, info = _validate_and_save(UPLOAD_CSV, f.filename)

    if not ok:
        if os.path.exists(UPLOAD_CSV):
            os.remove(UPLOAD_CSV)
        return jsonify({'error': message}), 422

    # Save upload record
    with open(UPLOAD_INFO, 'w') as fp:
        json.dump({'filename': f.filename, 'info': info}, fp)

    # Clear stale cache
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    # Auto-start training
    t = threading.Thread(target=_run_training, args=(UPLOAD_CSV,), daemon=True)
    t.start()

    return jsonify({
        'success': True,
        'message': message,
        'info':    info,
        'training_started': True
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    if _training_status['running']:
        return jsonify({'error': 'Cannot reset while training is running.'}), 409
    for path in [UPLOAD_CSV, UPLOAD_INFO, CACHE_PATH]:
        if os.path.exists(path):
            os.remove(path)
    # Restart training on default dataset
    t = threading.Thread(target=_run_training, daemon=True)
    t.start()
    return jsonify({'message': 'Reset to default dataset. Training started.'})


@app.route('/api/train', methods=['POST'])
def train():
    if _training_status['running']:
        return jsonify({'message': 'Training already in progress'}), 409
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    t = threading.Thread(target=_run_training, daemon=True)
    t.start()
    return jsonify({'message': 'Training started'})


@app.route('/api/forecast')
def forecast():
    cache = _load_cache()
    if not cache:
        return jsonify({'error': 'No trained model yet.'}), 404
    return jsonify({
        'dates':   cache['dates'],
        'actuals': cache['actuals'],
        'sarima':  cache['sarima'],
        'gru':     cache['gru'],
        'hybrid':  cache['hybrid'],
        'weights': cache.get('hybrid_weights', {})
    })


@app.route('/api/anomalies')
def anomalies():
    cache = _load_cache()
    if not cache:
        return jsonify({'error': 'No trained model yet.'}), 404
    return jsonify({
        'anomalies': cache['anomalies'],
        'summary':   cache.get('anomaly_summary', {})
    })


@app.route('/api/metrics')
def metrics():
    cache = _load_cache()
    if not cache:
        return jsonify({'error': 'No trained model yet.'}), 404
    return jsonify({'metrics': cache['metrics']})


@app.route('/api/dataset-info')
def dataset_info():
    cache = _load_cache()
    if not cache:
        return jsonify({'error': 'No trained model yet.'}), 404
    return jsonify(cache.get('dataset_info', {}))


if __name__ == '__main__':
    if not os.path.exists(CACHE_PATH):
        print("No cache — training on default dataset...")
        t = threading.Thread(target=_run_training, daemon=True)
        t.start()
    app.run(debug=True, port=5000, use_reloader=False)
