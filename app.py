import pickle
import bz2
from flask import Flask, request, jsonify, render_template, make_response
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app_logger import log
from mongodb import mongodbconnection
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os
import requests


load_dotenv()
FIRMS_API_KEY = os.getenv("FIRMS_API_KEY")

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Fixed feature schema used across scaler and predictions
FEATURE_NAMES = ['Temperature', 'Ws', 'FFMC', 'DMC', 'ISI']
ALTERNATIVE_FEATURE_NAMES = {
    'Temperature': ['Temperature', 'Temp', 'temperature', 'temp'],
    'Ws': ['Ws', 'Wind_speed', 'WindSpeed', 'wind_speed', 'ws'],
    'FFMC': ['FFMC', 'ffmc'],
    'DMC': ['DMC', 'dmc'],
    'ISI': ['ISI', 'isi'],
}

# Simple in-memory history for dashboard demo
PREDICTION_HISTORY = []

# Import Classification and Regression model file with fallback to training
def _load_models_with_fallback(scaler_obj, feature_df: pd.DataFrame):
    try:
        pickle_in = bz2.BZ2File('model/classification.pkl', 'rb')
        R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
        model_C_loaded = pickle.load(pickle_in)
        model_R_loaded = pickle.load(R_pickle_in)
        log.info('Models loaded from pickle files')
        return model_C_loaded, model_R_loaded
    except Exception as e:
        log.error('Failed to load pickled models, training fallback models: %s', e)
        # Train simple models from local data
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            # Features and targets
            # Restrict to known features for consistency with UI
            X_local_all = feature_df.drop(['FWI', 'Classes'], axis=1)
            # Select only features we actually use
            available = [c for c in FEATURE_NAMES if c in X_local_all.columns]
            X_local = X_local_all[available]
            # Impute missing with medians computed earlier (or recompute if None)
            if feature_medians is not None:
                X_local = X_local.fillna(feature_medians)
            else:
                X_local = X_local.fillna(X_local.median(numeric_only=True))
            y_class = feature_df['Classes']
            y_reg = feature_df['FWI']
            X_scaled_local = scaler_obj.transform(X_local)
            clf = LogisticRegression(max_iter=200)
            reg = LinearRegression()
            clf.fit(X_scaled_local, y_class)
            reg.fit(X_scaled_local, y_reg)
            log.info('Fallback models trained successfully')
            return clf, reg
        except Exception as e2:
            log.error('Fallback training failed: %s', e2)
            raise


# Data source and Standardization setup (MongoDB -> fallback to local CSV)
scaler = StandardScaler()
feature_columns = FEATURE_NAMES.copy()
feature_medians = None
try:
    # Try retrieving data from MongoDB for scaler fitting
    dbcon = mongodbconnection(username='mongodb', password='12345')
    list_cursor = dbcon.getdata(dbName='FireDataML', collectionName='ml_task')
    log.info('Connected to Mongodb and data retrieved')

    df = pd.DataFrame(list_cursor)
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)
    log.info('DataFrame created from MongoDB')
    X_all = df.drop(['FWI', 'Classes'], axis=1)
    X = X_all[[c for c in FEATURE_NAMES if c in X_all.columns]]
    feature_columns = list(X.columns)
    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    _ = scaler.fit_transform(X)
    log.info('Standardization done using MongoDB data')
except Exception as e:
    # Fallback to local CSV if MongoDB is not reachable
    log.error('MongoDB connection failed, falling back to local CSV: %s', e)
    csv_candidates = [
        'dataset/Fire_dataset_cleaned.csv',
        'dataset/Algerian_forest_fires_dataset_CLEANED.csv',
        'dataset/Algerian_forest_fires_dataset_UPDATE.csv',
    ]
    last_error = None
    df = None
    for csv_path in csv_candidates:
        try:
            df_candidate = pd.read_csv(csv_path)
            if all(col in df_candidate.columns for col in ['FWI', 'Classes']):
                df = df_candidate
                log.info(f'Loaded local CSV for scaler: {csv_path}')
                break
        except Exception as csv_e:
            last_error = csv_e
            continue
    if df is None:
        raise RuntimeError('Unable to initialize scaler: MongoDB unreachable and no suitable local CSV found') from last_error
    X_all = df.drop(['FWI', 'Classes'], axis=1)
    X = X_all[[c for c in FEATURE_NAMES if c in X_all.columns]]
    feature_columns = list(X.columns)
    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    _ = scaler.fit_transform(X)
    log.info('Standardization done using local CSV data')

# Load models or train fallback ones if needed
model_C, model_R = _load_models_with_fallback(scaler, df)


# Landing page
@app.route('/')
def home():
    log.info('Landing page loaded successfully')
    return render_template('home.html')

# Main app page
@app.route('/app')
def app_page():
    log.info('Index page loaded successfully')
    return render_template('index.html')

# Dashboard page (static UI demo)
@app.route('/dashboard')
def dashboard():
    log.info('Dashboard page loaded successfully')
    return render_template('dashboard.html', risk_status=None, fwi_value=None, history=PREDICTION_HISTORY)


@app.route('/dashboard/predict', methods=['POST'])
def dashboard_predict():
    """Handle file upload (CSV or image placeholder) and produce predictions."""
    risk_text = None
    fwi_pred = None
    uploaded_name = None
    try:
        file = request.files.get('data_file')
        row = None
        if not file or not file.filename:
            # Require a file selection; do not auto-predict with medians
            uploaded_name = 'No file selected'
            raise ValueError('Please select a CSV or Excel file before predicting.')

        if file and file.filename:
            uploaded_name = file.filename
        if file and file.filename:
            fname = file.filename.lower()
        else:
            fname = ''

        if fname.endswith('.csv'):
            df_upload = pd.read_csv(file)
        elif fname.endswith('.xlsx') or fname.endswith('.xls'):
            df_upload = pd.read_excel(file)
        elif fname:
            raise ValueError('Unsupported file type. Please upload CSV or Excel (.xlsx/.xls).')
        else:
            df_upload = None

        if df_upload is not None:
            # Normalize columns by matching aliases
            normalized = {}
            for feat in FEATURE_NAMES:
                value = None
                for alias in ALTERNATIVE_FEATURE_NAMES.get(feat, [feat]):
                    if alias in df_upload.columns:
                        value = df_upload.iloc[0][alias]
                        break
                normalized[feat] = float(value) if pd.notna(value) else float(feature_medians.get(feat, 0))
            row = [normalized[k] for k in feature_columns]
        else:
            # Unsupported file type already handled above
            uploaded_name = uploaded_name or 'No file provided'
            raise ValueError('Unsupported or empty file. Please upload CSV or Excel.')

        X_row = scaler.transform([row])
        cls = int(model_C.predict(X_row)[0])
        fwi_pred = float(model_R.predict(X_row)[0])
        risk_text = 'High Risk' if cls == 1 or fwi_pred > 15 else 'Low Risk'

        # Append to history
        PREDICTION_HISTORY.append({
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'region': 'N/A',
            'risk': risk_text,
            'fwi': round(fwi_pred, 2),
        })
        log.info('Dashboard prediction done: %s, FWI=%.3f', risk_text, fwi_pred)
        return jsonify({
            'success': True,
            'risk_status': risk_text,
            'fwi_value': fwi_pred,
            'uploaded_name': uploaded_name
        })
    except Exception as e:
        log.error('Dashboard prediction failed: %s', e)
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/fires')
def api_fires():
    """Fetch live FIRMS VIIRS hotspots for a given Indian state and lookback window.
    Query params: state=Tamil%20Nadu&hours=24
    """
    try:
        state = request.args.get('state', 'Maharashtra')
        hours = int(request.args.get('hours', '24'))
        key = os.getenv('FIRMS_API_KEY')
        if not key:
            return jsonify({ 'success': False, 'error': 'FIRMS_API_KEY not configured' }), 500

        # FIRMS area API expects region keyword; use India and filter by state bbox locally
        # For higher precision, switch to shapefile filter if available.
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/India/{hours}/1?key={key}"
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        text = r.text
        # Parse CSV robustly (trim headers, case-insensitive)
        import csv
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines or lines[0].lower().startswith('<html'):
            return jsonify({ 'success': False, 'error': 'Unexpected response from FIRMS' }), 502
        reader = csv.reader(lines)
        raw_header = next(reader)
        header = [h.strip() for h in raw_header]
        header_lc = [h.lower() for h in header]
        def find_col(candidates):
            for name in candidates:
                if name in header_lc:
                    return header_lc.index(name)
            return None
        idx_lat = find_col(['latitude','lat'])
        idx_lon = find_col(['longitude','lon','long'])
        idx_conf = find_col(['confidence','confidence_text'])
        idx_acq = find_col(['acq_date','acq_datetime','acq_time'])
        if idx_lat is None or idx_lon is None:
            return jsonify({ 'success': False, 'error': 'Missing latitude/longitude columns in FIRMS response' }), 502

        # Rough state bounding boxes (deg). Replace with GeoJSON polygons for accuracy later.
        state_bboxes = {
            'Maharashtra': (15.6, 72.5, 22.3, 80.9),
            'Madhya Pradesh': (21.0, 74.0, 26.9, 82.0),
            'Chhattisgarh': (17.8, 80.2, 24.5, 84.5),
            'Odisha': (18.5, 81.3, 22.6, 87.5),
            'Gujarat': (20.1, 68.0, 24.7, 74.6),
            'Rajasthan': (23.3, 69.3, 30.2, 78.2),
            'Karnataka': (11.6, 74.0, 18.5, 78.6),
            'Telangana': (16.0, 77.0, 19.6, 81.8),
            'Andhra Pradesh': (13.6, 77.0, 19.2, 84.5),
            'Kerala': (8.1, 74.6, 12.9, 77.5),
            'Tamil Nadu': (8.1, 76.0, 13.5, 80.5)
        }
        bbox = state_bboxes.get(state, state_bboxes['Maharashtra'])
        minlat, minlon, maxlat, maxlon = bbox

        features = []
        counts = { 'Low': 0, 'Moderate': 0, 'High': 0 }
        for cols in reader:
            try:
                lat = float(cols[idx_lat])
                lon = float(cols[idx_lon])
            except Exception:
                continue
            if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
                continue
            confidence = None
            if idx_conf is not None and idx_conf < len(cols):
                conf_val = cols[idx_conf]
                try:
                    confidence = float(conf_val)
                except Exception:
                    confidence = conf_val  # may be text like 'high'
            acq_date = cols[idx_acq] if (idx_acq is not None and idx_acq < len(cols)) else ''

            # Risk classification: numeric confidence -> thresholds; text -> mapping
            risk = 'Moderate'
            if isinstance(confidence, (int, float)):
                if confidence >= 80:
                    risk = 'High'
                elif confidence >= 50:
                    risk = 'Moderate'
                else:
                    risk = 'Low'
            elif isinstance(confidence, str):
                cv = confidence.strip().lower()
                if cv in ('high', 'h'):
                    risk = 'High'
                elif cv in ('nominal', 'n', 'normal', 'moderate', 'm'):
                    risk = 'Moderate'
                elif cv in ('low', 'l'):
                    risk = 'Low'

            counts[risk] = counts.get(risk, 0) + 1
            features.append({
                'type': 'Feature',
                'geometry': { 'type': 'Point', 'coordinates': [lon, lat] },
                'properties': { 'confidence': confidence, 'acq_date': acq_date, 'risk': risk }
            })

        any_high = counts.get('High', 0) > 0
        return jsonify({ 'success': True, 'count': len(features), 'features': features, 'summary': { 'counts': counts, 'any_high': any_high, 'state': state, 'hours': hours } })
    except Exception as e:
        log.error('FIRMS fetch failed: %s', e)
        return jsonify({ 'success': False, 'error': str(e) }), 500
@app.route('/profile')
def profile_page():
    return render_template('profile.html')


# Report download for dashboard predictions
@app.route('/dashboard/report')
def dashboard_report():
    try:
        fmt = (request.args.get('format') or 'csv').lower()
        if not PREDICTION_HISTORY:
            # Return empty report with headers
            empty = pd.DataFrame(columns=['date', 'region', 'risk', 'fwi'])
            csv = empty.to_csv(index=False)
            resp = make_response(csv)
            resp.headers['Content-Type'] = 'text/csv'
            resp.headers['Content-Disposition'] = 'attachment; filename="prediction_report.csv"'
            return resp

        df_hist = pd.DataFrame(PREDICTION_HISTORY)
        if fmt == 'json':
            return jsonify({ 'success': True, 'rows': PREDICTION_HISTORY })
        if fmt == 'xlsx' or fmt == 'excel':
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df_hist.to_excel(writer, index=False, sheet_name='Predictions')
            buf.seek(0)
            resp = make_response(buf.read())
            resp.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            resp.headers['Content-Disposition'] = 'attachment; filename="prediction_report.xlsx"'
            return resp
        # default CSV
        csv = df_hist.to_csv(index=False)
        resp = make_response(csv)
        resp.headers['Content-Type'] = 'text/csv'
        resp.headers['Content-Disposition'] = 'attachment; filename="prediction_report.csv"'
        return resp
    except Exception as e:
        log.error('Report generation failed: %s', e)
        return jsonify({ 'success': False, 'error': str(e) }), 500

# Route for API Testing
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        log.info('Input from Api testing: %s', data)
        # Build feature vector in fixed order
        new_row = [float(data.get(k, 0)) for k in feature_columns]
        new_data = [new_row]
        final_data = scaler.transform(new_data)
        output = int(model_C.predict(final_data)[0])
        if output == 1:
            text = 'The Forest in Danger'
        else:
            text = 'Forest is Safe'
        return jsonify(text, output)
    except Exception as e:
        output = 'Check the input again!'
        log.error('error in input from Postman: %s', e)
        return jsonify(output)


# Route for Classification Model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Build in fixed order from form fields
        values_map = {
            'Temperature': request.form.get('Temperature'),
            'Ws': request.form.get('Ws'),
            'FFMC': request.form.get('FFMC'),
            'DMC': request.form.get('DMC'),
            'ISI': request.form.get('ISI'),
        }
        row = [float(values_map.get(k, 0)) for k in feature_columns]
        final_features = [np.array(row)]
        final_features = scaler.transform(final_features)
        output = model_C.predict(final_features)[0]
        log.info('Prediction done for Classification model')
        if output == 0:
            text = 'Forest is Safe!'
        else:
            text = 'Forest is in Danger!'
        return render_template('index.html', prediction_text1="{} --- Chance of Fire is {}".format(text, output))
    except Exception as e:
        log.error('Input error, check input: %s', e)
        return render_template('index.html', prediction_text1="Check the Input again!!!")


# Route for Regression Model
@app.route('/predictR', methods=['POST'])
def predictR():
    try:
        # Regression: use same five features; map form fields to schema
        values_map = {
            'Temperature': request.form.get('Temperature') or request.form.get('Temperature1'),
            'Ws': request.form.get('Ws') or request.form.get('Wind_speed'),
            'FFMC': request.form.get('FFMC') or request.form.get('FFMC1'),
            'DMC': request.form.get('DMC') or request.form.get('DMC1'),
            'ISI': request.form.get('ISI') or request.form.get('ISI1'),
        }
        row = [float(values_map.get(k, 0)) for k in feature_columns]
        X_row = scaler.transform([row])
        output = model_R.predict(X_row)[0]
        log.info('Prediction done for Regression model')
        if output > 15:
            return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Warning!!! High hazard rating".format(output))
        else:
            return render_template('index.html', prediction_text2="Fuel Moisture Code index is {:.4f} ---- Safe.. Low hazard rating".format(output))
    except Exception as e:
        log.error('Input error, check input: %s', e)
        return render_template('index.html', prediction_text2="Check the Input again!!!")


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
