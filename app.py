"""
Flask web application for TGH Storm Closure Duration Prediction
Provides a web interface for users to input storm data and get closure duration predictions
"""

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask import request, jsonify
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__, static_folder='dist' if Path('dist').exists() else None, static_url_path='')
CORS(app)

MODEL_PATH = Path('models/closure_model.pkl')

def load_model():
    """Load the trained closure prediction model"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_models.py first.")

try:
    model = load_model()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

@app.route('/')
def index():
    """Serve React app"""
    if app.static_folder and Path(app.static_folder, 'index.html').exists():
        return send_from_directory(app.static_folder, 'index.html')
    else:
        return jsonify({
            'message': 'React app not built. Run "npm run build" or use "npm run dev" for development.',
            'dev_server': 'http://localhost:3000'
        }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first by running train_models.py'
        }), 500
    
    try:
        data = request.get_json()
        
        category = float(data.get('category', 0))
        max_wind = float(data.get('max_wind', 0))
        storm_surge = float(data.get('storm_surge', 0))
        track_distance = float(data.get('track_distance', 0))
        forward_speed = float(data.get('forward_speed', 0))
        month = float(data.get('month', 1))
        
        # Validate inputs
        if category < 0 or category > 5:
            return jsonify({'error': 'Category must be between 0 and 5'}), 400
        if max_wind < 0 or max_wind > 200:
            return jsonify({'error': 'Max wind must be between 0 and 200 mph'}), 400
        if storm_surge < 0 or storm_surge > 20:
            return jsonify({'error': 'Storm surge must be between 0 and 20 feet'}), 400
        if track_distance < 0 or track_distance > 200:
            return jsonify({'error': 'Track distance must be between 0 and 200 miles'}), 400
        if forward_speed < 0 or forward_speed > 30:
            return jsonify({'error': 'Forward speed must be between 0 and 30 mph'}), 400
        if month < 1 or month > 12:
            return jsonify({'error': 'Month must be between 1 and 12'}), 400
        
        features = pd.DataFrame([[
            category,
            max_wind,
            storm_surge,
            track_distance,
            forward_speed,
            month
        ]], columns=['category', 'max_wind', 'storm_surge', 
                     'track_distance', 'forward_speed', 'month'])
        
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)
        prediction = round(prediction, 1)
        
        return jsonify({
            'prediction': prediction,
            'prediction_hours': prediction,
            'prediction_days': round(prediction / 24, 1),
            'features': {
                'category': category,
                'max_wind': max_wind,
                'storm_surge': storm_surge,
                'track_distance': track_distance,
                'forward_speed': forward_speed,
                'month': month
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

