"""
Flask web application for TGH Storm Closure Duration Prediction
Provides a web interface for users to input storm data and get closure duration predictions
"""

from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
from flask import request, jsonify
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from backend.data_fetcher import StormDataFetcher
from backend.staffing_engine import calculate_bridge_event_staffing
from datetime import datetime

app = Flask(__name__, 
            static_folder='dist' if Path('dist').exists() else 'static',
            static_url_path='',
            template_folder='templates')
# Configure CORS to allow all origins (for development)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')
    return response

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
    """Serve React app or fallback template"""
    # Check if React build exists in dist folder
    if Path('dist').exists() and Path('dist', 'index.html').exists():
        return send_from_directory('dist', 'index.html')
    # Otherwise serve the template with static files
    elif Path('templates', 'index.html').exists():
        return render_template('index.html')
    else:
        return jsonify({
            'message': 'React app not built. Run "npm run build" or use "npm run dev" for development.',
            'dev_server': 'http://localhost:3000'
        }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle prediction requests"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    if model is None:
        response = jsonify({
            'error': 'Model not loaded. Please train the model first by running train_models.py'
        })
        response.status_code = 500
        return response
    
    try:
        # Check if request has JSON content
        if not request.is_json:
            print(f"Request is not JSON. Content-Type: {request.content_type}")
            print(f"Request data: {request.data}")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json(force=True)  # force=True to parse even if Content-Type is wrong
        print(f"Received data: {data}")
        print(f"Request method: {request.method}")
        print(f"Request content type: {request.content_type}")
        print(f"Request headers: {dict(request.headers)}")
        
        if not data:
            print("Warning: Empty data received")
            return jsonify({'error': 'No data received'}), 400
        
        # Safely convert to float, handling None and empty strings
        def safe_float(value, default=0):
            if value is None or value == '':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        max_wind = safe_float(data.get('max_wind'), 0)
        storm_surge = safe_float(data.get('storm_surge'), 0)
        track_distance = safe_float(data.get('track_distance'), 0)
        forward_speed = safe_float(data.get('forward_speed'), 0)
        month = safe_float(data.get('month'), 1)
        
        print(f"Parsed values: max_wind={max_wind}, storm_surge={storm_surge}, track_distance={track_distance}, forward_speed={forward_speed}, month={month}")
        
        # Validate inputs
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
            max_wind,
            storm_surge,
            track_distance,
            forward_speed,
            month
        ]], columns=['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month'])
        
        # Get prediction
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)
        
        # Calculate 95% confidence interval
        # We need to fit a statsmodels model to get prediction intervals
        try:
            fetcher = StormDataFetcher()
            df = fetcher.get_storms_dataframe(include_no_closure=False)
            feature_cols = ['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
            X_train = df[feature_cols]
            y_train = df['closure_hours']
            
            # Fit statsmodels model for prediction intervals
            X_train_const = sm.add_constant(X_train)
            model_sm = sm.OLS(y_train, X_train_const).fit()
            
            # Prepare features with constant for prediction
            X_pred_const = sm.add_constant(features)
            
            # Get prediction interval (95% confidence)
            prediction_result = model_sm.get_prediction(X_pred_const)
            pred_int = prediction_result.conf_int(alpha=0.05)  # 95% CI
            
            ci_lower = max(0, float(pred_int.iloc[0, 0]))
            ci_upper = max(0, float(pred_int.iloc[0, 1]))
            
        except Exception as e:
            # Fallback: use RMSE from model stats if available
            print(f"Warning: Could not calculate prediction interval: {e}")
            # Estimate using MAE (rough approximation)
            # Load data to get MAE
            try:
                fetcher = StormDataFetcher()
                df = fetcher.get_storms_dataframe(include_no_closure=False)
                feature_cols = ['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
                X_train = df[feature_cols]
                y_train = df['closure_hours']
                
                # Fit model to get residuals
                temp_model = LinearRegression()
                temp_model.fit(X_train, y_train)
                y_pred_train = temp_model.predict(X_train)
                residuals = y_train - y_pred_train
                rmse = np.sqrt(np.mean(residuals**2))
                
                # Approximate 95% CI using t-distribution (rough estimate)
                # For small samples, use t-value ~ 2 for 95% CI
                margin = 2 * rmse
                ci_lower = max(0, prediction - margin)
                ci_upper = prediction + margin
            except:
                # Final fallback: use a simple percentage
                margin = prediction * 0.2  # 20% margin
                ci_lower = max(0, prediction - margin)
                ci_upper = prediction + margin
        
        prediction = round(prediction, 1)
        ci_lower = round(ci_lower, 1)
        ci_upper = round(ci_upper, 1)
        
        response = jsonify({
            'prediction': prediction,
            'prediction_hours': prediction,
            'prediction_days': round(prediction / 24, 1),
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
                'lower_days': round(ci_lower / 24, 1),
                'upper_days': round(ci_upper / 24, 1)
            },
            'features': {
                'max_wind': max_wind,
                'storm_surge': storm_surge,
                'track_distance': track_distance,
                'forward_speed': forward_speed,
                'month': month
            }
        })
        # Add CORS headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e)
        print(f"="*60)
        print(f"PREDICTION ERROR: {error_msg}")
        print(f"="*60)
        print(f"Traceback:\n{error_trace}")
        print(f"="*60)
        
        # Always return JSON, even for errors
        try:
            response = jsonify({
                'error': f'Prediction error: {error_msg}',
                'details': error_trace if app.debug else None
            })
            response.status_code = 500
            # Add CORS headers even for errors
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
            return response
        except Exception as json_error:
            # If even JSON creation fails, return minimal error
            print(f"Failed to create JSON error response: {json_error}")
            return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({'status': 'ok', 'message': 'API is working'})

@app.route('/api/model-stats', methods=['GET'])
def get_model_stats():
    """Get regression statistics for the model"""
    try:
        # Load data
        fetcher = StormDataFetcher()
        df = fetcher.get_storms_dataframe(include_no_closure=False)
        
        if len(df) == 0:
            raise ValueError("No storm data available")
        
        # Prepare features and target
        features = ['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
        
        # Check if all required columns exist
        missing_cols = [col for col in features + ['closure_hours'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X = df[features]
        y = df['closure_hours']
        
        # Check for NaN values and handle them
        nan_cols_X = X.columns[X.isnull().any()].tolist()
        nan_cols_y = ['closure_hours'] if y.isnull().any() else []
        if nan_cols_X or nan_cols_y:
            # Try to fill NaN values with median for numeric columns
            for col in nan_cols_X:
                if col in X.columns:
                    X[col] = X[col].fillna(X[col].median())
            if 'closure_hours' in nan_cols_y:
                y = y.fillna(y.median())
            print(f"Warning: Filled NaN values in columns: {nan_cols_X + nan_cols_y}")
        
        # Check if we have enough data points
        if len(df) <= len(features):
            raise ValueError(f"Not enough data points ({len(df)}) for {len(features)} features")
        
        # Add constant for statsmodels (intercept)
        X_with_const = sm.add_constant(X)
        
        # Fit model using statsmodels (for p-values)
        model_sm = sm.OLS(y, X_with_const).fit()
        
        # Also fit with sklearn for consistency
        model_sk = LinearRegression()
        model_sk.fit(X, y)
        y_pred = model_sk.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = len(features)
        # Avoid division by zero
        if (n - p - 1) <= 0:
            adj_r2 = r2
        else:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Get coefficient statistics
        coefficients = []
        try:
            conf_int = model_sm.conf_int()
        except:
            conf_int = None
        
        all_feature_names = ['const'] + features  # statsmodels uses 'const' for intercept
        for i, feature_name in enumerate(all_feature_names):
            try:
                # Try to get by index first (most reliable)
                coef_value = float(model_sm.params.iloc[i])
                std_err = float(model_sm.bse.iloc[i])
                t_val = float(model_sm.tvalues.iloc[i])
                p_val = float(model_sm.pvalues.iloc[i])
                
                # Get confidence intervals
                if conf_int is not None:
                    ci_lower = float(conf_int.iloc[i, 0])
                    ci_upper = float(conf_int.iloc[i, 1])
                else:
                    ci_lower = 0.0
                    ci_upper = 0.0
                
                # Use 'Intercept' for const, otherwise use feature name
                display_name = 'Intercept' if feature_name == 'const' else feature_name
                
                coef_data = {
                    'feature': display_name,
                    'coefficient': coef_value,
                    'std_error': std_err,
                    't_value': t_val,
                    'p_value': p_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
            except Exception as e:
                # Fallback if anything fails
                print(f"Warning: Could not get coefficient for {feature_name}: {e}")
                display_name = 'Intercept' if feature_name == 'const' else feature_name
                coef_data = {
                    'feature': display_name,
                    'coefficient': 0.0,
                    'std_error': 0.0,
                    't_value': 0.0,
                    'p_value': 1.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0
                }
            
            # Add significance star
            p_val = coef_data['p_value']
            if p_val < 0.001:
                coef_data['significance'] = '***'
            elif p_val < 0.01:
                coef_data['significance'] = '**'
            elif p_val < 0.05:
                coef_data['significance'] = '*'
            elif p_val < 0.1:
                coef_data['significance'] = '.'
            else:
                coef_data['significance'] = ''
            
            coefficients.append(coef_data)
        
        # Model significance
        try:
            f_stat = float(model_sm.fvalue) if hasattr(model_sm, 'fvalue') else 0.0
            f_pvalue = float(model_sm.f_pvalue) if hasattr(model_sm, 'f_pvalue') else 1.0
        except:
            f_stat = 0.0
            f_pvalue = 1.0
        
        # Durbin-Watson
        try:
            dw = float(sm.stats.durbin_watson(model_sm.resid))
        except:
            dw = 0.0
        
        # Residual statistics
        residuals = y - y_pred
        residual_stats = {
            'mean': float(residuals.mean()),
            'std_dev': float(residuals.std()),
            'min': float(residuals.min()),
            'max': float(residuals.max())
        }
        
        # Normality test
        normality_test = None
        if len(residuals) <= 50:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normality_test = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': bool(shapiro_p > 0.05)
            }
        
        response = jsonify({
            'dataset_size': len(df),
            'num_features': len(features),
            'model_fit': {
                'r2': float(r2),
                'adj_r2': float(adj_r2),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'coefficients': coefficients,
            'model_significance': {
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'is_significant': bool(f_pvalue < 0.05)
            },
            'durbin_watson': dw,
            'residual_stats': residual_stats,
            'normality_test': normality_test
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e)
        print(f"="*60)
        print(f"ERROR GETTING MODEL STATS: {error_msg}")
        print(f"="*60)
        print(f"Traceback:\n{error_trace}")
        print(f"="*60)
        
        # Return error response with more details for debugging
        try:
            response = jsonify({
                'error': f'Error calculating model stats: {error_msg}',
                'details': error_trace if app.debug else error_msg,
                'traceback': error_trace if app.debug else None
            })
            response.status_code = 500
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as json_error:
            # If JSON creation fails, return minimal error
            print(f"Failed to create JSON error response: {json_error}")
            return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/data-stats', methods=['GET'])
def get_data_stats():
    """Get data statistics from the dataset"""
    try:
        fetcher = StormDataFetcher()
        df = fetcher.get_storms_dataframe(include_no_closure=False)
        
        if len(df) == 0:
            raise ValueError("No storm data available")
        
        # Check if required columns exist
        required_cols = ['closure_hours', 'max_wind', 'storm_surge', 'track_distance', 
                        'forward_speed', 'month', 'year', 'data_quality']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Closure duration statistics
        closure_stats = {
            'mean': float(df['closure_hours'].mean()),
            'median': float(df['closure_hours'].median()),
            'std_dev': float(df['closure_hours'].std()),
            'min': float(df['closure_hours'].min()),
            'max': float(df['closure_hours'].max()),
            'q25': float(df['closure_hours'].quantile(0.25)),
            'q75': float(df['closure_hours'].quantile(0.75)),
            'iqr': float(df['closure_hours'].quantile(0.75) - df['closure_hours'].quantile(0.25))
        }
        
        # Storm characteristics
        wind_stats = {
            'mean': float(df['max_wind'].mean()),
            'median': float(df['max_wind'].median()),
            'min': float(df['max_wind'].min()),
            'max': float(df['max_wind'].max()),
            'std_dev': float(df['max_wind'].std())
        }
        
        surge_stats = {
            'mean': float(df['storm_surge'].mean()),
            'median': float(df['storm_surge'].median()),
            'min': float(df['storm_surge'].min()),
            'max': float(df['storm_surge'].max()),
            'std_dev': float(df['storm_surge'].std())
        }
        
        track_stats = {
            'mean': float(df['track_distance'].mean()),
            'median': float(df['track_distance'].median()),
            'min': float(df['track_distance'].min()),
            'max': float(df['track_distance'].max()),
            'std_dev': float(df['track_distance'].std())
        }
        
        speed_stats = {
            'mean': float(df['forward_speed'].mean()),
            'median': float(df['forward_speed'].median()),
            'min': float(df['forward_speed'].min()),
            'max': float(df['forward_speed'].max()),
            'std_dev': float(df['forward_speed'].std())
        }
        
        # Temporal distribution
        month_dist = df.groupby('month').size().to_dict()
        
        # Closure duration distribution
        short_closures = len(df[df['closure_hours'] <= 12])
        medium_closures = len(df[(df['closure_hours'] > 12) & (df['closure_hours'] <= 24)])
        long_closures = len(df[df['closure_hours'] >= 25])
        
        # Correlations
        correlations = {
            'forward_speed': float(df['forward_speed'].corr(df['closure_hours'])),
            'storm_surge': float(df['storm_surge'].corr(df['closure_hours'])),
            'max_wind': float(df['max_wind'].corr(df['closure_hours'])),
            'track_distance': float(df['track_distance'].corr(df['closure_hours'])),
            'month': float(df['month'].corr(df['closure_hours']))
        }
        
        # Dataset overview
        year_range = {
            'min': int(df['year'].min()),
            'max': int(df['year'].max()),
            'span': int(df['year'].max() - df['year'].min())
        }
        
        data_quality_dist = df.groupby('data_quality').size().to_dict()
        
        response = jsonify({
            'dataset_overview': {
                'total_storms': len(df),
                'year_range': year_range,
                'data_quality': data_quality_dist
            },
            'closure_stats': closure_stats,
            'closure_distribution': {
                'short': short_closures,
                'medium': medium_closures,
                'long': long_closures
            },
            'wind_stats': wind_stats,
            'surge_stats': surge_stats,
            'track_stats': track_stats,
            'speed_stats': speed_stats,
            'temporal_distribution': {
                'by_month': {int(k): int(v) for k, v in month_dist.items()}
            },
            'correlations': correlations
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e)
        print(f"="*60)
        print(f"ERROR GETTING DATA STATS: {error_msg}")
        print(f"="*60)
        print(f"Traceback:\n{error_trace}")
        print(f"="*60)
        response = jsonify({
            'error': f'Error calculating data stats: {error_msg}',
            'details': error_trace if app.debug else error_msg
        })
        response.status_code = 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/api/staffing-projections', methods=['POST', 'OPTIONS'])
def get_staffing_projections():
    """Calculate staffing projections based on predicted closure duration"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Get prediction data
        predicted_duration_hours = float(data.get('predicted_duration_hours', 0))
        month = int(data.get('month', datetime.now().month))
        closure_start_str = data.get('closure_start')
        
        # Parse closure start datetime or use current time
        if closure_start_str:
            try:
                closure_start = datetime.fromisoformat(closure_start_str.replace('Z', '+00:00'))
            except:
                closure_start = datetime.now()
        else:
            closure_start = datetime.now()
        
        # Override month if provided (handle edge cases like Feb 31)
        if month:
            try:
                closure_start = closure_start.replace(month=month)
            except ValueError:
                # If day doesn't exist in target month, use last day of month
                from calendar import monthrange
                last_day = monthrange(closure_start.year, month)[1]
                closure_start = closure_start.replace(month=month, day=min(closure_start.day, last_day))
        
        # Optional: get current census if provided
        current_census = data.get('current_census')
        if current_census is not None:
            current_census = int(current_census)
        
        # Calculate staffing projections
        staffing_result = calculate_bridge_event_staffing(
            predicted_closure_start=closure_start,
            predicted_duration_hours=predicted_duration_hours,
            current_actual_census=current_census
        )
        
        response = jsonify(staffing_result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e)
        print(f"="*60)
        print(f"ERROR CALCULATING STAFFING PROJECTIONS: {error_msg}")
        print(f"="*60)
        print(f"Traceback:\n{error_trace}")
        print(f"="*60)
        
        response = jsonify({
            'error': f'Error calculating staffing projections: {error_msg}',
            'details': error_trace if app.debug else error_msg
        })
        response.status_code = 500
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

