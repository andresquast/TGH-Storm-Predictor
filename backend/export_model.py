"""
Export model parameters to JSON for static hosting
This allows the model to run entirely in the browser without a backend
"""

import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from data_fetcher import StormDataFetcher

def export_model():
    """Export closure prediction model to JSON"""
    
    # Load the trained model
    model_path = Path('../models/closure_model.pkl')
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_models.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get feature names (must match training order)
    features = ['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
    
    # Extract coefficients and intercept
    coefficients = model.coef_.tolist()
    intercept = float(model.intercept_)
    
    # Calculate confidence intervals using statsmodels
    # We need to refit with statsmodels to get prediction intervals
    fetcher = StormDataFetcher()
    df = fetcher.get_storms_dataframe(include_no_closure=False)
    X_train = df[features]
    y_train = df['closure_hours']
    
    # Fit statsmodels model for prediction intervals
    X_train_const = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_train_const).fit()
    
    # Get standard errors and other stats
    std_errors = model_sm.bse[1:].tolist()  # Skip intercept std error for now
    r2 = float(model_sm.rsquared)
    adj_r2 = float(model_sm.rsquared_adj)
    mse = float(model_sm.mse_resid)
    
    # Export model data
    model_data = {
        'type': 'linear_regression',
        'features': features,
        'coefficients': coefficients,
        'intercept': intercept,
        'r2': r2,
        'adj_r2': adj_r2,
        'mse': mse,
        'std_errors': std_errors,
        'n_samples': len(df)
    }
    
    # Save to public directory (will be included in build)
    output_path = Path('../public/model.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"✓ Model exported to {output_path}")
    print(f"  Features: {features}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Samples: {len(df)}")
    
    return model_data


def export_seasonal_census():
    """Export seasonal census data to JSON"""
    
    # Load seasonal census
    census_path = Path('../models/seasonal_census.pkl')
    if not census_path.exists():
        # Generate it if it doesn't exist
        from train_models import get_seasonal_census_avg
        seasonal_census = get_seasonal_census_avg()
    else:
        with open(census_path, 'rb') as f:
            seasonal_census = pickle.load(f)
    
    # Export to JSON
    output_path = Path('../public/seasonal_census.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(seasonal_census, f, indent=2)
    
    print(f"✓ Seasonal census exported to {output_path}")
    
    return seasonal_census


def export_staffing_constants():
    """Export staffing calculation constants to JSON"""
    
    from staffing_engine import TGHConstants, TemporalAdjustment
    
    constants = {
        'average_daily_census': TGHConstants.AVERAGE_DAILY_CENSUS,
        'licensed_beds': TGHConstants.LICENSED_BEDS,
        'annual_er_visits': TGHConstants.ANNUAL_ER_VISITS,
        'daily_er_visits_avg': TGHConstants.DAILY_ER_VISITS_AVG,
        'inpatient_crisis_ratio': TGHConstants.INPATIENT_CRISIS_RATIO,
        'er_md_patient_ratio': TGHConstants.ER_MD_PATIENT_RATIO,
        'specialty_skeleton_crew': TGHConstants.SPECIALTY_SKELETON_CREW,
        'shift_duration_hours': TGHConstants.SHIFT_DURATION_HOURS
    }
    
    temporal = TemporalAdjustment()
    
    temporal_data = {
        'monthly_census_multipliers': {
            str(k): v for k, v in temporal.monthly_census_multipliers.items()
        },
        'diurnal_er_multipliers': temporal.diurnal_er_multipliers,
        'storm_surge_months': [6, 7, 8, 9],  # Jun-Sep
        'storm_surge_multiplier': 1.2
    }
    
    staffing_data = {
        'constants': constants,
        'temporal': temporal_data
    }
    
    output_path = Path('../public/staffing_constants.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(staffing_data, f, indent=2)
    
    print(f"✓ Staffing constants exported to {output_path}")
    
    return staffing_data


def export_model_stats():
    """Export model statistics to JSON (same format as /api/model-stats endpoint)"""
    
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
    
    # Handle NaN values
    nan_cols_X = X.columns[X.isnull().any()].tolist()
    nan_cols_y = ['closure_hours'] if y.isnull().any() else []
    if nan_cols_X or nan_cols_y:
        for col in nan_cols_X:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        if 'closure_hours' in nan_cols_y:
            y = y.fillna(y.median())
    
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
    
    all_feature_names = ['const'] + features
    for i, feature_name in enumerate(all_feature_names):
        try:
            coef_value = float(model_sm.params.iloc[i])
            std_err = float(model_sm.bse.iloc[i])
            t_val = float(model_sm.tvalues.iloc[i])
            p_val = float(model_sm.pvalues.iloc[i])
            
            if conf_int is not None:
                ci_lower = float(conf_int.iloc[i, 0])
                ci_upper = float(conf_int.iloc[i, 1])
            else:
                ci_lower = 0.0
                ci_upper = 0.0
            
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
    
    model_stats = {
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
    }
    
    # Save to public directory
    output_path = Path('../public/model_stats.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(model_stats, f, indent=2)
    
    print(f"✓ Model stats exported to {output_path}")
    print(f"  R²: {r2:.4f}, MAE: {mae:.2f} hours")
    
    return model_stats


def export_data_stats():
    """Export data statistics to JSON (same format as /api/data-stats endpoint)"""
    
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
    
    data_stats = {
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
    }
    
    # Save to public directory
    output_path = Path('../public/data_stats.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data_stats, f, indent=2)
    
    print(f"✓ Data stats exported to {output_path}")
    print(f"  Total storms: {len(df)}, Year range: {year_range['min']}-{year_range['max']}")
    
    return data_stats


if __name__ == '__main__':
    print("="*60)
    print("EXPORTING MODEL FOR STATIC HOSTING")
    print("="*60)
    print()
    
    try:
        model_data = export_model()
        print()
        
        census_data = export_seasonal_census()
        print()
        
        staffing_data = export_staffing_constants()
        print()
        
        model_stats_data = export_model_stats()
        print()
        
        data_stats_data = export_data_stats()
        print()
        
        print("="*60)
        print("EXPORT COMPLETE")
        print("="*60)
        print("\nFiles created in public/ directory:")
        print("  - model.json (prediction model)")
        print("  - seasonal_census.json (monthly census averages)")
        print("  - staffing_constants.json (staffing calculation constants)")
        print("  - model_stats.json (model statistics)")
        print("  - data_stats.json (dataset statistics)")
        print("\nThese files will be included in the build and can be loaded")
        print("by the frontend for client-side predictions and stats viewing.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

