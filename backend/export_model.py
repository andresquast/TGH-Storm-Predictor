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
        
        print("="*60)
        print("EXPORT COMPLETE")
        print("="*60)
        print("\nFiles created in public/ directory:")
        print("  - model.json (prediction model)")
        print("  - seasonal_census.json (monthly census averages)")
        print("  - staffing_constants.json (staffing calculation constants)")
        print("\nThese files will be included in the build and can be loaded")
        print("by the frontend for client-side predictions.")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

