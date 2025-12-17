"""
Model Training for TGH Storm Prediction
Uses historical storm data to train bridge closure prediction models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path

# Import our data fetcher
from data_fetcher import StormDataFetcher


def train_closure_model():
    """Train bridge closure duration prediction model"""
    
    # Load data
    fetcher = StormDataFetcher()
    df = fetcher.get_high_quality_data()
    
    print("\n" + "="*60)
    print("TRAINING BRIDGE CLOSURE PREDICTION MODEL")
    print("="*60)
    
    # Features for prediction
    features = ['category', 'max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
    X = df[features]
    y = df['closure_hours']
    
    # Train model (LinearRegression works better with small datasets)
    model = LinearRegression()
    model.fit(X, y)
    
    # Display feature importance (coefficients)
    print("\nModel Coefficients (feature importance):")
    coeffs = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    for _, row in coeffs.iterrows():
        print(f"  {row['feature']:20s}: {row['coefficient']:8.2f}")
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    with open('models/closure_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to models/closure_model.pkl")
    print(f"Training data: {len(df)} storms from {df['year'].min()}-{df['year'].max()}")
    
    return model, df


def calculate_staffing_requirements(census, closure_hours):
    """
    Calculate required staff based on census and closure duration
    
    Args:
        census: Expected hospital census (number of patients)
        closure_hours: Expected bridge closure duration in hours
    
    Returns:
        dict: Required staff by role
    """
    
    # Nursing ratios by unit
    general_floor_ratio = 6  # 1 nurse : 6 patients
    icu_ratio = 2  # 1 nurse : 2 patients
    
    icu_census = int(census * 0.10)  # ~10% ICU
    general_census = census - icu_census
    
    # Base staff needed at any moment
    base_nurses = (general_census / general_floor_ratio) + (icu_census / icu_ratio)
    base_er_physicians = 3
    base_icu_physicians = 2
    base_surgeons = 2
    base_hospitalists = 4
    
    # Calculate rotations needed
    # Assume 12-hour shifts, but max 24 hours work over 48+ hour period
    shifts_needed = closure_hours / 12
    rotation_factor = min(shifts_needed / 2, 1.5)  # Cap at 1.5x for overlap
    
    buffer = 1.20  # 20% buffer for absences/illness
    
    requirements = {
        'nurses': int(base_nurses * rotation_factor * buffer),
        'er_physicians': int(base_er_physicians * rotation_factor * buffer),
        'icu_physicians': int(base_icu_physicians * rotation_factor * buffer),
        'surgeons': int(base_surgeons * rotation_factor * buffer),
        'hospitalists': int(base_hospitalists * rotation_factor * buffer),
    }
    
    return requirements


def get_seasonal_census_avg():
    """Get average census by month (based on typical hospital patterns)"""
    
    # Typical hospital census patterns
    # Higher in winter (flu season), lower in summer
    seasonal_census = {
        1: 520,   # January - peak flu season
        2: 510,   # February
        3: 490,   # March
        4: 470,   # April
        5: 460,   # May
        6: 440,   # June - summer low
        7: 435,   # July
        8: 440,   # August
        9: 455,   # September
        10: 470,  # October
        11: 490,  # November
        12: 515,  # December - holiday season
    }
    
    # Save for use in app
    Path('models').mkdir(exist_ok=True)
    with open('models/seasonal_census.pkl', 'wb') as f:
        pickle.dump(seasonal_census, f)
    
    print("\n" + "="*60)
    print("SEASONAL CENSUS AVERAGES")
    print("="*60)
    print("\nMonthly baseline census (450-bed hospital):")
    for month, census in seasonal_census.items():
        month_name = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]
        print(f"  {month_name:3s}: {census} patients")
    
    print("\nSaved to models/seasonal_census.pkl")
    
    return seasonal_census


def test_model_predictions():
    """Test model on actual historical storms"""
    
    print("\n" + "="*60)
    print("TESTING MODEL ON HISTORICAL STORMS")
    print("="*60)
    
    # Load model
    with open('models/closure_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    fetcher = StormDataFetcher()
    df = fetcher.get_high_quality_data()
    
    features = ['category', 'max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
    X = df[features]
    y_actual = df['closure_hours']
    
    # Predict
    y_pred = model.predict(X)
    
    # Show results
    results = pd.DataFrame({
        'Storm': df['name'].values,
        'Year': df['year'].values,
        'Actual (hrs)': y_actual.values,
        'Predicted (hrs)': y_pred,
        'Error (hrs)': np.abs(y_actual.values - y_pred)
    })
    
    print("\n")
    print(results.to_string(index=False))
    
    # Overall metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    print(f"\nMean Absolute Error: ±{mae:.1f} hours")
    print(f"Root Mean Squared Error: ±{rmse:.1f} hours")
    
    print("\nNote: This is in-sample error (trained and tested on same data)")
    print("For honest validation, use validate_model.py with LOOCV")


if __name__ == '__main__':
    print("TGH STORM STAFFING PREDICTOR")
    print("Model Training Pipeline\n")
    
    # Train closure prediction model
    model, data = train_closure_model()
    
    # Generate seasonal census data
    seasonal_census = get_seasonal_census_avg()
    
    # Test model
    test_model_predictions()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("""
Next steps:
1. Run validate_model.py for honest LOOCV validation
2. Run app.py to see the interactive dashboard
3. Use the dashboard to predict closure times for upcoming storms
    """)
