#!/usr/bin/env python3
"""
Master script to run complete TGH Storm Prediction analysis
Runs all steps: data loading, training, validation, and summary
"""

import sys
from io import StringIO

# Suppress verbose output
class SuppressOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self.stdout

print("TGH Storm Staffing Predictor - Running Analysis...\n")

# Step 1: Data Loading
print("Loading data...", end=" ", flush=True)
try:
    from data_fetcher import StormDataFetcher
    
    with SuppressOutput():
        fetcher = StormDataFetcher()
        fetcher.load_data()
        stats = fetcher.get_summary_statistics()
    
    print("Done")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Step 2: Model Training
print("Training models...", end=" ", flush=True)
try:
    from train_models import train_closure_model, get_seasonal_census_avg, test_model_predictions
    
    with SuppressOutput():
        model, data = train_closure_model()
        seasonal = get_seasonal_census_avg()
        test_model_predictions()
    
    print("Done")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Step 3: Validation
print("Validating model...", end=" ", flush=True)
try:
    from validate_model import leave_one_out_validation, visualize_validation_results, visualize_feature_importance, feature_importance_analysis
    
    with SuppressOutput():
        results = leave_one_out_validation()
        feature_importance_analysis()
        fig = visualize_validation_results(results)
        fig2 = visualize_feature_importance()
    
    print("Done")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Final Summary
print("\nAnalysis complete.")
print(f"Mean Absolute Error: ±{results['error_hours'].mean():.1f} hours")
print("\nFiles generated:")
print("  - validation_results.png")
print("  - feature_importance.png")
print("  - models/closure_model.pkl")
print("  - models/seasonal_census.pkl")
