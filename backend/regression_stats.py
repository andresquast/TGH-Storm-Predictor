#!/usr/bin/env python3
"""
Regression Statistics and Diagnostics for TGH Storm Predictor
Calculates R², p-values, F-statistics, and other regression diagnostics
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from scipy import stats
from data_fetcher import StormDataFetcher

def calculate_regression_statistics():
    """Calculate comprehensive regression statistics"""
    
    print("="*70)
    print("REGRESSION STATISTICS & DIAGNOSTICS")
    print("="*70)
    
    # Load data
    fetcher = StormDataFetcher()
    df = fetcher.get_storms_dataframe(include_no_closure=False)
    
    print(f"\nDataset: {len(df)} storms")
    print(f"Features: 5 (max_wind, storm_surge, track_distance, forward_speed, month)")
    print(f"Target: closure_hours")
    
    # Prepare features and target
    features = ['max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
    X = df[features]
    y = df['closure_hours']
    
    # Add constant for statsmodels (intercept)
    X_with_const = sm.add_constant(X)
    
    # Fit model using statsmodels (for p-values)
    model_sm = sm.OLS(y, X_with_const).fit()
    
    # Also fit with sklearn for consistency
    model_sk = LinearRegression()
    model_sk.fit(X, y)
    y_pred = model_sk.predict(X)
    
    print("\n" + "="*70)
    print("MODEL FIT STATISTICS")
    print("="*70)
    
    # R-squared
    r2 = r2_score(y, y_pred)
    n = len(y)
    p = len(features)
    print(f"\nR² (Coefficient of Determination): {r2:.4f}")
    print(f"  Interpretation: {r2*100:.1f}% of variance in closure duration is explained by the model")
    
    # Adjusted R-squared
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"\nAdjusted R²: {adj_r2:.4f}")
    print(f"  Accounts for number of predictors (penalizes for extra features)")
    
    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f} hours")
    print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
    
    # RMSE confidence interval (approximate using standard error)
    rmse_se = rmse / np.sqrt(2 * (n - p - 1))
    t_critical = stats.t.ppf(0.975, n - p - 1)
    rmse_ci_lower = max(0, rmse - t_critical * rmse_se)
    rmse_ci_upper = rmse + t_critical * rmse_se
    print(f"  RMSE 95% Confidence Interval: [{rmse_ci_lower:.2f}, {rmse_ci_upper:.2f}] hours")
    
    print("\n" + "="*70)
    print("COEFFICIENT STATISTICS")
    print("="*70)
    
    # Create coefficient summary
    conf_int = model_sm.conf_int()
    coef_df = pd.DataFrame({
        'Feature': ['Intercept'] + features,
        'Coefficient': [model_sm.params[0]] + list(model_sm.params[1:]),
        'Std Error': [model_sm.bse[0]] + list(model_sm.bse[1:]),
        't-value': [model_sm.tvalues[0]] + list(model_sm.tvalues[1:]),
        'p-value': [model_sm.pvalues[0]] + list(model_sm.pvalues[1:]),
        '95% CI Lower': [conf_int.iloc[0, 0]] + list(conf_int.iloc[1:, 0]),
        '95% CI Upper': [conf_int.iloc[0, 1]] + list(conf_int.iloc[1:, 1])
    })
    
    # Add significance stars
    def significance_star(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        elif p < 0.1:
            return '.'
        else:
            return ''
    
    coef_df['Significance'] = coef_df['p-value'].apply(significance_star)
    
    # Add interpretation for confidence intervals
    coef_df['CI Interpretation'] = coef_df.apply(
        lambda row: 'Significant' if (row['95% CI Lower'] > 0 and row['95% CI Upper'] > 0) or 
                                      (row['95% CI Lower'] < 0 and row['95% CI Upper'] < 0) 
                   else 'Not Significant (CI contains 0)', axis=1
    )
    
    print("\n" + coef_df.to_string(index=False))
    print("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    print("\nConfidence Interval Interpretation:")
    print("  - If 95% CI does not contain 0, the coefficient is statistically significant at α=0.05")
    print("  - CI Lower > 0 and CI Upper > 0: Positive effect (significant)")
    print("  - CI Lower < 0 and CI Upper < 0: Negative effect (significant)")
    print("  - CI Lower < 0 and CI Upper > 0: Effect not statistically significant")
    
    print("\n" + "="*70)
    print("MODEL SIGNIFICANCE TESTS")
    print("="*70)
    
    # F-statistic
    f_stat = model_sm.fvalue
    f_pvalue = model_sm.f_pvalue
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"F-statistic p-value: {f_pvalue:.6f}")
    if f_pvalue < 0.05:
        print("  ✓ Model is statistically significant (p < 0.05)")
    else:
        print("  ✗ Model is NOT statistically significant (p >= 0.05)")
    
    # Durbin-Watson (for autocorrelation)
    dw = sm.stats.durbin_watson(model_sm.resid)
    print(f"\nDurbin-Watson statistic: {dw:.4f}")
    print("  (Tests for autocorrelation in residuals)")
    print("  Values close to 2 indicate no autocorrelation")
    if 1.5 < dw < 2.5:
        print("  ✓ No significant autocorrelation detected")
    else:
        print("  ⚠ Possible autocorrelation in residuals")
    
    print("\n" + "="*70)
    print("RESIDUAL ANALYSIS")
    print("="*70)
    
    residuals = y - y_pred
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.4f} hours (should be ~0)")
    print(f"  Std Dev: {residuals.std():.2f} hours")
    print(f"  Min: {residuals.min():.2f} hours")
    print(f"  Max: {residuals.max():.2f} hours")
    
    # Normality test (Shapiro-Wilk)
    if len(residuals) <= 50:  # Shapiro-Wilk works best for n <= 50
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"\nShapiro-Wilk normality test:")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  p-value: {shapiro_p:.4f}")
        if shapiro_p > 0.05:
            print("  ✓ Residuals appear normally distributed (p > 0.05)")
        else:
            print("  ⚠ Residuals may not be normally distributed (p <= 0.05)")
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Standardized Coefficients)")
    print("="*70)
    
    # Standardize features to compare coefficients
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    
    model_std = LinearRegression()
    model_std.fit(X_std, y_std)
    
    std_coefs = pd.DataFrame({
        'Feature': features,
        'Standardized Coefficient': model_std.coef_,
        'Absolute Value': np.abs(model_std.coef_)
    }).sort_values('Absolute Value', ascending=False)
    
    print("\n" + std_coefs[['Feature', 'Standardized Coefficient']].to_string(index=False))
    print("\n(Standardized coefficients allow comparison of feature importance)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    significant_features = coef_df[coef_df['p-value'] < 0.05]['Feature'].tolist()
    significant_features = [f for f in significant_features if f != 'Intercept']
    
    print(f"\n✓ Model R²: {r2:.3f} ({r2*100:.1f}% variance explained)")
    print(f"✓ Statistically significant features (p < 0.05): {len(significant_features)}")
    if significant_features:
        print(f"  - {', '.join(significant_features)}")
    print(f"✓ Model F-statistic p-value: {f_pvalue:.6f}")
    print(f"✓ Mean prediction error: ±{mae:.2f} hours")
    
    print("\n" + "="*70)
    
    return model_sm, coef_df

if __name__ == '__main__':
    try:
        model, coefs = calculate_regression_statistics()
    except ImportError as e:
        print(f"Error: Missing required package. Install with: pip install statsmodels")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

