# Regression Statistics Instructions

## To Get R² and P-values:

1. **Install required packages** (if not already installed):
```bash
pip install statsmodels scipy
```

2. **Run the regression statistics script**:
```bash
python3 regression_stats.py
```

## What You'll Get:

### Model Fit Statistics:
- **R² (Coefficient of Determination)**: How well the model fits (0-1, higher is better)
- **Adjusted R²**: R² adjusted for number of predictors
- **RMSE** and **MAE**: Error metrics

### Coefficient Statistics:
For each feature, you'll see:
- **Coefficient**: The actual value
- **Standard Error**: Uncertainty in the coefficient
- **t-value**: Test statistic
- **p-value**: Statistical significance (p < 0.05 = significant)
- **95% Confidence Interval**: Range where true coefficient likely lies
- **Significance stars**: `***` (p<0.001), `**` (p<0.01), `*` (p<0.05)

### Model Significance:
- **F-statistic** and **p-value**: Tests if the overall model is significant
- **Durbin-Watson**: Tests for autocorrelation in residuals

### Residual Analysis:
- Tests if residuals are normally distributed
- Checks for patterns in prediction errors

## Expected Output Format:

```
======================================================================
REGRESSION STATISTICS & DIAGNOSTICS
======================================================================

Dataset: 17 storms
Features: 6 (category, max_wind, storm_surge, track_distance, forward_speed, month)
Target: closure_hours

======================================================================
MODEL FIT STATISTICS
======================================================================

R² (Coefficient of Determination): 0.XXXX
  Interpretation: XX.X% of variance in closure duration is explained by the model

Adjusted R²: 0.XXXX
  Accounts for number of predictors (penalizes for extra features)

Root Mean Squared Error (RMSE): XX.XX hours
Mean Absolute Error (MAE): XX.XX hours

======================================================================
COEFFICIENT STATISTICS
======================================================================

Feature          Coefficient  Std Error  t-value   p-value   95% CI Lower  95% CI Upper  Significance
Intercept        X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         
category         X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         *
max_wind         X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         **
storm_surge      X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         *
track_distance   X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         
forward_speed    X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         ***
month            X.XX         X.XX       X.XX       X.XXXX     X.XX          X.XX         

Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

======================================================================
MODEL SIGNIFICANCE TESTS
======================================================================

F-statistic: X.XXXX
F-statistic p-value: X.XXXXXX
  ✓ Model is statistically significant (p < 0.05)

Durbin-Watson statistic: X.XXXX
  (Tests for autocorrelation in residuals)
  Values close to 2 indicate no autocorrelation
  ✓ No significant autocorrelation detected
```

## Interpretation Guide:

### R² Values:
- **0.7-1.0**: Excellent fit
- **0.5-0.7**: Good fit
- **0.3-0.5**: Moderate fit
- **<0.3**: Poor fit

### P-values:
- **p < 0.001 (***)**: Highly significant
- **p < 0.01 (**)**: Very significant
- **p < 0.05 (*)**: Significant
- **p < 0.1 (.)**: Marginally significant
- **p >= 0.1**: Not significant

### What to Look For:
1. **R² > 0.5**: Model explains at least half the variance
2. **F-statistic p < 0.05**: Overall model is statistically significant
3. **Individual feature p-values < 0.05**: Those features are significant predictors
4. **Durbin-Watson ~2**: No autocorrelation issues
5. **Normally distributed residuals**: Model assumptions are met

## Alternative: Quick R² Calculation

If you just need R² quickly without p-values, you can use:

```python
from sklearn.metrics import r2_score
from data_fetcher import StormDataFetcher
from train_models import train_closure_model

model, df = train_closure_model()
features = ['category', 'max_wind', 'storm_surge', 'track_distance', 'forward_speed', 'month']
X = df[features]
y = df['closure_hours']
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² = {r2:.4f}")
```

