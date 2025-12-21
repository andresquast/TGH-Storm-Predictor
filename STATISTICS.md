# Statistical Summary - TGH Storm Predictor
## Dataset: 17 Storms (1996-2024)

## 📊 Dataset Overview
- **Total storms with closure data:** 17
- **Date range:** 1996 - 2024 (28 years)
- **Data quality breakdown:**
  - High quality: 12 storms
  - Medium quality: 5 storms

## ⏱️ Bridge Closure Duration Statistics

| Statistic | Value |
|-----------|-------|
| Mean | 21.3 hours (0.9 days) |
| Median | 18.0 hours (0.75 days) |
| Standard Deviation | 11.0 hours |
| Minimum | 6 hours (Andrea, 2013) |
| Maximum | 48 hours (Debby 2012) |
| 25th Percentile | 14.0 hours |
| 75th Percentile | 27.0 hours |
| Interquartile Range (IQR) | 13.0 hours |

**Closure Duration Distribution:**
- Short closures (≤12 hours): 3 storms (18%)
- Medium closures (13-24 hours): 8 storms (47%)
- Long closures (≥25 hours): 6 storms (35%)

## 🌪️ Storm Characteristics

### Wind Speed (mph)
- **Mean:** 72.1 mph
- **Median:** 70.0 mph
- **Range:** 48 - 150 mph
- **Standard Deviation:** 30.2 mph

### Storm Surge (feet)
- **Mean:** 4.2 feet
- **Median:** 4.0 feet
- **Range:** 2.0 - 9.0 feet
- **Standard Deviation:** 1.6 feet

### Track Distance (miles from Tampa)
- **Mean:** 78.2 miles
- **Median:** 60.0 miles
- **Range:** 20 - 150 miles
- **Standard Deviation:** 42.1 miles

### Forward Speed (mph)
- **Mean:** 12.1 mph
- **Median:** 13.0 mph
- **Range:** 2 - 24 mph
- **Standard Deviation:** 5.8 mph

## 📅 Temporal Distribution

### By Month
- **June:** 2 storms (12%)
- **July:** 1 storm (6%)
- **August:** 3 storms (18%)
- **September:** 7 storms (41%) ⭐ Peak season
- **October:** 2 storms (12%)
- **November:** 2 storms (12%)

### By Category
- **Category 1 (Tropical Storm):** 13 storms (76%)
- **Category 3:** 2 storms (12%)
- **Category 4:** 2 storms (12%)

## 🔗 Correlations with Closure Duration

| Feature | Correlation | Direction |
|---------|-------------|-----------|
| **forward_speed** | -0.623 | ↓ Strong negative |
| **storm_surge** | +0.456 | ↑ Moderate positive |
| **max_wind** | +0.389 | ↑ Moderate positive |
| **track_distance** | +0.234 | ↑ Weak positive |
| **category** | +0.198 | ↑ Weak positive |
| **month** | -0.089 | ↓ Very weak negative |

**Key Insights:**
- **Forward speed** is the strongest predictor (slower storms = longer closures)
- **Storm surge** and **wind speed** also positively correlate with closure duration
- **Track distance** has a weak positive correlation (closer storms tend to have longer closures)

## 🎯 Model Performance (Expected with 17 storms)

Based on Leave-One-Out Cross-Validation with 17 storms:
- **Mean Absolute Error:** ~4-6 hours (estimated)
- **Model uses:** Linear Regression with 6 features
- **Training:** Each prediction uses model trained on 16 other storms

## 📈 Comparison: 12 vs 17 Storms

| Metric | 12 Storms (High Quality Only) | 17 Storms (All) |
|--------|-------------------------------|-----------------|
| Date Range | 2012-2024 | 1996-2024 |
| Mean Closure | ~20 hours | 21.3 hours |
| Forward Speed Range | 7-24 mph | 2-24 mph |
| Historical Coverage | 12 years | 28 years |

**Benefits of 17-storm dataset:**
- Includes slower storms (2-5 mph) that weren't in high-quality set
- Better temporal coverage (1996-2024 vs 2012-2024)
- More diverse storm characteristics
- Better representation of extreme events

