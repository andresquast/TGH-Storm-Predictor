# TGH Storm Closure Duration Predictor

A predictive model for estimating bridge closure duration and staffing requirements during hurricanes and tropical storms affecting the Tampa Bay area. The model uses historical storm data (1996-2024) to predict how long bridges connecting Tampa General Hospital to the mainland will remain closed, enabling proactive staffing and resource planning.

The system provides:

- **Bridge closure duration predictions** based on storm characteristics
- **Physician staffing requirements** calculated from predicted closure time and seasonal hospital census
- **Client-side predictions** - works entirely in the browser without requiring a backend server

Live site: [https://andresquast.github.io/TGH-Storm-Predictor/](https://andresquast.github.io/TGH-Storm-Predictor/)

## Data

### Historical Storms (1996-2024)

The dataset includes 17 storms with bridge closure data:

**High Quality Data (12 storms):**

- Hurricane Irma (2017) - 22 hours
- Tropical Storm Eta (2020) - 18 hours
- Hurricane Ian (2022) - 27 hours
- Hurricane Idalia (2023) - 12 hours
- Hurricane Debby (2024) - 40 hours
- Hurricane Helene (2024) - 29 hours
- Hurricane Milton (2024) - 24 hours
- Tropical Storm Elsa (2021) - 18 hours
- Tropical Storm Nicole (2022) - 9 hours
- Hurricane Hermine (2016) - 27 hours
- Tropical Storm Andrea (2013) - 6 hours
- Tropical Storm Debby (2012) - 48 hours

**Medium Quality Data (5 storms):**

- Hurricane Fay (2008) - 22 hours
- Hurricane Jeanne (2004) - 14 hours
- Hurricane Frances (2004) - 16 hours
- Tropical Storm Gabrielle (2001) - 12 hours
- Hurricane Josephine (1996) - 18 hours

**Closure Duration Statistics:**

- Mean: 21.3 hours
- Median: 18.0 hours
- Range: 6 - 48 hours
- Standard Deviation: 11.0 hours

### Data Sources

- NOAA National Hurricane Center
- Florida Highway Patrol
- Tampa Bay Times
- Florida Department of Transportation
- National Weather Service Tampa Bay

## Features Used for Prediction

The model uses 5 features to predict bridge closure duration:

1. **Max Wind** - Maximum sustained winds at Tampa (mph)
2. **Storm Surge** - Storm surge at Tampa Bay (feet)
3. **Track Distance** - Distance from TGH at closest approach (miles)
4. **Forward Speed** - Storm forward motion (mph) - **KEY PREDICTOR**
5. **Month** - Month of year (for seasonal effects)

**Key Finding**: Forward speed has the strongest correlation with closure duration (-0.623). Slower-moving storms result in significantly longer bridge closures:

- Slow storms (2-5 mph) → 40-48 hour closures
- Fast storms (18+ mph) → 6-18 hour closures

## Model Performance

### Leave-One-Out Cross-Validation Results

The model is validated using Leave-One-Out Cross-Validation (LOOCV), where each storm is predicted using a model trained on all other storms:

- **Mean Absolute Error**: ±4-6 hours (estimated with 17 storms)
- **Model Type**: Linear Regression
- **Features**: 5 features (max_wind, storm_surge, track_distance, forward_speed, month)
- **Training Method**: Each prediction uses model trained on 16 other storms

**Validation Method:**

- Uses LOOCV - each storm predicted using model trained on all OTHER storms
- Honest error estimates - not backfit
- Proof-of-concept validated on historical data

## Current Limitations

1. **Small sample size** - 17 storms with closure data (though covering 28 years)
2. **Missing features** - Angle of approach, tide timing, time of day, bridge-specific factors
3. **Data quality variation** - 5 storms have medium-quality data (estimated closure times from historical records)
4. **Limited feature engineering** - Simple linear model; could benefit from non-linear relationships
5. **Temporal gaps** - Some years have no storms (e.g., 2005-2011 had no bridge closures)

---

**Last Updated**: December 2024  
**Data Through**: Hurricane Milton (October 2024)  
**Model Version**: 1.0
