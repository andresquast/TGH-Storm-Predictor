# TGH Storm Staffing Predictor

Personal project: Predictive model for bridge closure duration and staffing requirements during hurricanes affecting the Tampa Bay area.

## Project Overview

This system predicts:
1. **Bridge closure duration** based on storm characteristics
2. **Staffing requirements** based on expected closure time and seasonal hospital census
3. **Model validation** using Leave-One-Out Cross-Validation

## Quick Start

### Backend Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Train the model (if not already trained)
python train_models.py

# 3. Run the Flask backend server
python app.py
```

The Flask API will run on `http://localhost:5000`

### Frontend Setup (React)

```bash
# 1. Install Node.js dependencies
npm install

# 2. Start React development server
npm run dev
```

The React app will run on `http://localhost:3000` and proxy API requests to the Flask backend.

### Production Build

```bash
# Build React app for production
npm run build

# The Flask app will serve the built React app from the / route
python app.py
```

### Running Analysis Pipeline

```bash
# Run complete analysis pipeline
python run_analysis.py

# Run individual components
python data_fetcher.py          # Load and explore data
python train_models.py          # Train prediction models
python validate_model.py        # Validate with LOOCV
```

## Project Structure

```
tgh-storm-predictor/
├── data/
│   ├── historical_storms.json     # Historical storm data with bridge closures
│   └── modeling_data.csv          # Exported clean data (auto-generated)
│
├── models/
│   ├── closure_model.pkl          # Trained prediction model
│   └── seasonal_census.pkl        # Monthly census averages
│
├── src/                           # React frontend source
│   ├── components/                # React components
│   │   ├── StormForm.jsx          # Input form component
│   │   ├── PredictionResult.jsx   # Results display component
│   │   └── ErrorMessage.jsx       # Error display component
│   ├── App.jsx                    # Main React app component
│   ├── main.jsx                   # React entry point
│   └── index.css                  # Global styles
│
├── app.py                         # Flask backend API server
├── data_fetcher.py                # Loads data from JSON
├── train_models.py                # Trains prediction models
├── validate_model.py              # LOOCV validation
├── run_analysis.py                # Complete analysis pipeline
├── package.json                   # Node.js dependencies
├── vite.config.js                 # Vite build configuration
├── validation_results.png         # LOOCV visualization (auto-generated)
├── feature_importance.png         # Feature coefficients visualization (auto-generated)
└── README.md                       # This file
```

## Data

### Historical Storms (2004-2024)

The dataset includes storms with varying levels of data quality:

**High Quality Data (8 storms with bridge closures):**
- Hurricane Irma (2017) - 22 hour closure
- Tropical Storm Eta (2020) - 18 hour closure  
- Hurricane Ian (2022) - 27 hour closure
- Hurricane Idalia (2023) - 12 hour closure
- Hurricane Debby (2024) - 40 hour closure
- Hurricane Helene (2024) - 29 hour closure
- Hurricane Milton (2024) - 24 hour closure
- Tropical Storm Elsa (2021) - 18 hour closure

### Data Sources

- NOAA National Hurricane Center
- Florida Highway Patrol
- Tampa Bay Times
- Florida Department of Transportation  
- National Weather Service Tampa Bay

## Features Used for Prediction

1. **Category** - Hurricane category (Saffir-Simpson scale)
2. **Max Wind** - Maximum sustained winds at Tampa (mph)
3. **Storm Surge** - Storm surge at Tampa Bay (feet)
4. **Track Distance** - Distance from TGH at closest approach (miles)
5. **Forward Speed** - Storm forward motion (mph) - KEY PREDICTOR
6. **Month** - Month of year (for seasonal effects)

## Model Performance

### Leave-One-Out Cross-Validation Results

- **Mean Absolute Error**: ±0.9 hours
- **Best case**: ±0.1 hours
- **Worst case**: ±3.4 hours

**Key Finding**: Forward speed is the most important predictor
- Slow storms (2 mph) → 40 hour closures
- Fast storms (18+ mph) → 12-18 hour closures

### Validation Method

- Uses LOOCV - each storm predicted using model trained on all OTHER storms
- Honest error estimates - not backfit
- Proof-of-concept validated

## Usage Examples

### 1. Load and Explore Data

```python
from data_fetcher import StormDataFetcher

fetcher = StormDataFetcher()
fetcher.load_data()
fetcher.get_summary_statistics()

df = fetcher.get_high_quality_data()
print(df.head())
```

### 2. Train Model

```python
# Automatically loads from historical_storms.json
python train_models.py
```

### 3. Validate Model

```python
# Performs LOOCV and creates visualization
python validate_model.py
```

### 4. Make Predictions

```python
import pickle
import pandas as pd

# Load model
with open('models/closure_model.pkl', 'rb') as f:
    model = pickle.load(f)

# New storm data
storm = pd.DataFrame([[
    3,      # category
    120,    # max_wind
    9,      # storm_surge
    20,     # track_distance
    16,     # forward_speed
    10      # month
]], columns=['category', 'max_wind', 'storm_surge', 
             'track_distance', 'forward_speed', 'month'])

prediction = model.predict(storm)[0]
print(f"Predicted closure: {prediction:.1f} hours")
```

## Current Limitations

1. **Small sample size** - 8 storms with closure data
2. **Missing features** - Angle of approach, tide timing, time of day
3. **Limited historical range** - Only back to 2004, gaps in 2005-2016

## Roadmap

### Phase 1: Data Expansion
- Gather 15-20 more historical storms (2000-2020)
- Include storms that approached but did not close bridges  
- Partner with FDOT/FHP for official closure logs
- Add missing features (angle, tide, etc.)

### Phase 2: Model Enhancement
- Reduce error to ±2 hours with expanded data
- Add ensemble methods
- Build confidence intervals around predictions
- Validate on recent storms prospectively

### Phase 3: Operational Deployment
- Real-time integration with NHC storm data
- Automated alerts at T-72, T-48, T-24 hours
- ✅ Frontend interface for predictions (React web app)

## Adding New Storms

To add new historical storms to the dataset:

1. Edit `data/historical_storms.json`
2. Add new entry to the `storms` array:

```json
{
  "name": "NewStorm",
  "date": "2025-09-15",
  "year": 2025,
  "month": 9,
  "category_at_tampa": 3,
  "max_wind_tampa": 110,
  "storm_surge_tampa": 7,
  "track_distance_miles": 30,
  "forward_speed_mph": 12,
  "closure_hours": 28,
  "bridge_closed_time": "2025-09-15T14:00:00",
  "bridge_opened_time": "2025-09-16T18:00:00",
  "bridges_affected": ["Sunshine Skyway", "Howard Frankland"],
  "landfall_location": "Location",
  "notes": "Additional context",
  "data_quality": "high"
}
```

3. Retrain model: `python train_models.py`
4. Revalidate: `python validate_model.py`

## License

Personal project - for educational and research purposes.

---

**Last Updated**: December 2024  
**Data Through**: Hurricane Milton (October 2024)  
**Model Version**: 1.0
