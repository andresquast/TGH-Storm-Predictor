# Model Data Files

This directory contains JSON files exported from the trained model for client-side predictions.

## Generating Model Files

To generate these files, run:

```bash
python3 export_model.py
```

This will create:
- `model.json` - Linear regression model coefficients and parameters
- `seasonal_census.json` - Monthly census averages
- `staffing_constants.json` - Staffing calculation constants

## Updating the Model

When you retrain the model (by running `train_models.py`), you should regenerate these JSON files by running `export_model.py` again. The updated files will be included in the next build.

## Note

These files are included in the build and allow the frontend to make predictions entirely in the browser without requiring a backend API.

