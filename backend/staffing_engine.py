"""
TGH Predictive Staffing Engine
Integrates storm-driven bridge closures with clinical demand for physician staffing
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path

# Optional imports for validation functionality
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    from sklearn.metrics import mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# CORE DATA CONSTANTS (FY2024 Metrics)
# ============================================================================

class TGHConstants:
    """Core hospital metrics and staffing ratios"""
    
    # Inpatient Metrics
    AVERAGE_DAILY_CENSUS = 1111  # ADC
    LICENSED_BEDS = 1353
    
    # Emergency/Trauma Metrics
    ANNUAL_ER_VISITS = 221334  # Combined Adult/Pediatric
    DAILY_ER_VISITS_AVG = ANNUAL_ER_VISITS / 365  # ~606 visits/day
    
    # Medical Staff
    TOTAL_CREDENTIALED_PHYSICIANS = 2567
    
    # Staffing Ratios
    INPATIENT_CRISIS_RATIO = 18  # 1 MD per 18 patients (Crisis staffing)
    ER_MD_PATIENT_RATIO = 2.5  # 1 MD per 2.5 patients/hour
    
    # Specialty Buffer (Skeleton Crew per shift)
    SPECIALTY_SKELETON_CREW = 15  # Transplant, Surgery, Anesthesia specialists
    
    # Shift Duration
    SHIFT_DURATION_HOURS = 12


# ============================================================================
# TEMPORAL & SEASONAL INTELLIGENCE
# ============================================================================

class TemporalAdjustment:
    """Handles temporal adjustments for census and ER flow"""
    
    def __init__(self):
        """Initialize temporal multipliers"""
        self.monthly_census_multipliers = self._initialize_monthly_multipliers()
        self.diurnal_er_multipliers = self._initialize_diurnal_multipliers()
    
    def _initialize_monthly_multipliers(self) -> Dict[int, float]:
        """
        Monthly multipliers for census
        Jan-Mar: 1.25 (Peak Snowbird/Flu)
        Apr-May & Oct-Dec: 1.0 (Baseline)
        Jun-Sep: 0.90 (Low Elective)
        """
        return {
            1: 1.25,   # January - Peak Snowbird/Flu
            2: 1.25,   # February - Peak Snowbird/Flu
            3: 1.25,   # March - Peak Snowbird/Flu
            4: 1.0,    # April - Baseline
            5: 1.0,    # May - Baseline
            6: 0.90,   # June - Low Elective
            7: 0.90,   # July - Low Elective
            8: 0.90,   # August - Low Elective
            9: 0.90,   # September - Low Elective
            10: 1.0,   # October - Baseline
            11: 1.0,   # November - Baseline
            12: 1.0,   # December - Baseline
        }
    
    def _initialize_diurnal_multipliers(self) -> Dict[str, float]:
        """
        Diurnal multipliers for ER flow
        10:00-22:00: 1.2x (Peak)
        22:00-10:00: 0.8x (Off-Peak)
        """
        return {
            'peak': 1.2,      # 10:00-22:00
            'off_peak': 0.8   # 22:00-10:00
        }
    
    def get_monthly_census_multiplier(self, month: int) -> float:
        """Get census multiplier for given month"""
        return self.monthly_census_multipliers.get(month, 1.0)
    
    def get_diurnal_er_multiplier(self, hour: int) -> float:
        """
        Get ER flow multiplier for given hour
        
        Args:
            hour: Hour of day (0-23)
        
        Returns:
            Multiplier (1.2 for peak hours 10-22, 0.8 for off-peak)
        """
        if 10 <= hour < 22:
            return self.diurnal_er_multipliers['peak']
        else:
            return self.diurnal_er_multipliers['off_peak']
    
    def get_storm_surge_er_multiplier(self, month: int) -> float:
        """
        Apply +20% surge factor to ER/Trauma for storm injuries during Jun-Sep
        
        Args:
            month: Month (1-12)
        
        Returns:
            Surge multiplier (1.2 for Jun-Sep, 1.0 otherwise)
        """
        if 6 <= month <= 9:  # Jun-Sep
            return 1.2  # +20% surge for storm injuries
        return 1.0
    
    def adjust_census(self, base_census: float, month: int) -> float:
        """Apply monthly adjustment to census"""
        multiplier = self.get_monthly_census_multiplier(month)
        return base_census * multiplier
    
    def adjust_er_flow(self, base_er_visits_per_hour: float, hour: int, month: int) -> float:
        """
        Apply diurnal and storm surge adjustments to ER flow
        
        Args:
            base_er_visits_per_hour: Base ER visits per hour
            hour: Hour of day (0-23)
            month: Month (1-12)
        
        Returns:
            Adjusted ER visits per hour
        """
        diurnal_mult = self.get_diurnal_er_multiplier(hour)
        surge_mult = self.get_storm_surge_er_multiplier(month)
        return base_er_visits_per_hour * diurnal_mult * surge_mult


# ============================================================================
# STAFFING CALCULATION FUNCTIONS
# ============================================================================

def calculate_inpatient_mds(census: float) -> int:
    """
    Calculate required inpatient MDs based on crisis ratio
    
    Args:
        census: Adjusted inpatient census
    
    Returns:
        Number of MDs required
    """
    return int(np.ceil(census / TGHConstants.INPATIENT_CRISIS_RATIO))


def calculate_er_mds(er_visits_per_hour: float) -> int:
    """
    Calculate required ER MDs based on patient flow
    
    Args:
        er_visits_per_hour: ER visits per hour
    
    Returns:
        Number of ER MDs required
    """
    return int(np.ceil(er_visits_per_hour / TGHConstants.ER_MD_PATIENT_RATIO))


def calculate_bridge_event_staffing(
    predicted_closure_start: datetime,
    predicted_duration_hours: float,
    current_actual_census: Optional[int] = None
) -> Dict:
    """
    Calculate physician staffing requirements for bridge closure event
    
    Args:
        predicted_closure_start: Datetime when bridge closure is predicted to start
        predicted_duration_hours: Predicted duration of bridge closure in hours
        current_actual_census: Current actual census (defaults to ADC if None)
    
    Returns:
        Structured JSON object with staffing requirements and metrics
    """
    
    # Initialize temporal adjustment
    temporal = TemporalAdjustment()
    
    # Use current census or default to ADC
    base_census = current_actual_census if current_actual_census is not None else TGHConstants.AVERAGE_DAILY_CENSUS
    
    # Apply monthly adjustment to census
    month = predicted_closure_start.month
    adjusted_census = temporal.adjust_census(base_census, month)
    
    # Calculate base inpatient MDs needed
    inpatient_mds = calculate_inpatient_mds(adjusted_census)
    
    # Calculate ER MDs needed (considering peak hours during closure)
    # Use average of peak and off-peak multipliers for duration
    base_er_visits_per_hour = TGHConstants.DAILY_ER_VISITS_AVG / 24
    peak_er_visits = temporal.adjust_er_flow(base_er_visits_per_hour, 14, month)  # Mid-afternoon peak
    off_peak_er_visits = temporal.adjust_er_flow(base_er_visits_per_hour, 2, month)  # Early morning off-peak
    avg_er_visits_per_hour = (peak_er_visits + off_peak_er_visits) / 2
    er_mds = calculate_er_mds(avg_er_visits_per_hour)
    
    # Calculate shifts needed
    shifts_needed = int(np.ceil(predicted_duration_hours / TGHConstants.SHIFT_DURATION_HOURS))
    
    # Team A: Island Isolation (MDs required to stay on Davis Island)
    # For duration > 12 hours, calculate 12-hour rotating shifts
    mds_per_shift = inpatient_mds + er_mds + TGHConstants.SPECIALTY_SKELETON_CREW
    
    if predicted_duration_hours <= TGHConstants.SHIFT_DURATION_HOURS:
        # Single shift - all MDs stay for full duration
        team_a_count = mds_per_shift
    else:
        # Multiple shifts - need rotation coverage
        # Standard rotation: 2 teams (one on, one off) for up to 48 hours
        # For longer closures (>48 hours), add additional teams to prevent burnout
        if predicted_duration_hours <= 48:
            # Two teams rotating (standard 12-hour shift rotation)
            team_a_count = mds_per_shift * 2
        else:
            # Extended closure: need 3 teams for adequate rest during long isolation
            # Each team works 12 hours, then rests 24 hours before next shift
            team_a_count = mds_per_shift * 3
    
    # Team B: Pre-positioning (MDs that must cross bridges before closure)
    # These are MDs who need to arrive before predicted_closure_start
    # Typically same as Team A count (they're the ones who will be isolated)
    team_b_preposition = team_a_count
    
    # Total physicians needed
    total_physicians_needed = team_a_count
    
    # Calculate risk score (1-10)
    # Factors: duration, census level, time of day, month
    risk_score = calculate_risk_score(
        predicted_duration_hours,
        adjusted_census,
        predicted_closure_start.hour,
        month
    )
    
    # Calculate capacity utilization
    capacity_utilization = (adjusted_census / TGHConstants.LICENSED_BEDS) * 100
    
    # Build result dictionary
    result = {
        "total_physicians_needed": int(total_physicians_needed),
        "team_a_count": int(team_a_count),
        "team_b_preposition": int(team_b_preposition),
        "risk_score": round(risk_score, 1),
        "capacity_utilization": round(capacity_utilization, 1),
        "details": {
            "predicted_closure_start": predicted_closure_start.isoformat(),
            "predicted_duration_hours": round(predicted_duration_hours, 1),
            "adjusted_census": round(adjusted_census, 0),
            "inpatient_mds": inpatient_mds,
            "er_mds": er_mds,
            "specialty_skeleton_crew": TGHConstants.SPECIALTY_SKELETON_CREW,
            "shifts_needed": shifts_needed,
            "month": month,
            "monthly_multiplier": temporal.get_monthly_census_multiplier(month),
            "er_surge_multiplier": temporal.get_storm_surge_er_multiplier(month)
        }
    }
    
    return result


def calculate_risk_score(
    duration_hours: float,
    census: float,
    hour: int,
    month: int
) -> float:
    """
    Calculate risk score (1-10) based on multiple factors
    
    Args:
        duration_hours: Closure duration
        census: Adjusted census
        hour: Hour of day
        month: Month
    
    Returns:
        Risk score from 1-10
    """
    risk = 1.0
    
    # Duration factor (longer = higher risk)
    if duration_hours <= 12:
        risk += 1.0
    elif duration_hours <= 24:
        risk += 2.5
    elif duration_hours <= 48:
        risk += 4.0
    else:
        risk += 5.5
    
    # Census factor (higher census = higher risk)
    census_utilization = census / TGHConstants.LICENSED_BEDS
    if census_utilization < 0.7:
        risk += 0.5
    elif census_utilization < 0.85:
        risk += 1.0
    elif census_utilization < 0.95:
        risk += 2.0
    else:
        risk += 3.0
    
    # Time of day factor (peak hours = higher risk)
    if 10 <= hour < 22:
        risk += 0.5
    
    # Month factor (peak season = higher risk)
    temporal = TemporalAdjustment()
    if temporal.get_monthly_census_multiplier(month) > 1.0:
        risk += 0.5
    
    # Cap at 10
    return min(risk, 10.0)


# ============================================================================
# VALIDATION & ERROR HANDLING
# ============================================================================

def validate_staffing_predictions_loocv(
    historical_data,
    closure_column: str = 'closure_hours',
    date_column: str = 'date',
    census_column: Optional[str] = None
) -> Dict:
    """
    LOOCV validation wrapper for staffing predictions
    
    Args:
        historical_data: DataFrame with historical bridge closures and actual staff shortages
        closure_column: Column name for closure duration in hours
        date_column: Column name for closure start date/datetime
        census_column: Optional column name for census at time of closure
    
    Returns:
        Dictionary with MAE and validation results
    """
    
    if not HAS_PANDAS:
        raise ImportError("pandas is required for validation. Install with: pip install pandas")
    
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for validation. Install with: pip install scikit-learn")
    
    if len(historical_data) < 2:
        raise ValueError("Need at least 2 data points for LOOCV")
    
    predictions = []
    actuals = []
    
    # Leave-One-Out Cross-Validation
    for i in range(len(historical_data)):
        # Test on one storm
        test_row = historical_data.iloc[i]
        
        # Parse closure start datetime
        closure_start_str = test_row[date_column]
        if isinstance(closure_start_str, str):
            try:
                closure_start = datetime.fromisoformat(closure_start_str)
            except:
                # Try parsing common date formats
                if HAS_PANDAS:
                    closure_start = pd.to_datetime(closure_start_str)
                else:
                    # Fallback to datetime parsing
                    from dateutil import parser
                    closure_start = parser.parse(closure_start_str)
        else:
            if HAS_PANDAS:
                closure_start = pd.to_datetime(closure_start_str)
            else:
                closure_start = datetime.fromisoformat(str(closure_start_str))
        
        predicted_duration = test_row[closure_column]
        census = test_row[census_column] if census_column and census_column in test_row else None
        
        # Make prediction
        try:
            result = calculate_bridge_event_staffing(
                predicted_closure_start=closure_start,
                predicted_duration_hours=predicted_duration,
                current_actual_census=census
            )
            
            predicted_staffing = result['total_physicians_needed']
            
            # Get actual staffing shortage (if available)
            if 'actual_staff_shortage' in test_row:
                actual_staffing = test_row['actual_staff_shortage']
                predictions.append(predicted_staffing)
                actuals.append(actual_staffing)
            elif 'actual_physicians_needed' in test_row:
                actual_staffing = test_row['actual_physicians_needed']
                predictions.append(predicted_staffing)
                actuals.append(actual_staffing)
            else:
                # No actual data available, skip this validation point
                continue
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    
    if len(predictions) == 0:
        return {
            "mae": None,
            "error": "No validation data available (missing 'actual_staff_shortage' or 'actual_physicians_needed' column)",
            "n_validations": 0
        }
    
    # Calculate MAE
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate additional metrics
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    # Percentage errors
    percent_errors = [(abs(p - a) / a * 100) if a > 0 else 0 
                      for p, a in zip(predictions, actuals)]
    mean_percent_error = np.mean(percent_errors)
    
    return {
        "mae": round(mae, 2),
        "mean_absolute_error": round(mean_error, 2),
        "max_error": round(max_error, 2),
        "min_error": round(min_error, 2),
        "mean_percent_error": round(mean_percent_error, 2),
        "n_validations": len(predictions),
        "predictions": predictions,
        "actuals": actuals,
        "errors": [round(e, 2) for e in errors]
    }


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TGH PREDICTIVE STAFFING ENGINE")
    print("="*70)
    
    # Example 1: Short closure (12 hours)
    print("\n" + "-"*70)
    print("EXAMPLE 1: Short Closure (12 hours, January - Peak Season)")
    print("-"*70)
    
    closure_start_1 = datetime(2024, 1, 15, 8, 0)  # Jan 15, 8 AM
    duration_1 = 12.0
    
    result_1 = calculate_bridge_event_staffing(
        predicted_closure_start=closure_start_1,
        predicted_duration_hours=duration_1,
        current_actual_census=None
    )
    
    print(json.dumps(result_1, indent=2))
    
    # Example 2: Long closure (48 hours)
    print("\n" + "-"*70)
    print("EXAMPLE 2: Long Closure (48 hours, June - Hurricane Season)")
    print("-"*70)
    
    closure_start_2 = datetime(2024, 6, 15, 14, 0)  # Jun 15, 2 PM
    duration_2 = 48.0
    
    result_2 = calculate_bridge_event_staffing(
        predicted_closure_start=closure_start_2,
        predicted_duration_hours=duration_2,
        current_actual_census=1200  # Higher than ADC
    )
    
    print(json.dumps(result_2, indent=2))
    
    # Example 3: Very long closure (72 hours)
    print("\n" + "-"*70)
    print("EXAMPLE 3: Very Long Closure (72 hours, September - Peak Hurricane)")
    print("-"*70)
    
    closure_start_3 = datetime(2024, 9, 10, 18, 0)  # Sep 10, 6 PM
    duration_3 = 72.0
    
    result_3 = calculate_bridge_event_staffing(
        predicted_closure_start=closure_start_3,
        predicted_duration_hours=duration_3,
        current_actual_census=None
    )
    
    print(json.dumps(result_3, indent=2))
    
    # Display constants
    print("\n" + "="*70)
    print("CORE CONSTANTS")
    print("="*70)
    print(f"Average Daily Census: {TGHConstants.AVERAGE_DAILY_CENSUS}")
    print(f"Licensed Beds: {TGHConstants.LICENSED_BEDS}")
    print(f"Annual ER Visits: {TGHConstants.ANNUAL_ER_VISITS:,}")
    print(f"Daily ER Visits (avg): {TGHConstants.DAILY_ER_VISITS_AVG:.1f}")
    print(f"Inpatient Crisis Ratio: 1 MD per {TGHConstants.INPATIENT_CRISIS_RATIO} patients")
    print(f"ER Ratio: 1 MD per {TGHConstants.ER_MD_PATIENT_RATIO} patients/hour")
    print(f"Specialty Skeleton Crew: {TGHConstants.SPECIALTY_SKELETON_CREW} MDs per shift")
    print(f"Shift Duration: {TGHConstants.SHIFT_DURATION_HOURS} hours")
    
    print("\n" + "="*70)
    print("TEMPORAL ADJUSTMENTS")
    print("="*70)
    temporal = TemporalAdjustment()
    print("\nMonthly Census Multipliers:")
    for month in range(1, 13):
        month_name = datetime(2024, month, 1).strftime('%B')
        mult = temporal.get_monthly_census_multiplier(month)
        print(f"  {month_name:12s}: {mult:.2f}x")
    
    print("\nDiurnal ER Multipliers:")
    print(f"  Peak (10:00-22:00): {temporal.diurnal_er_multipliers['peak']:.1f}x")
    print(f"  Off-Peak (22:00-10:00): {temporal.diurnal_er_multipliers['off_peak']:.1f}x")
    
    print("\nStorm Surge ER Multipliers (Jun-Sep):")
    for month in range(6, 10):
        month_name = datetime(2024, month, 1).strftime('%B')
        mult = temporal.get_storm_surge_er_multiplier(month)
        print(f"  {month_name:12s}: {mult:.1f}x (+20% surge for storm injuries)")
    
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    print("\nNote: To run LOOCV validation, provide historical data with:")
    print("  - closure_hours: Predicted/actual closure duration")
    print("  - date: Closure start datetime")
    print("  - actual_staff_shortage or actual_physicians_needed: Actual staffing needs")
    print("\nExample:")
    print("  validation_data = pd.DataFrame({")
    print("      'date': [datetime(2023, 9, 10, 18, 0)],")
    print("      'closure_hours': [48.0],")
    print("      'actual_physicians_needed': [150]")
    print("  })")
    print("  results = validate_staffing_predictions_loocv(validation_data)")
    print("  print(f'MAE: {results[\"mae\"]} physicians')")

