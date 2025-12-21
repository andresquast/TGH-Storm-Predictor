/**
 * Client-side prediction module for TGH Storm Predictor
 * Runs entirely in the browser without requiring a backend
 */

// Model data will be loaded from JSON files
let modelData = null;
let seasonalCensus = null;
let staffingConstants = null;

/**
 * Load model data from JSON files
 * Handles both GitHub Pages (with base path) and local development
 */
export async function loadModelData() {
  try {
    // Determine base path - check if we're on GitHub Pages
    const basePath = window.location.pathname.includes('/TGH-Storm-Predictor/') 
      ? '/TGH-Storm-Predictor' 
      : '';

    // Helper function to try multiple paths
    const fetchWithFallback = async (paths) => {
      for (const path of paths) {
        try {
          const response = await fetch(path);
          if (response.ok) {
            return await response.json();
          }
        } catch (e) {
          // Try next path
        }
      }
      throw new Error(`Failed to load from all paths: ${paths.join(', ')}`);
    };

    // Load model parameters
    modelData = await fetchWithFallback([
      `${basePath}/model.json`,
      '/model.json',
      './model.json'
    ]);

    // Load seasonal census
    seasonalCensus = await fetchWithFallback([
      `${basePath}/seasonal_census.json`,
      '/seasonal_census.json',
      './seasonal_census.json'
    ]);

    // Load staffing constants
    staffingConstants = await fetchWithFallback([
      `${basePath}/staffing_constants.json`,
      '/staffing_constants.json',
      './staffing_constants.json'
    ]);

    return true;
  } catch (error) {
    console.error('Error loading model data:', error);
    return false;
  }
}

/**
 * Predict bridge closure duration using linear regression
 * @param {Object} features - Storm features
 * @param {number} features.max_wind - Maximum wind speed (mph)
 * @param {number} features.storm_surge - Storm surge (feet)
 * @param {number} features.track_distance - Track distance from Tampa (miles)
 * @param {number} features.forward_speed - Forward speed (mph)
 * @param {number} features.month - Month (1-12)
 * @returns {Object} Prediction result with hours and confidence interval
 */
export function predictClosureDuration(features) {
  if (!modelData) {
    throw new Error('Model data not loaded. Call loadModelData() first.');
  }

  const { coefficients, intercept, features: featureNames } = modelData;

  // Ensure features are in the correct order
  const featureValues = [
    features.max_wind || 0,
    features.storm_surge || 0,
    features.track_distance || 0,
    features.forward_speed || 0,
    features.month || 1
  ];

  // Calculate prediction: intercept + sum(coef_i * feature_i)
  let prediction = intercept;
  for (let i = 0; i < coefficients.length; i++) {
    prediction += coefficients[i] * featureValues[i];
  }

  // Ensure non-negative
  prediction = Math.max(0, prediction);

  // Calculate approximate confidence interval using MSE
  // This is a simplified version - for exact intervals we'd need the full statsmodels output
  const mse = modelData.mse || 10; // Default MSE if not available
  const stdError = Math.sqrt(mse);
  const ciMargin = 1.96 * stdError; // 95% confidence interval

  const ciLower = Math.max(0, prediction - ciMargin);
  const ciUpper = prediction + ciMargin;

  return {
    prediction_hours: Math.round(prediction * 10) / 10, // Round to 1 decimal
    ci_lower: Math.round(ciLower * 10) / 10,
    ci_upper: Math.round(ciUpper * 10) / 10,
    features: {
      max_wind: featureValues[0],
      storm_surge: featureValues[1],
      track_distance: featureValues[2],
      forward_speed: featureValues[3],
      month: featureValues[4]
    }
  };
}

/**
 * Calculate staffing requirements for bridge closure event
 * @param {Object} params - Staffing calculation parameters
 * @param {number} params.predicted_duration_hours - Predicted closure duration
 * @param {number} params.month - Month (1-12)
 * @param {Date|string} params.closure_start - Closure start datetime
 * @param {number} [params.current_census] - Current census (optional)
 * @returns {Object} Staffing requirements
 */
export function calculateStaffing(params) {
  if (!staffingConstants) {
    throw new Error('Staffing constants not loaded. Call loadModelData() first.');
  }

  const { constants, temporal } = staffingConstants;
  const { predicted_duration_hours, month, closure_start, current_census } = params;

  // Parse closure start date
  const closureDate = closure_start instanceof Date 
    ? closure_start 
    : new Date(closure_start);
  
  const hour = closureDate.getHours();

  // Get base census
  const baseCensus = current_census || constants.average_daily_census;

  // Apply monthly adjustment
  const monthlyMultiplier = temporal.monthly_census_multipliers[month.toString()] || 1.0;
  const adjustedCensus = baseCensus * monthlyMultiplier;

  // Calculate inpatient MDs
  const inpatientMDs = Math.ceil(adjustedCensus / constants.inpatient_crisis_ratio);

  // Calculate ER MDs
  const baseERVisitsPerHour = constants.daily_er_visits_avg / 24;
  
  // Get diurnal multiplier
  const isPeak = hour >= 10 && hour < 22;
  const diurnalMultiplier = isPeak 
    ? temporal.diurnal_er_multipliers.peak 
    : temporal.diurnal_er_multipliers.off_peak;
  
  // Get storm surge multiplier (Jun-Sep)
  const stormSurgeMultiplier = temporal.storm_surge_months.includes(month)
    ? temporal.storm_surge_multiplier
    : 1.0;
  
  // Average of peak and off-peak for duration
  const avgERVisitsPerHour = baseERVisitsPerHour * 
    ((temporal.diurnal_er_multipliers.peak + temporal.diurnal_er_multipliers.off_peak) / 2) *
    stormSurgeMultiplier;
  
  const erMDs = Math.ceil(avgERVisitsPerHour / constants.er_md_patient_ratio);

  // Calculate shifts needed
  const shiftsNeeded = Math.ceil(predicted_duration_hours / constants.shift_duration_hours);

  // Calculate MDs per shift
  const mdsPerShift = inpatientMDs + erMDs + constants.specialty_skeleton_crew;

  // Calculate team sizes based on duration
  let teamACount;
  if (predicted_duration_hours <= constants.shift_duration_hours) {
    // Single shift
    teamACount = mdsPerShift;
  } else if (predicted_duration_hours <= 48) {
    // Two teams rotating
    teamACount = mdsPerShift * 2;
  } else {
    // Extended closure: three teams
    teamACount = mdsPerShift * 3;
  }

  // Calculate risk score
  const riskScore = calculateRiskScore(
    predicted_duration_hours,
    adjustedCensus,
    hour,
    month,
    constants.licensed_beds
  );

  // Calculate capacity utilization
  const capacityUtilization = (adjustedCensus / constants.licensed_beds) * 100;

  return {
    total_physicians_needed: teamACount,
    team_a_count: teamACount,
    team_b_preposition: teamACount,
    risk_score: Math.round(riskScore * 10) / 10,
    capacity_utilization: Math.round(capacityUtilization * 10) / 10,
    details: {
      predicted_closure_start: closureDate.toISOString(),
      predicted_duration_hours: Math.round(predicted_duration_hours * 10) / 10,
      adjusted_census: Math.round(adjustedCensus),
      inpatient_mds: inpatientMDs,
      er_mds: erMDs,
      specialty_skeleton_crew: constants.specialty_skeleton_crew,
      shifts_needed: shiftsNeeded,
      month: month,
      monthly_multiplier: monthlyMultiplier,
      er_surge_multiplier: stormSurgeMultiplier
    }
  };
}

/**
 * Calculate risk score (1-10) based on multiple factors
 */
function calculateRiskScore(durationHours, census, hour, month, licensedBeds) {
  let risk = 1.0;

  // Duration factor
  if (durationHours <= 12) {
    risk += 1.0;
  } else if (durationHours <= 24) {
    risk += 2.5;
  } else if (durationHours <= 48) {
    risk += 4.0;
  } else {
    risk += 5.5;
  }

  // Census factor
  const censusUtilization = census / licensedBeds;
  if (censusUtilization < 0.7) {
    risk += 0.5;
  } else if (censusUtilization < 0.85) {
    risk += 1.0;
  } else if (censusUtilization < 0.95) {
    risk += 2.0;
  } else {
    risk += 3.0;
  }

  // Time of day factor
  if (hour >= 10 && hour < 22) {
    risk += 0.5;
  }

  // Month factor (peak season)
  if (month >= 1 && month <= 3) {
    risk += 0.5; // Peak snowbird/flu season
  }

  return Math.min(risk, 10.0);
}

/**
 * Get seasonal census for a given month
 */
export function getSeasonalCensus(month) {
  if (!seasonalCensus) {
    throw new Error('Seasonal census not loaded. Call loadModelData() first.');
  }
  return seasonalCensus[month] || 450; // Default fallback
}

