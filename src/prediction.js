let modelData = null;
let seasonalCensus = null;
let staffingConstants = null;

export async function loadModelData() {
  try {
    const basePath = window.location.pathname.includes("/TGH-Storm-Predictor/")
      ? "/TGH-Storm-Predictor"
      : "";

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
      throw new Error(`Failed to load from all paths: ${paths.join(", ")}`);
    };

    modelData = await fetchWithFallback([
      `${basePath}/model.json`,
      "/model.json",
      "./model.json",
    ]);

    seasonalCensus = await fetchWithFallback([
      `${basePath}/seasonal_census.json`,
      "/seasonal_census.json",
      "./seasonal_census.json",
    ]);

    staffingConstants = await fetchWithFallback([
      `${basePath}/staffing_constants.json`,
      "/staffing_constants.json",
      "./staffing_constants.json",
    ]);

    return true;
  } catch (error) {
    return false;
  }
}

export function predictClosureDuration(features) {
  if (!modelData) {
    throw new Error("Model data not loaded. Call loadModelData() first.");
  }

  const { coefficients, intercept } = modelData;

  const featureValues = [
    features.max_wind || 0,
    features.storm_surge || 0,
    features.track_distance || 0,
    features.forward_speed || 0,
    features.month || 1,
  ];

  let prediction = intercept;
  for (let i = 0; i < coefficients.length; i++) {
    prediction += coefficients[i] * featureValues[i];
  }

  prediction = Math.max(0, prediction);

  const mse = modelData.mse || 10;
  const stdError = Math.sqrt(mse);
  const ciMargin = 1.96 * stdError;

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
      month: featureValues[4],
    },
  };
}

export function calculateStaffing(params) {
  if (!staffingConstants) {
    throw new Error(
      "Staffing constants not loaded. Call loadModelData() first."
    );
  }

  const { constants, temporal } = staffingConstants;
  const { predicted_duration_hours, month, closure_start, current_census } =
    params;

  const closureDate =
    closure_start instanceof Date ? closure_start : new Date(closure_start);

  const hour = closureDate.getHours();
  const baseCensus = current_census || constants.average_daily_census;
  const monthlyMultiplier =
    temporal.monthly_census_multipliers[month.toString()] || 1.0;
  const adjustedCensus = baseCensus * monthlyMultiplier;
  const inpatientMDs = Math.ceil(
    adjustedCensus / constants.inpatient_crisis_ratio
  );

  const baseERVisitsPerHour = constants.daily_er_visits_avg / 24;
  const isPeak = hour >= 10 && hour < 22;
  const diurnalMultiplier = isPeak
    ? temporal.diurnal_er_multipliers.peak
    : temporal.diurnal_er_multipliers.off_peak;

  const stormSurgeMultiplier = temporal.storm_surge_months.includes(month)
    ? temporal.storm_surge_multiplier
    : 1.0;

  const avgERVisitsPerHour =
    baseERVisitsPerHour *
    ((temporal.diurnal_er_multipliers.peak +
      temporal.diurnal_er_multipliers.off_peak) /
      2) *
    stormSurgeMultiplier;

  const erMDs = Math.ceil(avgERVisitsPerHour / constants.er_md_patient_ratio);
  const shiftsNeeded = Math.ceil(
    predicted_duration_hours / constants.shift_duration_hours
  );
  const mdsPerShift = inpatientMDs + erMDs + constants.specialty_skeleton_crew;

  let teamACount;
  if (predicted_duration_hours <= constants.shift_duration_hours) {
    teamACount = mdsPerShift;
  } else if (predicted_duration_hours <= 48) {
    teamACount = mdsPerShift * 2;
  } else {
    teamACount = mdsPerShift * 3;
  }

  const riskScore = calculateRiskScore(
    predicted_duration_hours,
    adjustedCensus,
    hour,
    month,
    constants.licensed_beds
  );

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
      er_surge_multiplier: stormSurgeMultiplier,
    },
  };
}

function calculateRiskScore(durationHours, census, hour, month, licensedBeds) {
  let risk = 1.0;

  if (durationHours <= 12) {
    risk += 1.0;
  } else if (durationHours <= 24) {
    risk += 2.5;
  } else if (durationHours <= 48) {
    risk += 4.0;
  } else {
    risk += 5.5;
  }

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

  if (hour >= 10 && hour < 22) {
    risk += 0.5;
  }

  if (month >= 1 && month <= 3) {
    risk += 0.5;
  }

  return Math.min(risk, 10.0);
}

export function getSeasonalCensus(month) {
  if (!seasonalCensus) {
    throw new Error("Seasonal census not loaded. Call loadModelData() first.");
  }
  return seasonalCensus[month] || 450;
}
