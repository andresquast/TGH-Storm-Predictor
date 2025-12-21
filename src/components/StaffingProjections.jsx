import React, { useState, useEffect } from "react";
import { API_BASE_URL } from "../config";
import { calculateStaffing, loadModelData } from "../prediction";
import "./StaffingProjections.css";

function StaffingProjections({ predictionData }) {
  const [staffing, setStaffing] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const [mae, setMae] = useState(null);
  const [useClientSide, setUseClientSide] = useState(false);

  // Check if model is available
  useEffect(() => {
    loadModelData()
      .then((success) => {
        setUseClientSide(success);
      })
      .catch(() => {
        setUseClientSide(false);
      });
  }, []);

  useEffect(() => {
    if (!predictionData || !predictionData.prediction_hours) {
      return;
    }

    const calculateStaffingRequirements = async () => {
      setLoading(true);
      setError(null);

      try {
        let data;

        if (useClientSide) {
          // Use client-side calculation
          data = calculateStaffing({
            predicted_duration_hours: predictionData.prediction_hours,
            month: predictionData.features.month,
            closure_start: new Date(),
          });
        } else {
          // Fall back to API
          const response = await fetch(
            `${API_BASE_URL}/api/staffing-projections`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                predicted_duration_hours: predictionData.prediction_hours,
                month: predictionData.features.month,
                closure_start: new Date().toISOString(),
              }),
            }
          );

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(
              errorData.error || "Failed to calculate staffing projections"
            );
          }

          data = await response.json();
        }

        setStaffing(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    calculateStaffingRequirements();
  }, [predictionData, useClientSide]);

  useEffect(() => {
    // Fetch MAE from model stats API if not using client-side
    if (!useClientSide) {
      fetch(`${API_BASE_URL}/api/model-stats`)
        .then((res) => res.json())
        .then((data) => {
          if (data.model_fit && data.model_fit.mae) {
            setMae(data.model_fit.mae);
          }
        })
        .catch((err) => {
          console.error("Failed to fetch MAE:", err);
          // Fallback to 0.9 if API fails
          setMae(0.9);
        });
    } else {
      // Use default MAE for client-side mode
      setMae(0.9);
    }
  }, [useClientSide]);

  if (loading) {
    return (
      <div className="staffing-container">
        <div className="staffing-card">
          <h3>Projected Staffing Requirements</h3>
          <p style={{ color: "#b0b0b0" }}>
            Calculating staffing requirements...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="staffing-container">
        <div className="staffing-card">
          <h3>Projected Staffing Requirements</h3>
          <p className="error-text">Error: {error}</p>
        </div>
      </div>
    );
  }

  if (!staffing) {
    return null;
  }

  const getRiskLevel = (score) => {
    if (score <= 3) return { level: "Low", color: "low-risk" };
    if (score <= 6) return { level: "Medium", color: "medium-risk" };
    if (score <= 8) return { level: "High", color: "high-risk" };
    return { level: "Critical", color: "critical-risk" };
  };

  const riskInfo = getRiskLevel(staffing.risk_score);

  // Determine if this is a short closure (single shift)
  const isShortClosure =
    staffing.details && staffing.details.predicted_duration_hours <= 12;
  const isSameTeam = staffing.team_a_count === staffing.team_b_preposition;

  return (
    <div className="staffing-container">
      <div className="staffing-card">
        <div className="staffing-header">
          <h3>Projected Staffing Requirements</h3>
          <button
            className="details-link"
            onClick={() => setShowDetails(!showDetails)}
            aria-expanded={showDetails}
            title="View staffing decision logic"
          >
            {showDetails ? "Hide Details" : "Details"}
          </button>
        </div>

        {showDetails && staffing.details && (
          <div className="details-popup">
            <div className="explanation-section">
              <h5>Team Assignment Logic</h5>
              <ul>
                <li>
                  <strong>
                    ≤12 hours (with ±{mae !== null ? mae.toFixed(1) : "..."}{" "}
                    hour MAE buffer):
                  </strong>{" "}
                  Single team only. Team A = Team B. All physicians stay for
                  full duration.
                </li>
                <li>
                  <strong>12-48 hours:</strong> Two rotating teams. Each team
                  works 12 hours, then rests 12 hours.
                </li>
                <li>
                  <strong>&gt;48 hours:</strong> Three rotating teams. Each team
                  works 12 hours, then rests 24 hours.
                </li>
              </ul>
            </div>

            <div className="explanation-section">
              <h5>Risk Score</h5>
              <p>
                The risk score (1-10) is calculated based on multiple factors:
              </p>
              <ul>
                <li>
                  <strong>Duration:</strong> Longer closures increase risk
                  (≤12h: +1.0, 12-24h: +2.5, 24-48h: +4.0, &gt;48h: +5.5)
                </li>
                <li>
                  <strong>Census Level:</strong> Higher hospital occupancy
                  increases risk (&lt;70%: +0.5, 70-85%: +1.0, 85-95%: +2.0,
                  ≥95%: +3.0)
                </li>
                <li>
                  <strong>Time of Day:</strong> Peak hours (10:00-22:00) add
                  +0.5
                </li>
                <li>
                  <strong>Season:</strong> Peak season months (Jan-Mar with
                  higher census multipliers) add +0.5
                </li>
              </ul>
            </div>

            <div className="explanation-section">
              <h5>Staffing Components</h5>
              <ul>
                <li>
                  <strong>{staffing.details.inpatient_mds} MDs</strong> were
                  used to serve the purpose of inpatient care (1 MD per 18
                  patients, configurable via <code>INPATIENT_CRISIS_RATIO</code>
                  )
                </li>
                <li>
                  <strong>{staffing.details.er_mds} MDs</strong> were used to
                  serve the purpose of emergency/trauma care (1 MD per 2.5
                  patients/hour, configurable via{" "}
                  <code>ER_MD_PATIENT_RATIO</code>)
                </li>
                <li>
                  <strong>
                    {staffing.details.specialty_skeleton_crew} MDs
                  </strong>{" "}
                  were used to serve the purpose of specialty coverage
                  (Transplant, Surgery, Anesthesia - configurable via{" "}
                  <code>SPECIALTY_SKELETON_CREW</code>)
                </li>
                <li>
                  Based on adjusted census of{" "}
                  <strong>
                    {Math.round(staffing.details.adjusted_census)}
                  </strong>{" "}
                  patients (monthly multiplier:{" "}
                  {staffing.details.monthly_multiplier.toFixed(2)}x)
                </li>
              </ul>
            </div>
          </div>
        )}

        <div className="staffing-summary">
          <div className="summary-item primary">
            <span className="summary-label">Total Physicians Needed</span>
            <span className="summary-value">
              {staffing.total_physicians_needed}
            </span>
          </div>

          <div className="summary-item">
            <span className="summary-label">Risk Score</span>
            <span className={`risk-badge ${riskInfo.color}`}>
              {staffing.risk_score.toFixed(1)} ({riskInfo.level})
            </span>
          </div>
        </div>

        <div className="staffing-details">
          <h4>Team Breakdown</h4>
          <div className="team-grid">
            {isSameTeam ? (
              <>
                <div className="team-item">
                  <span className="team-label">Required Physicians</span>
                  <span className="team-value">
                    {staffing.team_a_count} physicians
                  </span>
                  <span className="team-description">
                    Must arrive on Davis Island before bridges close and remain
                    for duration of closure. For closures ≤12 hours (with ±
                    {mae !== null ? mae.toFixed(1) : "..."} hour MAE buffer),
                    Team A = Team B - only one team is needed as all physicians
                    stay for the full duration.
                  </span>
                </div>
              </>
            ) : (
              <>
                <div className="team-item">
                  <span className="team-label">Team A (Island Isolation)</span>
                  <span className="team-value">
                    {staffing.team_a_count} physicians
                  </span>
                  <span className="team-description">
                    Required to stay on Davis Island during closure. Team A
                    physicians must be pre-positioned on the island before
                    bridges close.
                  </span>
                </div>

                <div className="team-item">
                  <span className="team-label">Team B (Pre-positioning)</span>
                  <span className="team-value">
                    {staffing.team_b_preposition} physicians
                  </span>
                  <span className="team-description">
                    Must cross bridges before closure to rotate with Team A. For
                    closures &gt;12 hours, Team B represents additional
                    physicians needed for shift rotations to provide continuous
                    coverage throughout the closure period.
                  </span>
                </div>
              </>
            )}
          </div>
        </div>

        {staffing.details && (
          <div className="staffing-breakdown">
            <h4>Detailed Breakdown</h4>
            <div className="breakdown-grid">
              <div className="breakdown-item">
                <span className="breakdown-label">Inpatient MDs:</span>
                <span className="breakdown-value">
                  {staffing.details.inpatient_mds}
                </span>
              </div>
              <div className="breakdown-item">
                <span className="breakdown-label">ER MDs:</span>
                <span className="breakdown-value">
                  {staffing.details.er_mds}
                </span>
              </div>
              <div className="breakdown-item">
                <span className="breakdown-label">Specialty Crew:</span>
                <span className="breakdown-value">
                  {staffing.details.specialty_skeleton_crew}
                </span>
              </div>
              <div className="breakdown-item">
                <span className="breakdown-label">Shifts Needed:</span>
                <span className="breakdown-value">
                  {staffing.details.shifts_needed}
                </span>
              </div>
              <div className="breakdown-item">
                <span className="breakdown-label">Expected Patients:</span>
                <span className="breakdown-value">
                  {Math.round(staffing.details.adjusted_census)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StaffingProjections;
