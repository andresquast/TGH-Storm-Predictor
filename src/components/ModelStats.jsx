import React, { useState, useEffect } from "react";
import { API_BASE_URL } from "../config";
import "./ModelStats.css";

function ModelStats() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/model-stats`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        const contentType = res.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
          throw new Error(
            "Response is not JSON. Make sure Flask server is running on port 5001."
          );
        }
        return res.json();
      })
      .then((data) => {
        if (data.error) {
          setError(data.error);
        } else {
          setStats(data);
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(
          err.message ||
            "Failed to load model statistics. Make sure Flask server is running."
        );
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="stats-container">
        <div className="stats-card">
          <p>Loading model statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="stats-container">
        <div className="stats-card">
          <p className="error-text">Error loading stats: {error}</p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return null;
  }

  return (
    <div className="stats-container">
      <div className="stats-card">
        <h2>Model Statistics</h2>

        <div className="stats-section">
          <h3>Model Fit</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">
                R² (Coefficient of Determination):
              </span>
              <span className="stat-value">
                {stats.model_fit.r2.toFixed(4)}
              </span>
              <span className="stat-description">
                {stats.model_fit.r2.toFixed(1)}% of variance explained
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Adjusted R²:</span>
              <span className="stat-value">
                {stats.model_fit.adj_r2.toFixed(4)}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">RMSE:</span>
              <span className="stat-value">
                {stats.model_fit.rmse.toFixed(2)} hours
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">MAE:</span>
              <span className="stat-value">
                {stats.model_fit.mae.toFixed(2)} hours
              </span>
            </div>
          </div>
        </div>

        <div className="stats-section">
          <h3>Coefficient Statistics</h3>
          <div className="coefficients-table">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Coefficient</th>
                  <th>Std Error</th>
                  <th>p-value</th>
                  <th>Significance</th>
                </tr>
              </thead>
              <tbody>
                {stats.coefficients.map((coef, idx) => {
                  return (
                    <tr key={idx}>
                      <td>{coef.feature}</td>
                      <td>{coef.coefficient.toFixed(3)}</td>
                      <td>{coef.std_error.toFixed(3)}</td>
                      <td>{coef.p_value.toFixed(4)}</td>
                      <td className="significance-cell">{coef.significance}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="significance-note">
              Significance codes: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, .
              p&lt;0.1
            </p>
          </div>
        </div>

        <div className="stats-section">
          <h3>Model Significance</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">F-statistic:</span>
              <span className="stat-value">
                {stats.model_significance.f_statistic.toFixed(4)}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">F-statistic p-value:</span>
              <span className="stat-value">
                {stats.model_significance.f_pvalue.toFixed(6)}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Model Status:</span>
              <span
                className={`stat-value ${
                  stats.model_significance.is_significant
                    ? "significant"
                    : "not-significant"
                }`}
              >
                {stats.model_significance.is_significant
                  ? "✓ Statistically Significant"
                  : "✗ Not Significant"}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Durbin-Watson:</span>
              <span className="stat-value">
                {stats.durbin_watson.toFixed(4)}
              </span>
              <span className="stat-description">
                (Values close to 2 indicate no autocorrelation)
              </span>
            </div>
          </div>
        </div>

        {stats.normality_test && (
          <div className="stats-section">
            <h3>Normality Test</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Shapiro-Wilk Statistic:</span>
                <span className="stat-value">
                  {stats.normality_test.statistic.toFixed(4)}
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">p-value:</span>
                <span className="stat-value">
                  {stats.normality_test.p_value.toFixed(4)}
                </span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Normally Distributed:</span>
                <span
                  className={`stat-value ${
                    stats.normality_test.is_normal
                      ? "significant"
                      : "not-significant"
                  }`}
                >
                  {stats.normality_test.is_normal ? "✓ Yes" : "✗ No"}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelStats;
