import React, { useState, useEffect } from 'react'
import { API_BASE_URL } from '../config'
import './DataStats.css'

const MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

function DataStats() {
    const [stats, setStats] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        const loadStats = async () => {
            // If API_BASE_URL is empty (GitHub Pages), try to load from JSON file
            if (!API_BASE_URL) {
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

                try {
                    const data = await fetchWithFallback([
                        `${basePath}/data_stats.json`,
                        "/data_stats.json",
                        "./data_stats.json",
                    ]);
                    setStats(data);
                    setLoading(false);
                } catch (err) {
                    console.error("Failed to load data_stats.json:", err);
                    setError(
                        `Data statistics not available. Failed to load data_stats.json from: ${basePath || "root"}. Error: ${err.message}`
                    );
                    setLoading(false);
                }
                return;
            }

            // Otherwise, fetch from API
            fetch(`${API_BASE_URL}/api/data-stats`)
                .then(res => {
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`)
                    }
                    const contentType = res.headers.get('content-type')
                    if (!contentType || !contentType.includes('application/json')) {
                        throw new Error('Response is not JSON. Make sure Flask server is running on port 5001.')
                    }
                    return res.json()
                })
                .then(data => {
                    if (data.error) {
                        setError(data.error)
                    } else {
                        setStats(data)
                    }
                    setLoading(false)
                })
                .catch(err => {
                    setError(err.message || 'Failed to load data statistics. Make sure Flask server is running.')
                    setLoading(false)
                })
        };

        loadStats();
    }, [])

    if (loading) {
        return (
            <div className="stats-container">
                <div className="stats-card">
                    <p>Loading data statistics...</p>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="stats-container">
                <div className="stats-card">
                    <p className="error-text">Error loading stats: {error}</p>
                </div>
            </div>
        )
    }

    if (!stats) {
        return null
    }

    const getCorrelationDirection = (value) => {
        if (Math.abs(value) >= 0.5) return 'strong'
        if (Math.abs(value) >= 0.3) return 'moderate'
        return 'weak'
    }

    const getCorrelationArrow = (value) => {
        return value >= 0 ? '↑' : '↓'
    }

    return (
        <div className="stats-container">
            <div className="stats-card">
                <h2>Data Statistics</h2>
                
                <div className="stats-section">
                    <h3>Dataset Overview</h3>
                    <div className="stats-grid">
                        <div className="stat-item">
                            <span className="stat-label">Total Storms:</span>
                            <span className="stat-value">{stats.dataset_overview.total_storms}</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Date Range:</span>
                            <span className="stat-value">
                                {stats.dataset_overview.year_range.min} - {stats.dataset_overview.year_range.max}
                            </span>
                            <span className="stat-description">
                                ({stats.dataset_overview.year_range.span} years)
                            </span>
                        </div>
                    </div>
                </div>

                <div className="stats-section">
                    <h3>Bridge Closure Duration</h3>
                    <div className="stats-grid">
                        <div className="stat-item">
                            <span className="stat-label">Mean:</span>
                            <span className="stat-value">{stats.closure_stats.mean.toFixed(1)} hours</span>
                            <span className="stat-description">
                                ({stats.closure_stats.mean.toFixed(1) / 24} days)
                            </span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Median:</span>
                            <span className="stat-value">{stats.closure_stats.median.toFixed(1)} hours</span>
                            <span className="stat-description">
                                ({stats.closure_stats.median.toFixed(1) / 24} days)
                            </span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Standard Deviation:</span>
                            <span className="stat-value">{stats.closure_stats.std_dev.toFixed(1)} hours</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Minimum:</span>
                            <span className="stat-value">{stats.closure_stats.min.toFixed(0)} hours</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">Maximum:</span>
                            <span className="stat-value">{stats.closure_stats.max.toFixed(0)} hours</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">IQR:</span>
                            <span className="stat-value">{stats.closure_stats.iqr.toFixed(1)} hours</span>
                        </div>
                    </div>
                    
                    <div className="distribution-section">
                        <h4>Closure Duration Distribution</h4>
                        <div className="distribution-grid">
                            <div className="distribution-item">
                                <span className="distribution-label">Short (≤12 hours):</span>
                                <span className="distribution-value">{stats.closure_distribution.short}</span>
                                <span className="distribution-percent">
                                    ({((stats.closure_distribution.short / stats.dataset_overview.total_storms) * 100).toFixed(0)}%)
                                </span>
                            </div>
                            <div className="distribution-item">
                                <span className="distribution-label">Medium (13-24 hours):</span>
                                <span className="distribution-value">{stats.closure_distribution.medium}</span>
                                <span className="distribution-percent">
                                    ({((stats.closure_distribution.medium / stats.dataset_overview.total_storms) * 100).toFixed(0)}%)
                                </span>
                            </div>
                            <div className="distribution-item">
                                <span className="distribution-label">Long (≥25 hours):</span>
                                <span className="distribution-value">{stats.closure_distribution.long}</span>
                                <span className="distribution-percent">
                                    ({((stats.closure_distribution.long / stats.dataset_overview.total_storms) * 100).toFixed(0)}%)
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="stats-section">
                    <h3>Storm Characteristics</h3>
                    
                    <div className="characteristic-group">
                        <h4>Wind Speed (mph)</h4>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-label">Mean:</span>
                                <span className="stat-value">{stats.wind_stats.mean.toFixed(1)} mph</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Median:</span>
                                <span className="stat-value">{stats.wind_stats.median.toFixed(1)} mph</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Range:</span>
                                <span className="stat-value">
                                    {stats.wind_stats.min.toFixed(0)} - {stats.wind_stats.max.toFixed(0)} mph
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Std Dev:</span>
                                <span className="stat-value">{stats.wind_stats.std_dev.toFixed(1)} mph</span>
                            </div>
                        </div>
                    </div>

                    <div className="characteristic-group">
                        <h4>Storm Surge (feet)</h4>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-label">Mean:</span>
                                <span className="stat-value">{stats.surge_stats.mean.toFixed(1)} ft</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Median:</span>
                                <span className="stat-value">{stats.surge_stats.median.toFixed(1)} ft</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Range:</span>
                                <span className="stat-value">
                                    {stats.surge_stats.min.toFixed(1)} - {stats.surge_stats.max.toFixed(1)} ft
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Std Dev:</span>
                                <span className="stat-value">{stats.surge_stats.std_dev.toFixed(1)} ft</span>
                            </div>
                        </div>
                    </div>

                    <div className="characteristic-group">
                        <h4>Track Distance (miles from Tampa)</h4>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-label">Mean:</span>
                                <span className="stat-value">{stats.track_stats.mean.toFixed(1)} miles</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Median:</span>
                                <span className="stat-value">{stats.track_stats.median.toFixed(1)} miles</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Range:</span>
                                <span className="stat-value">
                                    {stats.track_stats.min.toFixed(0)} - {stats.track_stats.max.toFixed(0)} miles
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Std Dev:</span>
                                <span className="stat-value">{stats.track_stats.std_dev.toFixed(1)} miles</span>
                            </div>
                        </div>
                    </div>

                    <div className="characteristic-group">
                        <h4>Forward Speed (mph)</h4>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-label">Mean:</span>
                                <span className="stat-value">{stats.speed_stats.mean.toFixed(1)} mph</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Median:</span>
                                <span className="stat-value">{stats.speed_stats.median.toFixed(1)} mph</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Range:</span>
                                <span className="stat-value">
                                    {stats.speed_stats.min.toFixed(0)} - {stats.speed_stats.max.toFixed(0)} mph
                                </span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Std Dev:</span>
                                <span className="stat-value">{stats.speed_stats.std_dev.toFixed(1)} mph</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="stats-section">
                    <h3>Temporal Distribution</h3>
                    
                    <div className="characteristic-group">
                        <h4>By Month</h4>
                        <div className="distribution-grid">
                            {Object.entries(stats.temporal_distribution.by_month)
                                .sort(([a], [b]) => parseInt(a) - parseInt(b))
                                .map(([month, count]) => (
                                    <div key={month} className="distribution-item">
                                        <span className="distribution-label">{MONTH_NAMES[parseInt(month)]}:</span>
                                        <span className="distribution-value">{count}</span>
                                        <span className="distribution-percent">
                                            ({((count / stats.dataset_overview.total_storms) * 100).toFixed(0)}%)
                                        </span>
                                    </div>
                                ))}
                        </div>
                    </div>

                </div>

                <div className="stats-section">
                    <h3>Correlations with Closure Duration</h3>
                    <div className="correlations-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Correlation</th>
                                    <th>Direction</th>
                                    <th>Strength</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(stats.correlations)
                                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                                    .map(([feature, value]) => (
                                        <tr key={feature}>
                                            <td>{feature.replace('_', ' ')}</td>
                                            <td>{value.toFixed(3)}</td>
                                            <td className={`direction-cell ${value >= 0 ? 'positive' : 'negative'}`}>
                                                {getCorrelationArrow(value)} {value >= 0 ? 'Positive' : 'Negative'}
                                            </td>
                                            <td className={`strength-cell ${getCorrelationDirection(value)}`}>
                                                {getCorrelationDirection(value).charAt(0).toUpperCase() + getCorrelationDirection(value).slice(1)}
                                            </td>
                                        </tr>
                                    ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default DataStats

