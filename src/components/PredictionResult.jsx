import React from 'react'
import StaffingProjections from './StaffingProjections'
import './PredictionResult.css'

const MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

function PredictionResult({ data }) {
    const { prediction_hours, prediction_days, features } = data || {}
    
    // Calculate prediction_days if not provided
    const days = prediction_days !== undefined 
        ? prediction_days 
        : (prediction_hours !== undefined ? Math.round((prediction_hours / 24) * 10) / 10 : 0)

    if (!data || prediction_hours === undefined) {
        return null
    }

    return (
        <div className="result-container">
            <div className="result-card">
                <h2>Predicted Closure Duration</h2>
                <div className="prediction-display">
                    <span className="prediction-value">{prediction_hours.toFixed(1)}</span>
                    <span className="prediction-unit">hours</span>
                </div>
                <div className="prediction-days">
                    (approximately {days.toFixed(1)} days)
                </div>
                
                <div className="feature-summary">
                    <h3>Input Summary</h3>
                    <div className="feature-grid">
                        <div className="feature-item">
                            <span className="feature-label">Max Wind:</span>
                            <span className="feature-value">{features.max_wind} mph</span>
                        </div>
                        <div className="feature-item">
                            <span className="feature-label">Storm Surge:</span>
                            <span className="feature-value">{features.storm_surge} ft</span>
                        </div>
                        <div className="feature-item">
                            <span className="feature-label">Track Distance:</span>
                            <span className="feature-value">{features.track_distance} miles</span>
                        </div>
                        <div className="feature-item">
                            <span className="feature-label">Forward Speed:</span>
                            <span className="feature-value">{features.forward_speed} mph</span>
                        </div>
                        <div className="feature-item">
                            <span className="feature-label">Month:</span>
                            <span className="feature-value">{MONTH_NAMES[features.month]}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <StaffingProjections predictionData={data} />
        </div>
    )
}

export default PredictionResult

