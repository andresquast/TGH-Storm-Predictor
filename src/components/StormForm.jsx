import React, { useState } from 'react'
import './StormForm.css'

const MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

const WIND_RANGES = {
    0: 30,
    1: 85,
    2: 100,
    3: 120,
    4: 145,
    5: 165
}

function StormForm({ onSubmit, loading }) {
    const [formData, setFormData] = useState({
        category: '0',
        max_wind: '',
        storm_surge: '',
        track_distance: '',
        forward_speed: '',
        month: new Date().getMonth() + 1
    })

    const handleChange = (e) => {
        const { name, value } = e.target
        setFormData(prev => ({
            ...prev,
            [name]: value
        }))

        if (name === 'category' && !formData.max_wind) {
            const suggestedWind = WIND_RANGES[parseInt(value)]
            if (suggestedWind) {
                setFormData(prev => ({
                    ...prev,
                    max_wind: suggestedWind
                }))
            }
        }
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        onSubmit({
            category: parseFloat(formData.category),
            max_wind: parseFloat(formData.max_wind),
            storm_surge: parseFloat(formData.storm_surge),
            track_distance: parseFloat(formData.track_distance),
            forward_speed: parseFloat(formData.forward_speed),
            month: parseFloat(formData.month)
        })
    }

    return (
        <form className="storm-form" onSubmit={handleSubmit}>
            <div className="form-grid">
                <div className="form-group">
                    <label htmlFor="category">Category</label>
                    <select
                        id="category"
                        name="category"
                        value={formData.category}
                        onChange={handleChange}
                        required
                    >
                        <option value="0">Tropical Depression</option>
                        <option value="1">Category 1</option>
                        <option value="2">Category 2</option>
                        <option value="3">Category 3</option>
                        <option value="4">Category 4</option>
                        <option value="5">Category 5</option>
                    </select>
                    <small>Saffir-Simpson scale</small>
                </div>

                <div className="form-group">
                    <label htmlFor="max_wind">Max Wind (mph)</label>
                    <input
                        type="number"
                        id="max_wind"
                        name="max_wind"
                        value={formData.max_wind}
                        onChange={handleChange}
                        min="0"
                        max="200"
                        step="1"
                        required
                    />
                    <small>Sustained winds at Tampa Bay</small>
                </div>

                <div className="form-group">
                    <label htmlFor="storm_surge">Storm Surge (ft)</label>
                    <input
                        type="number"
                        id="storm_surge"
                        name="storm_surge"
                        value={formData.storm_surge}
                        onChange={handleChange}
                        min="0"
                        max="20"
                        step="0.1"
                        required
                    />
                    <small>Expected surge at Tampa Bay</small>
                </div>

                <div className="form-group">
                    <label htmlFor="track_distance">Track Distance (mi)</label>
                    <input
                        type="number"
                        id="track_distance"
                        name="track_distance"
                        value={formData.track_distance}
                        onChange={handleChange}
                        min="0"
                        max="200"
                        step="1"
                        required
                    />
                    <small>Distance from TGH</small>
                </div>

                <div className="form-group">
                    <label htmlFor="forward_speed">Forward Speed (mph)</label>
                    <input
                        type="number"
                        id="forward_speed"
                        name="forward_speed"
                        value={formData.forward_speed}
                        onChange={handleChange}
                        min="0"
                        max="30"
                        step="0.1"
                        required
                    />
                    <small>Storm forward motion</small>
                </div>

                <div className="form-group">
                    <label htmlFor="month">Month</label>
                    <select
                        id="month"
                        name="month"
                        value={formData.month}
                        onChange={handleChange}
                        required
                    >
                        {MONTH_NAMES.slice(1).map((month, index) => (
                            <option key={index + 1} value={index + 1}>
                                {month}
                            </option>
                        ))}
                    </select>
                    <small>Expected month</small>
                </div>
            </div>

            <button
                type="submit"
                className="btn-primary"
                disabled={loading}
            >
                {loading ? 'Predicting...' : 'Predict Closure Duration'}
            </button>
        </form>
    )
}

export default StormForm

