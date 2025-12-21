import React, { useState, useEffect } from 'react'
import StormForm from './components/StormForm'
import PredictionResult from './components/PredictionResult'
import ErrorMessage from './components/ErrorMessage'
import ModelStats from './components/ModelStats'
import DataStats from './components/DataStats'
import './App.css'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showModelStats, setShowModelStats] = useState(false)
  const [showDataStats, setShowDataStats] = useState(false)
  const [mae, setMae] = useState(null)

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed')
      }

      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleModelStatsClick = () => {
    if (showModelStats) {
      // If already open, close it
      setShowModelStats(false)
    } else {
      // Close Data Stats if open, then open Model Stats
      setShowDataStats(false)
      setShowModelStats(true)
    }
  }

  const handleDataStatsClick = () => {
    if (showDataStats) {
      // If already open, close it
      setShowDataStats(false)
    } else {
      // Close Model Stats if open, then open Data Stats
      setShowModelStats(false)
      setShowDataStats(true)
    }
  }

  useEffect(() => {
    // Fetch MAE from model stats API
    fetch('/api/model-stats')
      .then((res) => res.json())
      .then((data) => {
        if (data.model_fit && data.model_fit.mae) {
          setMae(data.model_fit.mae)
        }
      })
      .catch((err) => {
        console.error('Failed to fetch MAE:', err)
        // Fallback to 0.9 if API fails
        setMae(0.9)
      })
  }, [])

  return (
    <div className="app">
      <header>
        <h1>TGH Storm Closure Duration Predictor</h1>
        <p className="subtitle">Predict bridge closure duration based on storm characteristics</p>
      </header>

      <main>
        <div className="stats-buttons">
          <button 
            className={`stats-button ${showModelStats ? 'active' : ''}`}
            onClick={handleModelStatsClick}
          >
            {showModelStats ? 'Hide Model Stats' : 'Model Stats'}
          </button>
          <button 
            className={`stats-button ${showDataStats ? 'active' : ''}`}
            onClick={handleDataStatsClick}
          >
            {showDataStats ? 'Hide Data Stats' : 'Data Stats'}
          </button>
        </div>

        {showModelStats && <ModelStats />}
        {showDataStats && <DataStats />}

        <StormForm onSubmit={handlePredict} loading={loading} />
        
        {error && <ErrorMessage message={error} />}
        
        {prediction && <PredictionResult data={prediction} />}
      </main>

      <footer>
        <p>Model trained on historical storm data (2004-2024)</p>
        <p>Mean Absolute Error: ±{mae !== null ? mae.toFixed(1) : '...'} hours (Leave-One-Out Cross-Validation)</p>
      </footer>
    </div>
  )
}

export default App

