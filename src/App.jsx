import React, { useState } from 'react'
import StormForm from './components/StormForm'
import PredictionResult from './components/PredictionResult'
import ErrorMessage from './components/ErrorMessage'
import './App.css'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

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

  return (
    <div className="app">
      <header>
        <h1>TGH Storm Closure Duration Predictor</h1>
        <p className="subtitle">Predict bridge closure duration based on storm characteristics</p>
      </header>

      <main>
        <StormForm onSubmit={handlePredict} loading={loading} />
        
        {error && <ErrorMessage message={error} />}
        
        {prediction && <PredictionResult data={prediction} />}
      </main>

      <footer>
        <p>Model trained on historical storm data (2004-2024)</p>
        <p>Mean Absolute Error: ±0.9 hours (Leave-One-Out Cross-Validation)</p>
      </footer>
    </div>
  )
}

export default App

