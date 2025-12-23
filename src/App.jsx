import React, { useState, useEffect } from "react";
import StormForm from "./components/StormForm";
import PredictionResult from "./components/PredictionResult";
import ErrorMessage from "./components/ErrorMessage";
import ModelStats from "./components/ModelStats";
import DataStats from "./components/DataStats";
import { API_BASE_URL } from "./config";
import { loadModelData, predictClosureDuration } from "./prediction";
import "./App.css";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showModelStats, setShowModelStats] = useState(false);
  const [showDataStats, setShowDataStats] = useState(false);
  const [mae, setMae] = useState(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [useClientSide, setUseClientSide] = useState(false);

  useEffect(() => {
    loadModelData()
      .then((success) => {
        if (success) {
          setModelLoaded(true);
          setUseClientSide(true);
        } else {
          setUseClientSide(false);
        }
      })
      .catch(() => {
        setUseClientSide(false);
      });
  }, []);

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      let data;

      if (useClientSide && modelLoaded) {
        const result = predictClosureDuration(formData);
        data = {
          prediction_hours: result.prediction_hours,
          prediction_days: Math.round((result.prediction_hours / 24) * 10) / 10,
          ci_lower: result.ci_lower,
          ci_upper: result.ci_upper,
          features: result.features,
        };
      } else {
        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Prediction failed");
        }

        data = await response.json();
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleModelStatsClick = () => {
    if (showModelStats) {
      setShowModelStats(false);
    } else {
      setShowDataStats(false);
      setShowModelStats(true);
    }
  };

  const handleDataStatsClick = () => {
    if (showDataStats) {
      setShowDataStats(false);
    } else {
      setShowModelStats(false);
      setShowDataStats(true);
    }
  };

  useEffect(() => {
    if (!useClientSide && API_BASE_URL) {
      fetch(`${API_BASE_URL}/api/model-stats`)
        .then((res) => res.json())
        .then((data) => {
          if (data.model_fit && data.model_fit.mae) {
            setMae(data.model_fit.mae);
          }
        })
        .catch(() => {
          setMae(0.9);
        });
    } else {
      setMae(0.9);
    }
  }, [useClientSide]);

  return (
    <div className="app">
      <header>
        <h1>TGH Storm Closure Duration Predictor</h1>
        <p className="subtitle">
          Predict bridge closure duration based on storm characteristics
        </p>
      </header>

      <main>
        <div className="stats-buttons">
          <button
            className={`stats-button ${showModelStats ? "active" : ""}`}
            onClick={handleModelStatsClick}
          >
            {showModelStats ? "Hide Model Stats" : "Model Stats"}
          </button>
          <button
            className={`stats-button ${showDataStats ? "active" : ""}`}
            onClick={handleDataStatsClick}
          >
            {showDataStats ? "Hide Data Stats" : "Data Stats"}
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
        <p>
          Created by{" "}
          <a
            href="https://andresquast.com"
            target="_blank"
            rel="noopener noreferrer"
          >
            Andres Quast
          </a>{" "}
          |{" "}
          <a
            href="/ProjectWriteDec2025.pdf"
            target="_blank"
            rel="noopener noreferrer"
          >
            Click for most recent writeup
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
