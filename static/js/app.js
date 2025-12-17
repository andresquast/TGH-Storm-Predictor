// TGH Storm Closure Duration Predictor - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('stormForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultContainer = document.getElementById('resultContainer');
    const errorContainer = document.getElementById('errorContainer');
    
    // Set current month as default
    const currentMonth = new Date().getMonth() + 1;
    document.getElementById('month').value = currentMonth;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results/errors
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        
        // Disable button and show loading state
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';
        
        // Collect form data
        const formData = {
            category: parseFloat(document.getElementById('category').value),
            max_wind: parseFloat(document.getElementById('max_wind').value),
            storm_surge: parseFloat(document.getElementById('storm_surge').value),
            track_distance: parseFloat(document.getElementById('track_distance').value),
            forward_speed: parseFloat(document.getElementById('forward_speed').value),
            month: parseFloat(document.getElementById('month').value)
        };
        
        try {
            // Make API request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }
            
            // Display results
            displayResults(data);
            
        } catch (error) {
            // Display error
            displayError(error.message);
        } finally {
            // Re-enable button
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Closure Duration';
        }
    });
    
    function displayResults(data) {
        // Update prediction display
        document.getElementById('predictionHours').textContent = data.prediction_hours.toFixed(1);
        document.getElementById('predictionDays').textContent = data.prediction_days.toFixed(1);
        
        // Update feature summary
        const features = data.features;
        document.getElementById('summaryCategory').textContent = features.category;
        document.getElementById('summaryWind').textContent = features.max_wind;
        document.getElementById('summarySurge').textContent = features.storm_surge;
        document.getElementById('summaryDistance').textContent = features.track_distance;
        document.getElementById('summarySpeed').textContent = features.forward_speed;
        
        // Convert month number to name
        const monthNames = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December'];
        document.getElementById('summaryMonth').textContent = monthNames[features.month];
        
        // Show result container
        resultContainer.classList.remove('hidden');
        
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    function displayError(message) {
        document.getElementById('errorMessage').textContent = message;
        errorContainer.classList.remove('hidden');
        
        // Scroll to error
        errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Add some helpful input validation and auto-fill suggestions
    const categorySelect = document.getElementById('category');
    const maxWindInput = document.getElementById('max_wind');
    
    // Auto-fill wind speed based on category
    categorySelect.addEventListener('change', function() {
        const category = parseInt(this.value);
        const windRanges = {
            0: 30,   // Tropical Depression
            1: 85,   // Category 1
            2: 100,  // Category 2
            3: 120,  // Category 3
            4: 145,  // Category 4
            5: 165   // Category 5
        };
        
        if (windRanges[category] !== undefined && !maxWindInput.value) {
            maxWindInput.value = windRanges[category];
        }
    });
});

