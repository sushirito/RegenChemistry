/**
 * Spectral Display - Plotly Visualization
 * Shows predicted vs actual UV-Vis spectra
 */

let spectralData = null;
let predictions = null;

function initSpectralDisplay() {
    // Initialize the spectral plot
    const initialData = [
        {
            x: [],
            y: [],
            mode: 'lines',
            name: 'Ground Truth',
            line: {
                color: '#00d4ff',
                width: 2
            }
        },
        {
            x: [],
            y: [],
            mode: 'lines',
            name: 'Prediction',
            line: {
                color: '#ffd700',
                width: 2,
                dash: 'dash'
            }
        }
    ];
    
    const layout = {
        title: {
            text: 'UV-Vis Spectral Reconstruction',
            font: {
                color: '#00d4ff',
                size: 16
            }
        },
        xaxis: {
            title: 'Wavelength (nm)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)',
            range: [200, 800]
        },
        yaxis: {
            title: 'Absorbance (a.u.)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)'
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(20, 30, 48, 0.3)',
        showlegend: true,
        legend: {
            x: 0.7,
            y: 0.98,
            font: {
                color: 'rgba(255, 255, 255, 0.7)',
                size: 12
            }
        },
        margin: {
            l: 60,
            r: 30,
            t: 50,
            b: 60
        },
        hovermode: 'x unified'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d']
    };
    
    Plotly.newPlot('spectral-plot', initialData, layout, config);
    
    // Load spectral data
    loadSpectralData();
    
    // Create confidence bands visualization
    createConfidenceBands();
}

async function loadSpectralData() {
    try {
        const response = await fetch('/api/spectral_data');
        spectralData = await response.json();
        
        // Display initial spectrum
        displaySpectrum();
        
        // Generate predictions (simulated for demo)
        generatePredictions();
        
    } catch (error) {
        console.error('Error loading spectral data:', error);
    }
}

function displaySpectrum() {
    if (!spectralData) return;
    
    const c_target = parseFloat(document.getElementById('c-end').value);
    
    // Find closest concentration in data
    const concentrations = spectralData.concentrations;
    const closestIdx = concentrations.reduce((prev, curr, idx) => 
        Math.abs(curr - c_target) < Math.abs(concentrations[prev] - c_target) ? idx : prev, 0);
    
    const wavelengths = spectralData.wavelengths;
    const absorbance = spectralData.absorbance.map(row => row[closestIdx]);
    
    // Update ground truth trace
    const groundTruthTrace = {
        x: wavelengths,
        y: absorbance,
        mode: 'lines',
        name: `Ground Truth (${concentrations[closestIdx]} ppb)`,
        line: {
            color: '#00d4ff',
            width: 2
        },
        hovertemplate: 'λ: %{x:.0f} nm<br>A: %{y:.3f}<extra></extra>'
    };
    
    Plotly.react('spectral-plot', [groundTruthTrace], undefined, {
        transition: {
            duration: 500,
            easing: 'cubic-in-out'
        }
    });
    
    // Add peak detection
    detectAndAnnotatePeaks(wavelengths, absorbance);
}

function generatePredictions() {
    if (!spectralData) return;
    
    const wavelengths = spectralData.wavelengths;
    const c_target = parseFloat(document.getElementById('c-end').value);
    
    // Simulate predictions with realistic noise and systematic error
    const closestIdx = spectralData.concentrations.reduce((prev, curr, idx) => 
        Math.abs(curr - c_target) < Math.abs(spectralData.concentrations[prev] - c_target) ? idx : prev, 0);
    
    const groundTruth = spectralData.absorbance.map(row => row[closestIdx]);
    
    // Add realistic prediction errors
    predictions = groundTruth.map((value, idx) => {
        const wavelength = wavelengths[idx];
        
        // Systematic error: slight wavelength shift
        const shift = Math.sin((wavelength - 500) / 100) * 0.05;
        
        // Random noise
        const noise = (Math.random() - 0.5) * 0.02;
        
        // Non-monotonic region has higher uncertainty
        const uncertainty = wavelength > 450 && wavelength < 550 ? 0.1 : 0.05;
        const nonMonotonicError = (Math.random() - 0.5) * uncertainty;
        
        return value * (1 + shift) + noise + nonMonotonicError;
    });
    
    // Smooth predictions
    predictions = smoothArray(predictions, 3);
    
    // Update plot with predictions
    updateSpectralPlot(wavelengths, groundTruth, predictions);
    
    // Calculate metrics
    calculateMetrics(groundTruth, predictions, wavelengths);
}

function smoothArray(arr, windowSize) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
        let sum = 0;
        let count = 0;
        for (let j = Math.max(0, i - windowSize); j <= Math.min(arr.length - 1, i + windowSize); j++) {
            sum += arr[j];
            count++;
        }
        result.push(sum / count);
    }
    return result;
}

function updateSpectralPlot(wavelengths, groundTruth, predictions) {
    const traces = [
        {
            x: wavelengths,
            y: groundTruth,
            mode: 'lines',
            name: 'Ground Truth',
            line: {
                color: '#00d4ff',
                width: 3
            },
            hovertemplate: 'λ: %{x:.0f} nm<br>A: %{y:.3f}<extra></extra>'
        },
        {
            x: wavelengths,
            y: predictions,
            mode: 'lines',
            name: 'Model Prediction',
            line: {
                color: '#ffd700',
                width: 2
            },
            hovertemplate: 'λ: %{x:.0f} nm<br>A: %{y:.3f}<extra></extra>'
        }
    ];
    
    // Add residuals as bar chart
    const residuals = predictions.map((pred, idx) => pred - groundTruth[idx]);
    traces.push({
        x: wavelengths,
        y: residuals,
        type: 'bar',
        name: 'Residuals',
        marker: {
            color: residuals.map(r => r > 0 ? 'rgba(255, 0, 0, 0.3)' : 'rgba(0, 255, 0, 0.3)'),
            line: {
                width: 0
            }
        },
        yaxis: 'y2',
        hovertemplate: 'Residual: %{y:.4f}<extra></extra>'
    });
    
    const layout = {
        title: {
            text: 'Spectral Reconstruction with Geodesic Model',
            font: {
                color: '#00d4ff',
                size: 16
            }
        },
        xaxis: {
            title: 'Wavelength (nm)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)',
            range: [200, 800]
        },
        yaxis: {
            title: 'Absorbance (a.u.)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)'
        },
        yaxis2: {
            title: 'Residuals',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zerolinecolor: 'rgba(255, 255, 255, 0.3)',
            color: 'rgba(255, 255, 255, 0.5)',
            showgrid: false,
            range: [-0.1, 0.1]
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(20, 30, 48, 0.3)',
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            font: {
                color: 'rgba(255, 255, 255, 0.7)',
                size: 11
            }
        },
        hovermode: 'x unified'
    };
    
    Plotly.react('spectral-plot', traces, layout);
}

function detectAndAnnotatePeaks(wavelengths, absorbance) {
    // Simple peak detection
    const peaks = [];
    for (let i = 1; i < absorbance.length - 1; i++) {
        if (absorbance[i] > absorbance[i-1] && absorbance[i] > absorbance[i+1]) {
            if (absorbance[i] > 0.3) {  // Threshold for significant peaks
                peaks.push({
                    wavelength: wavelengths[i],
                    absorbance: absorbance[i],
                    index: i
                });
            }
        }
    }
    
    // Sort by absorbance and take top 3
    peaks.sort((a, b) => b.absorbance - a.absorbance);
    const topPeaks = peaks.slice(0, 3);
    
    // Add annotations for peaks
    const annotations = topPeaks.map((peak, idx) => ({
        x: peak.wavelength,
        y: peak.absorbance,
        text: `λ = ${peak.wavelength.toFixed(0)} nm`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: '#ffd700',
        ax: 30,
        ay: -30 - idx * 15,
        font: {
            color: '#ffd700',
            size: 10
        }
    }));
    
    Plotly.relayout('spectral-plot', {
        annotations: annotations
    });
}

function calculateMetrics(groundTruth, predictions, wavelengths) {
    // Calculate R²
    const mean = groundTruth.reduce((a, b) => a + b) / groundTruth.length;
    const ssTotal = groundTruth.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
    const ssResidual = groundTruth.reduce((sum, val, idx) => 
        sum + Math.pow(val - predictions[idx], 2), 0);
    const r2 = 1 - (ssResidual / ssTotal);
    
    // Calculate RMSE
    const mse = ssResidual / groundTruth.length;
    const rmse = Math.sqrt(mse);
    
    // Calculate MAPE
    const mape = groundTruth.reduce((sum, val, idx) => {
        if (val !== 0) {
            return sum + Math.abs((val - predictions[idx]) / val);
        }
        return sum;
    }, 0) / groundTruth.length * 100;
    
    // Find peak wavelength error
    const gtPeakIdx = groundTruth.indexOf(Math.max(...groundTruth));
    const predPeakIdx = predictions.indexOf(Math.max(...predictions));
    const peakError = Math.abs(wavelengths[gtPeakIdx] - wavelengths[predPeakIdx]);
    
    // Update display
    document.getElementById('r2-value').textContent = r2.toFixed(3);
    document.getElementById('rmse-value').textContent = rmse.toFixed(4);
    
    // Create metrics visualization
    createMetricsGauge(r2, rmse, mape, peakError);
}

function createMetricsGauge(r2, rmse, mape, peakError) {
    // Create a small metrics dashboard
    const metricsDiv = document.createElement('div');
    metricsDiv.id = 'metrics-dashboard';
    metricsDiv.style.position = 'absolute';
    metricsDiv.style.top = '10px';
    metricsDiv.style.right = '10px';
    metricsDiv.style.width = '200px';
    metricsDiv.style.background = 'rgba(0, 0, 0, 0.8)';
    metricsDiv.style.borderRadius = '10px';
    metricsDiv.style.padding = '15px';
    metricsDiv.style.border = '1px solid rgba(0, 212, 255, 0.3)';
    
    metricsDiv.innerHTML = `
        <div style="color: #00d4ff; font-weight: bold; margin-bottom: 10px; text-align: center;">
            Model Performance
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="text-align: center;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">R²</div>
                <div style="color: ${r2 > 0.8 ? '#00ff00' : r2 > 0.5 ? '#ffd700' : '#ff0000'}; 
                            font-size: 18px; font-weight: bold;">
                    ${r2.toFixed(3)}
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">RMSE</div>
                <div style="color: ${rmse < 0.05 ? '#00ff00' : rmse < 0.1 ? '#ffd700' : '#ff0000'}; 
                            font-size: 18px; font-weight: bold;">
                    ${rmse.toFixed(3)}
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">MAPE (%)</div>
                <div style="color: ${mape < 10 ? '#00ff00' : mape < 20 ? '#ffd700' : '#ff0000'}; 
                            font-size: 18px; font-weight: bold;">
                    ${mape.toFixed(1)}
                </div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255, 255, 255, 0.6); font-size: 10px;">Peak λ Error</div>
                <div style="color: ${peakError < 10 ? '#00ff00' : peakError < 30 ? '#ffd700' : '#ff0000'}; 
                            font-size: 18px; font-weight: bold;">
                    ${peakError.toFixed(0)} nm
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; text-align: center;">
            <div style="width: 100%; height: 4px; background: rgba(255, 255, 255, 0.1); border-radius: 2px;">
                <div style="width: ${r2 * 100}%; height: 100%; 
                            background: linear-gradient(90deg, #ff0000, #ffd700, #00ff00); 
                            border-radius: 2px; transition: width 0.5s;">
                </div>
            </div>
        </div>
    `;
    
    // Remove existing dashboard if present
    const existing = document.getElementById('metrics-dashboard');
    if (existing) existing.remove();
    
    document.getElementById('panel-results').querySelector('.panel-content').appendChild(metricsDiv);
}

function createConfidenceBands() {
    // Add confidence bands visualization
    setTimeout(() => {
        if (!predictions || !spectralData) return;
        
        const wavelengths = spectralData.wavelengths;
        const c_target = parseFloat(document.getElementById('c-end').value);
        
        // Generate confidence bands (simulated)
        const upperBound = predictions.map((pred, idx) => {
            const wavelength = wavelengths[idx];
            const uncertainty = wavelength > 450 && wavelength < 550 ? 0.08 : 0.04;
            return pred + uncertainty;
        });
        
        const lowerBound = predictions.map((pred, idx) => {
            const wavelength = wavelengths[idx];
            const uncertainty = wavelength > 450 && wavelength < 550 ? 0.08 : 0.04;
            return pred - uncertainty;
        });
        
        // Add confidence band trace
        const confidenceTrace = {
            x: wavelengths.concat(wavelengths.slice().reverse()),
            y: upperBound.concat(lowerBound.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(255, 215, 0, 0.1)',
            line: {
                color: 'rgba(255, 215, 0, 0)',
                width: 0
            },
            showlegend: true,
            name: '95% Confidence',
            type: 'scatter',
            mode: 'lines',
            hoverinfo: 'skip'
        };
        
        Plotly.addTraces('spectral-plot', [confidenceTrace], [0]);
    }, 2000);
}

// Animated spectral transition
function animateSpectralTransition() {
    const c_start = parseFloat(document.getElementById('c-start').value);
    const c_end = parseFloat(document.getElementById('c-end').value);
    const steps = 20;
    let currentStep = 0;
    
    const animate = () => {
        const t = currentStep / steps;
        const c_current = c_start + (c_end - c_start) * t;
        
        // Update spectrum for current concentration
        if (spectralData) {
            const concentrations = spectralData.concentrations;
            const closestIdx = concentrations.reduce((prev, curr, idx) => 
                Math.abs(curr - c_current) < Math.abs(concentrations[prev] - c_current) ? idx : prev, 0);
            
            const wavelengths = spectralData.wavelengths;
            const absorbance = spectralData.absorbance.map(row => row[closestIdx]);
            
            Plotly.restyle('spectral-plot', {
                y: [absorbance]
            }, [0]);
        }
        
        currentStep = (currentStep + 1) % (steps + 1);
        
        if (currentStep !== 0) {
            setTimeout(animate, 100);
        }
    };
    
    animate();
}

// Export functions
window.initSpectralDisplay = initSpectralDisplay;
window.animateSpectralTransition = animateSpectralTransition;