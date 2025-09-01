/**
 * Shooting Method Visualizer
 * Shows the boundary value problem solver iterations
 */

let shootingData = null;
let currentIteration = 0;

function initShootingVisualizer() {
    // Initialize Plotly plot for shooting method
    const initialData = [{
        x: [],
        y: [],
        mode: 'lines',
        name: 'Geodesic Path',
        line: {
            color: '#ffd700',
            width: 3
        }
    }];
    
    const layout = {
        title: {
            text: 'Phase Space (c, v)',
            font: {
                color: '#00d4ff',
                size: 14
            }
        },
        xaxis: {
            title: 'Concentration (ppb)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)'
        },
        yaxis: {
            title: 'Velocity (ppb/t)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)'
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(20, 30, 48, 0.3)',
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            font: {
                color: 'rgba(255, 255, 255, 0.7)',
                size: 10
            }
        },
        margin: {
            l: 50,
            r: 30,
            t: 40,
            b: 50
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('shooting-plot', initialData, layout, config);
    
    // Load shooting iterations
    loadShootingIterations();
}

async function loadShootingIterations() {
    const params = {
        c_start: parseFloat(document.getElementById('c-start').value),
        c_end: parseFloat(document.getElementById('c-end').value),
        wavelength: parseFloat(document.getElementById('wavelength').value)
    };
    
    try {
        const response = await fetch('/api/shooting_iterations', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });
        
        shootingData = await response.json();
        visualizeShootingIterations();
        createObjectiveLandscape();
        
    } catch (error) {
        console.error('Error loading shooting iterations:', error);
    }
}

function visualizeShootingIterations() {
    if (!shootingData || !shootingData.iterations) return;
    
    const iterations = shootingData.iterations;
    const c_target = parseFloat(document.getElementById('c-end').value);
    
    // Create traces for each iteration
    const traces = iterations.map((iter, index) => ({
        x: iter.path,
        y: iter.velocity,
        mode: 'lines',
        name: `v₀ = ${iter.v0.toFixed(2)}`,
        line: {
            color: getIterationColor(iter.error),
            width: iter.error < 1 ? 3 : 1,
            dash: iter.error < 1 ? 'solid' : 'dot'
        },
        opacity: iter.error < 1 ? 1 : 0.3,
        visible: index === 0 ? true : 'legendonly',
        hovertemplate: 
            'c: %{x:.2f} ppb<br>' +
            'v: %{y:.3f} ppb/t<br>' +
            'Error: ' + iter.error.toFixed(3) + '<extra></extra>'
    }));
    
    // Add target marker
    traces.push({
        x: [c_target],
        y: [0],
        mode: 'markers',
        name: 'Target',
        marker: {
            color: '#00ff00',
            size: 12,
            symbol: 'star'
        },
        hovertemplate: 'Target: %{x:.2f} ppb<extra></extra>'
    });
    
    // Update plot
    Plotly.react('shooting-plot', traces, {
        title: {
            text: `Shooting Method: Finding v₀`,
            font: {
                color: '#00d4ff',
                size: 14
            }
        },
        xaxis: {
            title: 'Concentration (ppb)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)',
            range: [
                Math.min(...iterations.flatMap(i => i.path)) - 5,
                Math.max(...iterations.flatMap(i => i.path)) + 5
            ]
        },
        yaxis: {
            title: 'Velocity (ppb/t)',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: 'rgba(255, 255, 255, 0.7)'
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(20, 30, 48, 0.3)',
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            font: {
                color: 'rgba(255, 255, 255, 0.7)',
                size: 9
            }
        }
    });
    
    // Animate through iterations
    animateShootingIterations(iterations);
}

function getIterationColor(error) {
    // Color gradient based on error
    if (error < 0.1) return '#00ff00';  // Green - converged
    if (error < 1) return '#ffd700';    // Gold - close
    if (error < 5) return '#ff8c00';    // Orange - moderate
    return '#ff0000';                   // Red - far
}

function animateShootingIterations(iterations) {
    let currentIdx = 0;
    const animationSpeed = parseFloat(document.getElementById('animation-speed').value);
    
    const animate = () => {
        // Find best iteration so far
        const bestIdx = iterations
            .slice(0, currentIdx + 1)
            .reduce((best, iter, idx) => 
                iter.error < iterations[best].error ? idx : best, 0);
        
        // Update visibility
        const update = {
            visible: iterations.map((_, idx) => {
                if (idx === bestIdx) return true;
                if (idx <= currentIdx) return 'legendonly';
                return false;
            })
        };
        
        Plotly.restyle('shooting-plot', update);
        
        // Update convergence indicator
        updateConvergenceIndicator(iterations[bestIdx]);
        
        currentIdx = (currentIdx + 1) % iterations.length;
        
        setTimeout(animate, 1000 / animationSpeed);
    };
    
    animate();
}

function createObjectiveLandscape() {
    if (!shootingData || !shootingData.iterations) return;
    
    const iterations = shootingData.iterations;
    
    // Create objective function landscape
    const v0_values = iterations.map(iter => iter.v0);
    const errors = iterations.map(iter => iter.error);
    
    // Add a subplot for objective function
    const objectiveTrace = {
        x: v0_values,
        y: errors,
        mode: 'lines+markers',
        name: 'Objective Function',
        line: {
            color: '#00d4ff',
            width: 2
        },
        marker: {
            size: 6,
            color: errors,
            colorscale: [
                [0, '#00ff00'],
                [0.5, '#ffd700'],
                [1, '#ff0000']
            ],
            showscale: false
        }
    };
    
    // Find minimum
    const minIdx = errors.indexOf(Math.min(...errors));
    const minTrace = {
        x: [v0_values[minIdx]],
        y: [errors[minIdx]],
        mode: 'markers',
        name: 'Optimal v₀',
        marker: {
            color: '#00ff00',
            size: 15,
            symbol: 'star',
            line: {
                color: '#fff',
                width: 2
            }
        }
    };
    
    // Create subplot div if it doesn't exist
    if (!document.getElementById('objective-subplot')) {
        const container = document.getElementById('panel-shooting').querySelector('.panel-content');
        const subplot = document.createElement('div');
        subplot.id = 'objective-subplot';
        subplot.style.position = 'absolute';
        subplot.style.bottom = '10px';
        subplot.style.right = '10px';
        subplot.style.width = '45%';
        subplot.style.height = '40%';
        subplot.style.background = 'rgba(0, 0, 0, 0.5)';
        subplot.style.borderRadius = '8px';
        subplot.style.border = '1px solid rgba(0, 212, 255, 0.3)';
        container.appendChild(subplot);
    }
    
    const layout = {
        title: {
            text: 'Objective: ||γ(1) - c₁||²',
            font: {
                color: '#ffd700',
                size: 11
            }
        },
        xaxis: {
            title: 'Initial Velocity v₀',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            color: 'rgba(255, 255, 255, 0.6)',
            titlefont: { size: 10 }
        },
        yaxis: {
            title: 'Error',
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            color: 'rgba(255, 255, 255, 0.6)',
            titlefont: { size: 10 },
            type: 'log'
        },
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(20, 30, 48, 0.3)',
        showlegend: false,
        margin: {
            l: 40,
            r: 20,
            t: 30,
            b: 40
        }
    };
    
    Plotly.newPlot('objective-subplot', [objectiveTrace, minTrace], layout, {
        responsive: true,
        displayModeBar: false
    });
}

function updateShootingVisualization(geodesicData) {
    if (!geodesicData) return;
    
    // Update main plot with optimal geodesic
    const optimalTrace = {
        x: geodesicData.path.map(p => p[0]),
        y: geodesicData.path.map(p => p[1]),
        mode: 'lines+markers',
        name: 'Optimal Geodesic',
        line: {
            color: '#00ff00',
            width: 4
        },
        marker: {
            size: 6,
            color: '#00ff00'
        }
    };
    
    Plotly.addTraces('shooting-plot', [optimalTrace]);
    
    // Update convergence indicator
    updateConvergenceIndicator({
        v0: geodesicData.initial_velocity,
        error: 0,
        convergence: geodesicData.convergence
    });
    
    // Show shooting equation
    const equation = `v_0^* = ${geodesicData.initial_velocity.toFixed(3)}`;
    renderMathEquation('shooting-equation', equation);
}

function updateConvergenceIndicator(iteration) {
    // Create or update convergence indicator
    let indicator = document.getElementById('convergence-indicator');
    
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'convergence-indicator';
        indicator.style.position = 'absolute';
        indicator.style.top = '10px';
        indicator.style.right = '10px';
        indicator.style.padding = '10px';
        indicator.style.background = 'rgba(0, 0, 0, 0.7)';
        indicator.style.borderRadius = '8px';
        indicator.style.border = '1px solid rgba(0, 212, 255, 0.5)';
        indicator.style.fontSize = '12px';
        indicator.style.color = '#fff';
        document.getElementById('panel-shooting').querySelector('.panel-content').appendChild(indicator);
    }
    
    const status = iteration.error < 0.1 ? '✅ Converged' : 
                   iteration.error < 1 ? '🟡 Close' : 
                   '🔴 Searching';
    
    indicator.innerHTML = `
        <div style="margin-bottom: 5px; color: #00d4ff; font-weight: bold;">Shooting Status</div>
        <div>v₀: ${iteration.v0.toFixed(3)}</div>
        <div>Error: ${iteration.error.toFixed(4)}</div>
        <div>Status: ${status}</div>
    `;
}

// Velocity field visualization
function drawVelocityField() {
    const container = document.getElementById('panel-shooting').querySelector('.panel-content');
    
    // Create canvas for velocity field
    const canvas = document.createElement('canvas');
    canvas.id = 'velocity-field';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.opacity = '0.3';
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    // Draw velocity field arrows
    const gridSize = 20;
    const arrowLength = 10;
    
    for (let x = gridSize; x < canvas.width; x += gridSize) {
        for (let y = gridSize; y < canvas.height; y += gridSize) {
            // Map canvas coordinates to phase space
            const c = (x / canvas.width) * 50 + 10;
            const v = ((canvas.height - y) / canvas.height) * 40 - 20;
            
            // Compute field direction (simplified)
            const gamma = 0.1 * Math.sin(c / 10);
            const dv = -gamma * v * v;
            
            // Draw arrow
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(Math.atan2(-dv, v));
            
            ctx.strokeStyle = `rgba(0, 212, 255, ${Math.abs(dv) / 10})`;
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(arrowLength, 0);
            ctx.lineTo(arrowLength - 3, -3);
            ctx.moveTo(arrowLength, 0);
            ctx.lineTo(arrowLength - 3, 3);
            ctx.stroke();
            
            ctx.restore();
        }
    }
}

// Initialize velocity field on load
setTimeout(drawVelocityField, 1000);

// Export function
window.updateShootingVisualization = updateShootingVisualization;