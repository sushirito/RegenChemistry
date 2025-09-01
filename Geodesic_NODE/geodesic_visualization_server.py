"""
Geodesic Spectral Manifold Visualization Server
Research-grade interactive visualization for the manifold-ODE spectral model
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
import json
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the spectral data
def load_spectral_data():
    """Load UV-Vis spectral data"""
    try:
        df = pd.read_csv('0.30MB_AuNP_As.csv')
        wavelengths = df['Wavelength (nm)'].values
        concentrations = df.columns[1:].values.astype(float)
        absorbance_matrix = df.iloc[:, 1:].values
        return wavelengths, concentrations, absorbance_matrix
    except:
        # Generate synthetic data for demonstration
        wavelengths = np.linspace(200, 800, 601)
        concentrations = np.array([10, 20, 30, 40, 50, 60])
        absorbance_matrix = np.random.rand(601, 6)
        return wavelengths, concentrations, absorbance_matrix

# Metric function (simplified for visualization)
def metric_function(c, lambda_val, params=None):
    """Compute Riemannian metric g(c, λ)"""
    c_norm = (c - 30) / 30
    lambda_norm = (lambda_val - 500) / 300
    
    # Simulate learned metric with interesting structure
    g = 1.0 + 0.5 * np.exp(-((c_norm - 0.5)**2 + (lambda_norm)**2) / 0.1)
    g += 0.3 * np.sin(2 * np.pi * lambda_norm) * np.exp(-c_norm**2)
    return max(0.1, g)  # Ensure positive definite

# Christoffel symbol computation
def compute_christoffel(c, lambda_val, epsilon=1e-4):
    """Compute Christoffel symbol Γ"""
    g_plus = metric_function(c + epsilon, lambda_val)
    g_minus = metric_function(c - epsilon, lambda_val)
    g_center = metric_function(c, lambda_val)
    
    dg_dc = (g_plus - g_minus) / (2 * epsilon)
    gamma = 0.5 * dg_dc / g_center
    return gamma

# Geodesic ODE system
def geodesic_ode(state, t, lambda_val):
    """Geodesic differential equation: d²c/dt² = -Γ(c,λ)(dc/dt)²"""
    c, v = state
    gamma = compute_christoffel(c, lambda_val)
    dc_dt = v
    dv_dt = -gamma * v**2
    return [dc_dt, dv_dt]

# Shooting method for BVP
def solve_geodesic_bvp(c_start, c_end, lambda_val, num_points=50):
    """Solve boundary value problem using shooting method"""
    def objective(v0):
        initial_state = [c_start, v0]
        t_span = np.linspace(0, 1, 10)
        solution = odeint(geodesic_ode, initial_state, t_span, args=(lambda_val,))
        c_final = solution[-1, 0]
        return (c_final - c_end)**2
    
    # Initial guess: linear interpolation velocity
    v0_init = c_end - c_start
    
    # Optimize to find correct initial velocity
    result = minimize_scalar(objective, bounds=(v0_init - 10, v0_init + 10), method='bounded')
    v0_optimal = result.x
    
    # Generate full geodesic path
    initial_state = [c_start, v0_optimal]
    t_span = np.linspace(0, 1, num_points)
    geodesic_path = odeint(geodesic_ode, initial_state, t_span, args=(lambda_val,))
    
    # Compute path length
    path_length = 0
    for i in range(1, len(geodesic_path)):
        dc = geodesic_path[i, 0] - geodesic_path[i-1, 0]
        g = metric_function((geodesic_path[i, 0] + geodesic_path[i-1, 0])/2, lambda_val)
        path_length += np.sqrt(g * dc**2)
    
    return {
        'path': geodesic_path.tolist(),
        't': t_span.tolist(),
        'path_length': float(path_length),
        'initial_velocity': float(v0_optimal),
        'convergence': result.success
    }

# Generate manifold mesh for 3D visualization
def generate_manifold_mesh(resolution=50):
    """Generate 3D mesh data for manifold visualization"""
    c_range = np.linspace(10, 60, resolution)
    lambda_range = np.linspace(200, 800, resolution)
    
    C, L = np.meshgrid(c_range, lambda_range)
    G = np.zeros_like(C)
    
    for i in range(resolution):
        for j in range(resolution):
            G[i, j] = metric_function(C[i, j], L[i, j])
    
    # Create surface with metric as height
    Z = G * 20  # Scale for visualization
    
    return {
        'x': C.tolist(),
        'y': L.tolist(),
        'z': Z.tolist(),
        'metric': G.tolist()
    }

@app.route('/')
def index():
    """Serve main visualization page"""
    return render_template('index.html')

@app.route('/api/spectral_data')
def get_spectral_data():
    """Get spectral data for visualization"""
    wavelengths, concentrations, absorbance = load_spectral_data()
    return jsonify({
        'wavelengths': wavelengths.tolist(),
        'concentrations': concentrations.tolist(),
        'absorbance': absorbance.tolist()
    })

@app.route('/api/manifold_mesh')
def get_manifold_mesh():
    """Get 3D manifold mesh data"""
    mesh_data = generate_manifold_mesh(resolution=40)
    return jsonify(mesh_data)

@app.route('/api/geodesic', methods=['POST'])
def compute_geodesic():
    """Compute geodesic path between two concentrations"""
    data = request.json
    c_start = data.get('c_start', 20)
    c_end = data.get('c_end', 50)
    lambda_val = data.get('wavelength', 500)
    
    result = solve_geodesic_bvp(c_start, c_end, lambda_val)
    return jsonify(result)

@app.route('/api/metric_field')
def get_metric_field():
    """Get metric tensor field for visualization"""
    resolution = request.args.get('resolution', 30, type=int)
    c_range = np.linspace(10, 60, resolution)
    lambda_range = np.linspace(200, 800, resolution)
    
    metric_field = []
    for c in c_range:
        for lam in lambda_range:
            g = metric_function(c, lam)
            gamma = compute_christoffel(c, lam)
            metric_field.append({
                'c': float(c),
                'lambda': float(lam),
                'g': float(g),
                'gamma': float(gamma)
            })
    
    return jsonify({
        'field': metric_field,
        'c_range': c_range.tolist(),
        'lambda_range': lambda_range.tolist()
    })

@app.route('/api/shooting_iterations', methods=['POST'])
def get_shooting_iterations():
    """Get shooting method iterations for visualization"""
    data = request.json
    c_start = data.get('c_start', 20)
    c_end = data.get('c_end', 50)
    lambda_val = data.get('wavelength', 500)
    
    # Generate multiple shooting attempts
    v0_range = np.linspace(c_end - c_start - 10, c_end - c_start + 10, 20)
    iterations = []
    
    for v0 in v0_range:
        initial_state = [c_start, v0]
        t_span = np.linspace(0, 1, 20)
        solution = odeint(geodesic_ode, initial_state, t_span, args=(lambda_val,))
        c_final = solution[-1, 0]
        error = abs(c_final - c_end)
        
        iterations.append({
            'v0': float(v0),
            'path': solution[:, 0].tolist(),
            'velocity': solution[:, 1].tolist(),
            'c_final': float(c_final),
            'error': float(error)
        })
    
    return jsonify({'iterations': iterations})

@app.route('/api/network_architecture')
def get_network_architecture():
    """Get neural network architecture details for visualization"""
    architecture = {
        'metric_network': {
            'layers': [
                {'name': 'Input', 'size': 2, 'activation': None},
                {'name': 'Hidden1', 'size': 64, 'activation': 'Tanh'},
                {'name': 'Hidden2', 'size': 128, 'activation': 'Tanh'},
                {'name': 'Output', 'size': 1, 'activation': 'Softplus'}
            ],
            'parameters': 64*2 + 64 + 128*64 + 128 + 1*128 + 1
        },
        'decoder_network': {
            'layers': [
                {'name': 'Features', 'size': 5, 'activation': None},
                {'name': 'Hidden1', 'size': 64, 'activation': 'Tanh'},
                {'name': 'Hidden2', 'size': 128, 'activation': 'Tanh'},
                {'name': 'Absorbance', 'size': 1, 'activation': None}
            ],
            'parameters': 64*5 + 64 + 128*64 + 128 + 1*128 + 1
        },
        'pipeline': [
            {'step': 'Input', 'description': '[c₀, c₁, λ]'},
            {'step': 'Metric Learning', 'description': 'g(c,λ) = MetricNet(c,λ)'},
            {'step': 'Christoffel', 'description': 'Γ = ½(∂g/∂c)/g'},
            {'step': 'Geodesic ODE', 'description': 'd²c/dt² = -Γ(dc/dt)²'},
            {'step': 'Shooting Method', 'description': 'Find v₀: γ(1) = c₁'},
            {'step': 'Path Features', 'description': '[c_final, c_mean, L, v_max]'},
            {'step': 'Decoder', 'description': 'A = Decoder(features, λ)'}
        ]
    }
    return jsonify(architecture)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    print("🚀 Geodesic Spectral Manifold Visualization Server")
    print("📊 Access visualization at: http://localhost:5000")
    print("🎨 Research-grade interactive architecture display")
    
    app.run(debug=True, port=5000)