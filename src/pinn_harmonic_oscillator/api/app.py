"""Flask API for PINN Harmonic Oscillator."""
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import numpy as np
import torch
import os
import json
from datetime import datetime

from pinn_harmonic_oscillator.core.physics import HarmonicOscillator, solve_ode
from pinn_harmonic_oscillator.models.pinn import PINN

# Get the project root directory (pinn-harmonic-oscillator/)
# Since the app.py is in src/pinn_harmonic_oscillator/api/, we go 3 levels up
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
print(f"SCRIPT_DIR: {SCRIPT_DIR}")
print(f"BASE_DIR: {BASE_DIR}")

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'frontend'), template_folder=os.path.join(BASE_DIR, 'frontend'))
CORS(app)

# Routes for serving frontend files (CSS, JS)
@app.route('/styles.css')
def styles(filename='styles.css'):
    return send_from_directory(os.path.join(BASE_DIR, 'frontend'), filename)

@app.route('/script.js')
def script(filename='script.js'):
    return send_from_directory(os.path.join(BASE_DIR, 'frontend'), filename)

# Global model storage (for persistence across requests)
models = {}
training_history = {}


def create_pinn_model(input_dim=1, hidden_dim=50, output_dim=1, omega=1.0, damping_coefficient=0.0):
    """Create a new PINN model.

    Args:
        input_dim: Dimension of input (default 1 for time)
        hidden_dim: Number of hidden units
        output_dim: Dimension of output (default 1 for scalar position)
        omega: Initial guess for angular frequency
        damping_coefficient: Damping coefficient for damped oscillations
    """
    model = PINN(input_dim, hidden_dim, output_dim, omega=omega, damping_coefficient=damping_coefficient)
    return model


def train_model(omega, damping_coefficient, t_max, num_domain_points, hidden_dim, epochs, learning_rate, weight_decay, lambda_ic, lambda_physics):
    """Train a PINN model with given parameters."""
    print(f"Training with parameters: omega={omega}, damping_coefficient={damping_coefficient}, t_max={t_max}, epochs={epochs}")

    # Create oscillator
    oscillator = HarmonicOscillator(mass=1.0, k_spring=omega**2, damping_coefficient=damping_coefficient)
    
    # Generate training data
    t_domain = np.linspace(0, t_max, num_domain_points)
    t_domain = torch.FloatTensor(t_domain).unsqueeze(1)
    t_domain.requires_grad = True
    
    t_ic = torch.FloatTensor([[0.0]])
    x_ic = torch.FloatTensor([[1.0]])  # Initial position = 1
    
    # Create model
    pinn = create_pinn_model(input_dim=1, hidden_dim=hidden_dim, output_dim=1, omega=omega, damping_coefficient=damping_coefficient)
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train
    history = {
        'train_loss': [],
        'ic_loss': [],
        'physics_loss': []
    }
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Predictions for domain
        x_domain = pinn.forward(t_domain)

        # Physics loss
        dx_dt = torch.autograd.grad(x_domain.sum(), t_domain, create_graph=True, retain_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_domain, create_graph=True)[0]

        # For damped oscillator: d2x/dt2 + gamma * dx/dt + omega^2 * x = 0
        # where gamma = damping_coefficient / mass (mass = 1.0)
        gamma = damping_coefficient
        ode_residual = d2x_dt2 + gamma * dx_dt + omega**2 * x_domain
        physics_loss = (ode_residual ** 2).mean()
        
        # IC loss
        x_ic_pred = pinn.forward(t_ic)
        ic_loss = torch.mean((x_ic_pred - x_ic) ** 2)
        
        # Total loss
        total_loss = lambda_ic * ic_loss + lambda_physics * physics_loss
        
        # Backprop
        total_loss.backward()
        optimizer.step()
        
        history['train_loss'].append(total_loss.item())
        history['ic_loss'].append(ic_loss.item())
        history['physics_loss'].append(physics_loss.item())
    
    # Get predictions
    t_test = torch.linspace(0, t_max, 500).unsqueeze(1)
    t_test.requires_grad = True
    
    with torch.no_grad():
        x_pred = pinn.forward(t_test)
    
    t_test_np = t_test.detach().squeeze().numpy()
    x_pred_np = x_pred.numpy().flatten()
    
    # Get analytical solution
    t_span = (0, t_max)
    y0 = [1.0, 0.0]
    t_eval = np.linspace(0, t_max, 500)
    analytical_sol = solve_ode(oscillator, t_span, y0, t_eval)
    x_analytical = analytical_sol.y[0]

    # Compute residual
    dt = np.diff(t_test_np).mean()
    dx_dt = np.gradient(x_pred_np, dt)
    d2x_dt2 = np.gradient(dx_dt, dt)
    # For damped oscillator: residual should be d2x/dt2 + gamma * dx/dt + omega^2 * x
    residual = d2x_dt2 + damping_coefficient * dx_dt + omega**2 * x_pred_np
    
    # Compute velocity for phase space
    dx_dt = np.gradient(x_pred_np, dt)
    
    return {
        't': t_test_np.tolist(),
        'x_pred': x_pred_np.tolist(),
        'x_analytical': x_analytical.tolist(),
        'residual': residual.tolist(),
        'velocity': dx_dt.tolist(),
        'history': {
            'train_loss': history['train_loss'][-100:],  # Last 100 epochs
            'ic_loss': history['ic_loss'][-100:],
            'physics_loss': history['physics_loss'][-100:]
        },
        'final_loss': history['train_loss'][-1]
    }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train():
    """Train a new model with given parameters."""
    data = request.json
    
    omega = float(data.get('omega', 1.0))
    t_max = float(data.get('t_max', 10.0))
    num_domain_points = int(data.get('num_domain_points', 500))
    hidden_dim = int(data.get('hidden_dim', 50))
    epochs = int(data.get('epochs', 1000))
    learning_rate = float(data.get('learning_rate', 0.02))
    weight_decay = float(data.get('weight_decay', 0.01))
    lambda_ic = float(data.get('lambda_ic', 10.0))
    lambda_physics = float(data.get('lambda_physics', 1.0))
    
    # Extract damping_coefficient from data
    damping_coefficient = float(data.get('damping_coefficient', 0.0))

    # Validate parameters
    if not (0.1 <= omega <= 10):
        return jsonify({'error': 'omega must be between 0.1 and 10'}), 400
    if not (0 <= damping_coefficient <= 10):
        return jsonify({'error': 'damping_coefficient must be between 0 and 10'}), 400
    if not (1 <= t_max <= 50):
        return jsonify({'error': 't_max must be between 1 and 50'}), 400
    if not (100 <= epochs <= 10000):
        return jsonify({'error': 'epochs must be between 100 and 10000'}), 400

    # Train model
    result = train_model(
        omega=omega,
        damping_coefficient=damping_coefficient,
        t_max=t_max,
        num_domain_points=num_domain_points,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lambda_ic=lambda_ic,
        lambda_physics=lambda_physics
    )
    
    return jsonify(result)


@app.route('/api/analytical', methods=['POST'])
def analytical():
    """Get analytical solution for given parameters."""
    data = request.json

    omega = float(data.get('omega', 1.0))
    damping_coefficient = float(data.get('damping_coefficient', 0.0))
    t_max = float(data.get('t_max', 10.0))

    # Create oscillator
    oscillator = HarmonicOscillator(mass=1.0, k_spring=omega**2, damping_coefficient=damping_coefficient)

    # Get analytical solution
    t_span = (0, t_max)
    y0 = [1.0, 0.0]
    t_eval = np.linspace(0, t_max, 500)
    analytical_sol = solve_ode(oscillator, t_span, y0, t_eval)

    return jsonify({
        't': t_eval.tolist(),
        'x_analytical': analytical_sol.y[0].tolist()
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
