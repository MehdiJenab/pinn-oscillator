# PINN Harmonic Oscillator

A Physics-Informed Neural Network (PINN) implementation for solving the harmonic oscillator differential equation.

## Overview

This project implements a PINN to solve the harmonic oscillator equation:
```
d²x/dt² + ω²x = 0
```

With initial conditions:
- x(0) = x₀ (initial position)
- dx/dt(0) = 0 (initial velocity)

## Project Structure

```
pinn-harmonic-oscillator/
├── src/
│   └── pinn_harmonic_oscillator/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── pinn.py
│       └── api/
│           ├── __init__.py
│           └── app.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd pinn-harmonic-oscillator

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
# Start the Flask server
cd src
python -m pinn_harmonic_oscillator.api.app

# Or use the command-line interface
pinn-oscillator
```

The web interface will be available at `http://localhost:5000`

### Using the Web Interface

1. **Adjust Parameters**: Use the controls on the left sidebar to modify:
   - `ω (omega)`: Natural frequency of the oscillator
   - `t_max`: Maximum time for simulation
   - `epochs`: Number of training iterations
   - `hidden_dim`: Neural network hidden layer size
   - `learning_rate`: Optimizer learning rate
   - `weight_decay`: L2 regularization strength
   - `λ_ic`: Weight for initial condition loss
   - `λ_physics`: Weight for physics-informed loss
   - `num_domain_points`: Number of collocation points

2. **Train Model**: Click "Train PINN" to start training

3. **View Results**: Switch between tabs to see:
   - **Predictions**: PINN solution vs analytical solution
   - **Training History**: Loss curves over epochs
   - **Residuals**: Physics-informed constraint satisfaction
   - **Phase Space**: Position vs velocity plot

## Technical Details

### Physics-Informed Neural Network

The PINN architecture consists of:
- **Input**: Time `t`
- **Hidden Layers**: 2 layers with configurable width
- **Activation**: tanh
- **Output**: Position `x(t)`

### Loss Function

```
Loss = λ_ic * Loss_ic + λ_physics * Loss_physics
```

Where:
- `Loss_ic`: Mean squared error for initial conditions
- `Loss_physics`: Mean squared error of the residual

## Mathematical Background

For the harmonic oscillator equation `d²x/dt² + ω²x = 0`, the analytical solution is:
```
x(t) = x₀ * cos(ωt)
```

The PINN approximates this solution using automatic differentiation to compute derivatives.

## License

MIT License