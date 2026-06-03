"""Main example script for PINN Harmonic Oscillator.

This script demonstrates how to use the PINN to solve the harmonic oscillator ODE:
    d²x/dt² + ω²x = 0

The script:
1. Creates a harmonic oscillator system
2. Generates training data
3. Trains a PINN model
4. Visualizes the results
"""

import numpy as np
import torch
import os
import sys

from pinn_harmonic_oscillator.core.physics import HarmonicOscillator, solve_ode
from pinn_harmonic_oscillator.models.pinn import PINN
from pinn_harmonic_oscillator.train.trainer import Trainer
from pinn_harmonic_oscillator.data.data_generator import (
    generate_domain_samples,
    generate_boundary_points,
    generate_initial_conditions,
)
from pinn_harmonic_oscillator.visualize.visualizer import Visualizer


def run_example(omega=1.0, t_max=10.0, num_domain_points=500, hidden_dim=64, epochs=30000, lr=0.01):
    """Run the PINN example for harmonic oscillator.

    Args:
        omega: Natural frequency of the oscillator
        t_max: Maximum time for training
        num_domain_points: Number of collocation points
        hidden_dim: Number of hidden units in each layer
        epochs: Number of training epochs
        lr: Learning rate for optimizer
    """
    print("=" * 60)
    print("PINN Harmonic Oscillator Example")
    print("=" * 60)
    print()

    print(f"Configuration:")
    print(f"  - Natural frequency (ω): {omega}")
    print(f"  - Time range: [0, {t_max}]")
    print(f"  - Domain points: {num_domain_points}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {lr}")
    print()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("1. Creating harmonic oscillator system...")
    oscillator = HarmonicOscillator(mass=1.0, k_spring=omega**2)
    print(f"   Mass: {oscillator.mass}")
    print(f"   Spring constant: {oscillator.k_spring}")
    print(f"   Angular frequency: {oscillator.angular_freq}")
    print()

    print("2. Computing analytical solution...")
    t_span = (0, t_max)
    y0 = [1.0, 0.0]
    t_eval = np.linspace(0, t_max, 500)
    analytical_sol = solve_ode(oscillator, t_span, y0, t_eval)

    x_analytical = analytical_sol.y[0]
    v_analytical = analytical_sol.y[1]
    print("   Analytical solution computed (cosine wave)")
    print()

    print("3. Generating training data...")
    t_domain = generate_domain_samples(
        t_min=0.0, t_max=t_max, n_samples=num_domain_points
    )
    t_domain = torch.FloatTensor(t_domain).unsqueeze(1)
    t_domain.requires_grad = True

    t_ic = torch.FloatTensor([[0.0]])
    x_ic = torch.FloatTensor([[y0[0]]])

    print(f"   Domain samples: {t_domain.shape[0]}")
    print(f"   Initial conditions: t=0, x={y0[0]}, v={y0[1]}")
    print()

    print("4. Creating PINN model...")
    input_dim = 1
    output_dim = 1

    pinn = PINN(input_dim, hidden_dim, output_dim, omega=omega)
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=lr, weight_decay=1e-5)

    print(f"   Input dim: {input_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Output dim: {output_dim}")
    print(f"   Parameters: {pinn.get_param_count()}")
    print()

    print("5. Training PINN...")
    trainer = Trainer(pinn, optimizer, epochs=epochs, lambda_ic=1.0, lambda_physics=1.0)

    history = trainer.train(
        t_domain=t_domain,
        t_ic=t_ic,
        x_ic=x_ic,
        verbose=True
    )
    print()

    print("6. Making predictions...")
    t_test = torch.linspace(0, t_max, 500).unsqueeze(1)
    t_test.requires_grad = True

    with torch.no_grad():
        x_pred = trainer.predict(t_test)

    t_test_np = t_test.detach().squeeze().numpy()
    print(f"   Predicted {len(x_pred)} points")
    print()

    print("7. Visualizing results...")
    visualizer = Visualizer()

    visualizer.plot_training_history(history, f"{output_dir}/training_history.png")
    print("   Saved training history to output/training_history.png")

    visualizer.plot_prediction(
        t_test_np, x_pred, x_analytical, omega,
        f"{output_dir}/prediction.png"
    )
    print("   Saved prediction plot to output/prediction.png")

    visualizer.plot_residual(
        t_test_np, x_pred, omega,
        f"{output_dir}/residual.png"
    )
    print("   Saved residual plot to output/residual.png")

    visualizer.plot_phase_space(
        t_test_np, x_pred, omega,
        f"{output_dir}/phase_space.png"
    )
    print("   Saved phase space plot to output/phase_space.png")

    visualizer.plot_all(
        t_test_np, x_pred, history, x_analytical, omega, output_dir
    )
    print("   Saved all plots to output directory")
    print()

    print("8. Calculating errors...")
    x_pred_interp = np.interp(t_eval, t_test_np, x_pred.flatten())

    print(f"   Sample analytical values (first 5): {x_analytical[:5]}")
    print(f"   Sample PINN values (first 5): {x_pred_interp[:5]}")

    rmse = np.sqrt(np.mean((x_pred_interp - x_analytical) ** 2))
    max_error = np.max(np.abs(x_pred_interp - x_analytical))

    print(f"   RMSE: {rmse:.6f}")
    print(f"   Max error: {max_error:.6f}")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Training completed in {epochs} epochs")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"RMSE vs analytical: {rmse:.6f}")
    print()
    print(f"All plots saved to: {output_dir}/")
    print("=" * 60)

    return rmse, max_error


def test_different_parameters():
    """Test the PINN with different parameter values."""
    print("\n" + "=" * 60)
    print("Testing with Different Parameters")
    print("=" * 60 + "\n")

    test_cases = [
        {"omega": 1.0, "t_max": 10.0, "hidden_dim": 64},
        {"omega": 2.0, "t_max": 5.0, "hidden_dim": 64},
        {"omega": 0.5, "t_max": 20.0, "hidden_dim": 64},
        {"omega": 3.0, "t_max": 3.0, "hidden_dim": 96},
    ]

    results = []
    for i, params in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"omega={params['omega']}, t_max={params['t_max']}, hidden_dim={params['hidden_dim']}")
        rmse, max_err = run_example(**params)
        results.append({
            "omega": params["omega"],
            "t_max": params["t_max"],
            "rmse": rmse,
            "max_err": max_err
        })

    print("\n" + "=" * 60)
    print("Summary of All Tests")
    print("=" * 60)
    for r in results:
        status = "OK" if r["rmse"] < 0.1 else "FAIL"
        print(f"ω={r['omega']:4.1f}: RMSE={r['rmse']:.6f}, MaxErr={r['max_err']:.6f} [{status}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-all":
        test_different_parameters()
    else:
        run_example()
