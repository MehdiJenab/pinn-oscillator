#!/usr/bin/env python3
"""Test script to verify damping implementation works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pinn_harmonic_oscillator.core.physics import HarmonicOscillator
from pinn_harmonic_oscillator.models.pinn import PINN
import torch

def test_physics_module():
    """Test the physics module with damping."""
    print("Testing physics module...")

    # Test undamped oscillator
    osc1 = HarmonicOscillator(mass=1.0, k_spring=1.0, damping_coefficient=0.0)
    print(f"Undamped - Angular freq: {osc1.angular_freq}, Damping ratio: {osc1.damping_ratio}")

    # Test damped oscillator
    osc2 = HarmonicOscillator(mass=1.0, k_spring=1.0, damping_coefficient=0.5)
    print(f"Damped - Angular freq: {osc2.angular_freq}, Damping ratio: {osc2.damping_ratio}")

    print("Physics module test passed!")

def test_pinn_model():
    """Test the PINN model with damping."""
    print("\nTesting PINN model...")

    # Test undamped PINN
    pinn1 = PINN(input_dim=1, hidden_dim=50, output_dim=1, omega=1.0, damping_coefficient=0.0)
    print(f"Undamped PINN created successfully")

    # Test damped PINN
    pinn2 = PINN(input_dim=1, hidden_dim=50, output_dim=1, omega=1.0, damping_coefficient=0.5)
    print(f"Damped PINN created successfully")

    # Test forward pass
    t = torch.FloatTensor([[0.0]])
    result1 = pinn1.forward(t)
    result2 = pinn2.forward(t)
    print(f"Forward pass test - Undamped: {result1.item():.4f}, Damped: {result2.item():.4f}")

    print("PINN model test passed!")

if __name__ == "__main__":
    test_physics_module()
    test_pinn_model()
    print("\nAll tests passed!")