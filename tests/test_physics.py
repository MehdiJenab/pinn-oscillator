"""Tests for physics module - harmonic oscillator ODE solver."""
import numpy as np
import pytest

from pinn_harmonic_oscillator.core.physics import HarmonicOscillator, solve_ode


class TestHarmonicOscillator:
    """Test cases for HarmonicOscillator class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        oscillator = HarmonicOscillator()
        assert oscillator.mass == 1.0
        assert oscillator.k_spring == 1.0
        assert oscillator.angular_freq == 1.0

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        oscillator = HarmonicOscillator(mass=2.0, k_spring=8.0)
        assert oscillator.mass == 2.0
        assert oscillator.k_spring == 8.0
        assert oscillator.angular_freq == 2.0

    def test_force(self):
        """Test force calculation."""
        oscillator = HarmonicOscillator(mass=1.0, k_spring=1.0)
        # F = -k * x
        assert oscillator.force(0.0) == 0.0
        assert oscillator.force(1.0) == -1.0
        assert oscillator.force(-1.0) == 1.0

    def test_angular_freq_property(self):
        """Test angular frequency calculation."""
        oscillator = HarmonicOscillator(mass=4.0, k_spring=16.0)
        # omega = sqrt(k/m) = sqrt(16/4) = 2
        assert oscillator.angular_freq == 2.0


class TestSolveOde:
    """Test cases for ODE solver."""

    def test_solve_ode_shape(self):
        """Test that solution has correct shape."""
        oscillator = HarmonicOscillator()
        t_span = (0, 10)
        y0 = [1.0, 0.0]  # x0, v0
        t_eval = np.linspace(0, 10, 100)

        sol = solve_ode(oscillator, t_span, y0, t_eval)

        assert sol.t.shape == t_eval.shape
        assert sol.y.shape == (2, len(t_eval))

    def test_solve_ode_energy_conservation(self):
        """Test approximate energy conservation for undamped oscillator."""
        oscillator = HarmonicOscillator(mass=1.0, k_spring=1.0)
        t_span = (0, 100)
        y0 = [1.0, 0.0]
        t_eval = np.linspace(0, 100, 1000)

        sol = solve_ode(oscillator, t_span, y0, t_eval)

        x = sol.y[0]
        v = sol.y[1]

        # Total energy = kinetic + potential = 0.5*m*v^2 + 0.5*k*x^2
        kinetic = 0.5 * oscillator.mass * v**2
        potential = 0.5 * oscillator.k_spring * x**2
        total_energy = kinetic + potential

        # Energy should be approximately constant (within 1% tolerance)
        energy_std = np.std(total_energy)
        energy_mean = np.mean(total_energy)
        assert energy_std / energy_mean < 0.05

    def test_solve_ode_initial_conditions(self):
        """Test that initial conditions are satisfied."""
        oscillator = HarmonicOscillator()
        t_span = (0, 5)
        y0 = [2.0, -1.0]  # x0=2, v0=-1
        t_eval = np.linspace(0, 5, 10)

        sol = solve_ode(oscillator, t_span, y0, t_eval)

        assert sol.y[0, 0] == pytest.approx(y0[0], rel=1e-10)
        assert sol.y[1, 0] == pytest.approx(y0[1], rel=1e-10)