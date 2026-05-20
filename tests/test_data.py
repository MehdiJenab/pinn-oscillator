"""Tests for data generation module."""
import numpy as np
import pytest

from pinn_harmonic_oscillator.core.physics import HarmonicOscillator
from pinn_harmonic_oscillator.data.data_generator import (
    generate_initial_conditions,
    generate_boundary_points,
    generate_domain_samples,
)


class TestGenerateInitialConditions:
    """Test cases for initial condition generation."""

    def test_generate_initial_conditions_single(self):
        """Test generating a single initial condition."""
        oscillator = HarmonicOscillator()
        x0_range = (-1.0, 1.0)
        v0_range = (-0.5, 0.5)

        x0, v0 = generate_initial_conditions(oscillator, 1, x0_range, v0_range)

        assert x0.shape == (1,)
        assert v0.shape == (1,)
        assert x0[0] >= x0_range[0] and x0[0] <= x0_range[1]
        assert v0[0] >= v0_range[0] and v0[0] <= v0_range[1]

    def test_generate_initial_conditions_multiple(self):
        """Test generating multiple initial conditions."""
        oscillator = HarmonicOscillator()
        x0_range = (-2.0, 2.0)
        v0_range = (-1.0, 1.0)

        x0, v0 = generate_initial_conditions(oscillator, 100, x0_range, v0_range)

        assert x0.shape == (100,)
        assert v0.shape == (100,)
        assert np.all(x0 >= x0_range[0]) and np.all(x0 <= x0_range[1])
        assert np.all(v0 >= v0_range[0]) and np.all(v0 <= v0_range[1])

    def test_generate_initial_conditions_default(self):
        """Test with default ranges."""
        oscillator = HarmonicOscillator()

        x0, v0 = generate_initial_conditions(oscillator, 10)

        assert x0.shape == (10,)
        assert v0.shape == (10,)


class TestGenerateBoundaryPoints:
    """Test cases for boundary point generation."""

    def test_generate_boundary_points_shape(self):
        """Test that boundary points have correct shape."""
        t_min, t_max = 0.0, 5.0
        n_boundary = 20

        t_b, x_b = generate_boundary_points(t_min, t_max, n_boundary)

        assert t_b.shape == (n_boundary,)
        assert x_b.shape == (n_boundary,)

    def test_generate_boundary_points_range(self):
        """Test that boundary points are in correct range."""
        t_min, t_max = 1.0, 10.0
        n_boundary = 50

        t_b, x_b = generate_boundary_points(t_min, t_max, n_boundary)

        assert np.all(t_b >= t_min) and np.all(t_b <= t_max)
        assert np.all(x_b >= -1.1) and np.all(x_b <= 1.1)  # Default x_range


class TestGenerateDomainSamples:
    """Test cases for domain sampling."""

    def test_generate_domain_samples_shape(self):
        """Test that domain samples have correct shape."""
        t_min, t_max = 0.0, 10.0
        n_samples = 1000

        t_f = generate_domain_samples(t_min, t_max, n_samples)

        assert t_f.shape == (n_samples,)

    def test_generate_domain_points_range(self):
        """Test that domain points are in correct range."""
        t_min, t_max = -5.0, 15.0
        n_samples = 200

        t_f = generate_domain_samples(t_min, t_max, n_samples)

        assert np.all(t_f >= t_min) and np.all(t_f <= t_max)