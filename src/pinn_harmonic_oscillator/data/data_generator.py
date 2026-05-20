"""Data generation module for PINN training."""
import numpy as np


def generate_initial_conditions(
    oscillator,
    n_samples: int,
    x0_range: tuple = (-1.0, 1.0),
    v0_range: tuple = (-0.5, 0.5),
) -> tuple:
    """Generate random initial conditions for the harmonic oscillator.

    Args:
        oscillator: HarmonicOscillator instance (not used currently, for API consistency)
        n_samples: Number of initial conditions to generate
        x0_range: Range of initial positions (min, max)
        v0_range: Range of initial velocities (min, max)

    Returns:
        Tuple of (x0, v0) arrays of shape (n_samples,)
    """
    x0 = np.random.uniform(x0_range[0], x0_range[1], n_samples)
    v0 = np.random.uniform(v0_range[0], v0_range[1], n_samples)
    return x0, v0


def generate_boundary_points(
    t_min: float,
    t_max: float,
    n_boundary: int,
    x_range: tuple = (-1.1, 1.1),
) -> tuple:
    """Generate boundary points for PINN training.

    Boundary points are sampled at t = t_min and t = t_max to enforce
    initial and final conditions.

    Args:
        t_min: Minimum time
        t_max: Maximum time
        n_boundary: Number of boundary points (half at each boundary)
        x_range: Range of position values (min, max)

    Returns:
        Tuple of (t_b, x_b) arrays
    """
    # Points at t = t_min (initial boundary)
    n_half = n_boundary // 2
    t_min_points = np.full(n_half, t_min)
    x_min_points = np.random.uniform(x_range[0], x_range[1], n_half)

    # Points at t = t_max (final boundary)
    t_max_points = np.full(n_half, t_max)
    x_max_points = np.random.uniform(x_range[0], x_range[1], n_half)

    t_b = np.concatenate([t_min_points, t_max_points])
    x_b = np.concatenate([x_min_points, x_max_points])

    return t_b, x_b


def generate_domain_samples(
    t_min: float,
    t_max: float,
    n_samples: int,
) -> np.ndarray:
    """Generate collocation points inside the domain.

    These points are used to enforce the physics (differential equation)
    throughout the domain.

    Args:
        t_min: Minimum time
        t_max: Maximum time
        n_samples: Number of domain samples

    Returns:
        Array of time points of shape (n_samples,)
    """
    t_f = np.random.uniform(t_min, t_max, n_samples)
    return t_f