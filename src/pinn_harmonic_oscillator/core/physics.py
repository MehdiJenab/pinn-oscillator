"""Physics module for harmonic oscillator modeling."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class HarmonicOscillator:
    """Harmonic oscillator model representing a mass-spring system.

    The equation of motion is: m * d²x/dt² + k * x = 0
    where m is mass, k is spring constant, and x is displacement.
    """

    mass: float = 1.0
    k_spring: float = 1.0

    @property
    def angular_freq(self) -> float:
        """Angular frequency ω = sqrt(k/m)."""
        return np.sqrt(self.k_spring / self.mass)

    def force(self, x: float) -> float:
        """Calculate spring force F = -k * x."""
        return -self.k_spring * x


def solve_ode(
    oscillator: HarmonicOscillator,
    t_span: Tuple[float, float],
    y0: Tuple[float, float],
    t_eval: np.ndarray,
) -> object:
    """Solve the harmonic oscillator ODE using scipy's solve_ivp.

    The second-order ODE m * x'' + k * x = 0 is rewritten as a system:
        x' = v
        v' = -(k/m) * x = -ω² * x

    Args:
        oscillator: HarmonicOscillator instance
        t_span: Time interval (t_start, t_end)
        y0: Initial conditions [x0, v0] (position, velocity)
        t_eval: Time points at which to store solution

    Returns:
        Solution object with attributes t and y
    """
    omega_sq = oscillator.angular_freq**2

    def system(t: float, y: np.ndarray) -> np.ndarray:
        """System of first-order ODEs."""
        x, v = y
        dxdt = v
        dvdt = -omega_sq * x
        return np.array([dxdt, dvdt])

    sol = solve_ivp(
        fun=system,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=False,
    )

    return sol