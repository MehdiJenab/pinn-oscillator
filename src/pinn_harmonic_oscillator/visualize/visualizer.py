"""Visualizer module for PINN results."""
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Visualizer for PINN training and predictions.

    This class provides methods to visualize:
    - Training loss curves
    - PINN predictions vs analytical solutions
    - Residual errors
    - Phase space trajectories
    """

    def __init__(self):
        """Initialize Visualizer."""
        pass

    def plot_training_history(self, history: dict, save_path: str = None) -> plt.Figure:
        """Plot training loss history.

        Args:
            history: Dictionary with 'train_loss', 'ic_loss', 'physics_loss' keys
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Total loss
        ax1.plot(history['train_loss'], label='Total Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)

        # Individual losses (log scale)
        if 'ic_loss' in history and 'physics_loss' in history:
            ax2.plot(history['ic_loss'], label='IC Loss', alpha=0.7)
            ax2.plot(history['physics_loss'], label='Physics Loss', alpha=0.7)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Individual Losses')
            ax2.legend()
            ax2.grid(True)
            ax2.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_prediction(
        self,
        t: np.ndarray,
        x_pred: np.ndarray,
        x_true: np.ndarray = None,
        omega: float = 1.0,
        save_path: str = None
    ) -> plt.Figure:
        """Plot PINN prediction vs analytical solution.

        Args:
            t: Time points
            x_pred: Predicted positions
            x_true: True analytical positions (optional)
            omega: Natural frequency
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot prediction
        ax.plot(t, x_pred, 'b-', linewidth=2, label='PINN Prediction')

        # Plot analytical solution if provided
        if x_true is not None:
            ax.plot(t, x_true, 'r--', linewidth=2, label='Analytical Solution')

        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Position (x)')
        ax.set_title(f'Harmonic Oscillator Solution (ω = {omega})')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_residual(
        self,
        t: np.ndarray,
        x_pred: np.ndarray,
        omega: float = 1.0,
        save_path: str = None
    ) -> plt.Figure:
        """Plot residual error of the ODE.

        Args:
            t: Time points
            x_pred: Predicted positions
            omega: Natural frequency
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Compute derivatives using finite differences
        t_flat = t.flatten() if t.ndim > 1 else t
        dt = np.diff(t_flat).mean()  # Use average spacing
        # Flatten arrays to 1D for proper gradient computation
        x_pred_flat = x_pred.flatten()
        dx_dt = np.gradient(x_pred_flat, dt)
        d2x_dt2 = np.gradient(dx_dt, dt)

        # ODE residual: d²x/dt² + ω²x
        residual = d2x_dt2 + omega**2 * x_pred_flat

        # Reshape back to original shape for plotting
        residual = residual.reshape(x_pred.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, residual, 'g-', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Residual (d²x/dt² + ω²x)')
        ax.set_title('ODE Residual Error')
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_phase_space(
        self,
        t: np.ndarray,
        x_pred: np.ndarray,
        omega: float = 1.0,
        save_path: str = None
    ) -> plt.Figure:
        """Plot phase space trajectory (x vs dx/dt).

        Args:
            t: Time points
            x_pred: Predicted positions
            omega: Natural frequency
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Compute velocity using finite differences
        t_flat = t.flatten() if t.ndim > 1 else t
        dt = np.diff(t_flat).mean()  # Use average spacing
        x_pred_flat = x_pred.flatten()
        dx_dt_flat = np.gradient(x_pred_flat, dt)
        dx_dt = dx_dt_flat.reshape(x_pred.shape)

        # Analytical phase space (ellipse for undamped oscillator)
        # x = A*cos(ωt), v = -A*ω*sin(ωt)
        # So x² + (v/ω)² = A² (ellipse)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_pred, dx_dt, 'b-', linewidth=2, label='PINN Prediction')

        # Plot analytical phase space
        max_x = np.max(np.abs(x_pred))
        max_v = np.max(np.abs(dx_dt))
        t_analytical = np.linspace(0, 2 * np.pi / omega, 100)
        x_analytical = max_x * np.cos(omega * t_analytical)
        v_analytical = -max_x * omega * np.sin(omega * t_analytical)
        ax.plot(x_analytical, v_analytical, 'r--', linewidth=2, label='Analytical')

        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Velocity (dx/dt)')
        ax.set_title('Phase Space Trajectory')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_all(
        self,
        t: np.ndarray,
        x_pred: np.ndarray,
        history: dict,
        x_true: np.ndarray = None,
        omega: float = 1.0,
        save_dir: str = None
    ) -> None:
        """Generate and optionally save all plots.

        Args:
            t: Time points
            x_pred: Predicted positions
            history: Training history dictionary
            x_true: True analytical positions (optional)
            omega: Natural frequency
            save_dir: Optional directory to save figures
        """
        # Training history
        if save_dir:
            self.plot_training_history(history, f"{save_dir}/training_history.png")
        else:
            self.plot_training_history(history)

        # Predictions
        if save_dir:
            self.plot_prediction(t, x_pred, x_true, omega, f"{save_dir}/prediction.png")
        else:
            self.plot_prediction(t, x_pred, x_true, omega)

        # Residual
        if save_dir:
            self.plot_residual(t, x_pred, omega, f"{save_dir}/residual.png")
        else:
            self.plot_residual(t, x_pred, omega)

        # Phase space
        if save_dir:
            self.plot_phase_space(t, x_pred, omega, f"{save_dir}/phase_space.png")
        else:
            self.plot_phase_space(t, x_pred, omega)