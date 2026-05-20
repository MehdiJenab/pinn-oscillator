"""Tests for the visualizer module."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import pytest
import numpy as np
import torch
from matplotlib import pyplot as plt

from pinn_harmonic_oscillator.visualize.visualizer import Visualizer
from pinn_harmonic_oscillator.models.pinn import PINN
from pinn_harmonic_oscillator.train.trainer import Trainer
from pinn_harmonic_oscillator.data.data_generator import generate_initial_conditions, generate_domain_samples


class TestVisualizer:
    """Test cases for the Visualizer class."""

    @pytest.fixture
    def visualizer(self):
        """Create a Visualizer instance."""
        return Visualizer()

    @pytest.fixture
    def trained_pinn(self):
        """Create a trained PINN for testing visualization."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)
        trainer = Trainer(pinn, optimizer, epochs=500)

        # Generate training data
        t_domain = torch.linspace(0, 2 * np.pi, 50).unsqueeze(1)
        t_domain.requires_grad = True
        t_ic = torch.zeros(1, 1)
        x_ic = torch.ones(1, 1)

        # Train
        trainer.train(t_domain, t_ic, x_ic, verbose=False)

        return pinn

    def test_init(self, visualizer):
        """Test that Visualizer initializes correctly."""
        assert visualizer is not None

    def test_plot_training_history(self, visualizer):
        """Test training history plotting."""
        history = {
            'train_loss': [0.1, 0.05, 0.02, 0.01, 0.005],
            'ic_loss': [0.05, 0.02, 0.01, 0.005, 0.002],
            'physics_loss': [0.05, 0.03, 0.01, 0.005, 0.003],
        }

        fig = visualizer.plot_training_history(history)

        assert fig is not None
        plt.close(fig)

    def test_plot_prediction(self, visualizer, trained_pinn):
        """Test prediction plotting."""
        t = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
        t_torch = torch.tensor(t, dtype=torch.float32)

        trained_pinn.eval()
        with torch.no_grad():
            x_pred = trained_pinn.forward(t_torch).numpy()

        # Analytical solution: x(t) = cos(t) for ω=1, x(0)=1, v(0)=0
        x_true = np.cos(t)

        fig = visualizer.plot_prediction(t, x_pred, x_true, omega=1.0)

        assert fig is not None
        plt.close(fig)

    def test_plot_residual(self, visualizer, trained_pinn):
        """Test residual plotting."""
        t = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
        t_torch = torch.tensor(t, dtype=torch.float32)

        trained_pinn.eval()
        with torch.no_grad():
            x_pred = trained_pinn.forward(t_torch).numpy()

        fig = visualizer.plot_residual(t, x_pred, omega=1.0)

        assert fig is not None
        plt.close(fig)

    def test_plot_phase_space(self, visualizer, trained_pinn):
        """Test phase space plotting."""
        t = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
        t_torch = torch.tensor(t, dtype=torch.float32)

        trained_pinn.eval()
        with torch.no_grad():
            x_pred = trained_pinn.forward(t_torch).numpy()

        fig = visualizer.plot_phase_space(t, x_pred, omega=1.0)

        assert fig is not None
        plt.close(fig)

    def test_plot_all(self, visualizer, trained_pinn):
        """Test plotting all visualizations."""
        t = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
        t_torch = torch.tensor(t, dtype=torch.float32)

        # Get predictions
        trained_pinn.eval()
        with torch.no_grad():
            x_pred = trained_pinn.forward(t_torch).numpy()

        # Get training history
        optimizer = torch.optim.Adam(trained_pinn.parameters(), lr=0.01)
        trainer = Trainer(trained_pinn, optimizer, epochs=100)
        history = trainer.train(t_torch, torch.zeros(1, 1), torch.ones(1, 1), verbose=False)

        # Test without save_dir (just display)
        visualizer.plot_all(t, x_pred, history, x_true=np.cos(t), omega=1.0)

        # Test with save_dir
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer.plot_all(t, x_pred, history, x_true=np.cos(t), omega=1.0, save_dir=tmpdir)

            # Check that files were created
            assert os.path.exists(f"{tmpdir}/training_history.png")
            assert os.path.exists(f"{tmpdir}/prediction.png")
            assert os.path.exists(f"{tmpdir}/residual.png")
            assert os.path.exists(f"{tmpdir}/phase_space.png")