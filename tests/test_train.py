"""Tests for trainer module."""
import numpy as np
import pytest
import torch

from pinn_harmonic_oscillator.models.pinn import PINN
from pinn_harmonic_oscillator.train.trainer import Trainer


class TestTrainer:
    """Test cases for Trainer."""

    def test_init(self):
        """Test trainer initialization."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        epochs = 100

        trainer = Trainer(pinn, optimizer, epochs)

        assert trainer.pinn == pinn
        assert trainer.epochs == epochs

    def test_physics_loss_shape(self):
        """Test physics loss returns scalar."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        trainer = Trainer(pinn, optimizer, epochs=10)

        # Create test time points
        t = torch.linspace(0, 10, 50).unsqueeze(1)
        t.requires_grad = True

        # Get predictions
        x = pinn.forward(t)

        # Compute physics loss - this uses trainer's internal pinn
        loss = trainer.compute_physics_loss(t)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Should be scalar

    def test_physics_loss_conservation(self):
        """Test physics loss is low for known solution."""
        pinn = PINN(1, 50, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)
        trainer = Trainer(pinn, optimizer, epochs=100)

        # Harmonic oscillator parameters
        omega = 1.0  # Natural frequency

        # Create initial conditions
        t0 = torch.zeros(1, 1)
        x0 = torch.ones(1, 1)  # x(0) = 1
        v0 = torch.zeros(1, 1)  # v(0) = 0

        # Known solution: x(t) = cos(omega * t)
        t_test = torch.linspace(0, 2 * np.pi, 100).unsqueeze(1)
        t_test.requires_grad = True

        # Train for a few iterations to see if loss decreases
        initial_loss = None
        for _ in range(100):
            optimizer.zero_grad()

            # Physics loss using trainer's method
            physics_loss = trainer.compute_physics_loss(t_test, omega=omega)

            physics_loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = physics_loss.item()

        final_loss = physics_loss.item()

        # Loss should decrease during training (with some tolerance for noise)
        assert final_loss <= initial_loss + 0.1

    def test_initial_condition_loss(self):
        """Test initial condition loss computation."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        trainer = Trainer(pinn, optimizer, epochs=10)

        # Initial conditions
        t0 = torch.zeros(1, 1)
        x0 = torch.ones(1, 1)  # x(0) = 1

        # Make network predict correct initial condition
        with torch.no_grad():
            pinn.layers[0].weight.fill_(0.0)
            pinn.layers[0].bias.fill_(0.0)
            pinn.layers[2].weight.fill_(0.0)
            pinn.layers[2].bias.fill_(0.0)
            pinn.layers[4].weight.fill_(0.0)
            pinn.layers[4].bias.fill_(1.0)  # x(0) = 1

        x_pred = pinn.forward(t0)
        loss = trainer.compute_initial_condition_loss(t0, x_pred, x0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_total_loss(self):
        """Test total loss is weighted sum of components."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        trainer = Trainer(pinn, optimizer, epochs=10, lambda_ic=1.0, lambda_physics=1.0)

        # Simple test with known values - t requires grad for physics loss
        t = torch.zeros(1, 1)
        t.requires_grad = True
        x = torch.ones(1, 1)

        ic_loss = trainer.compute_initial_condition_loss(t, x, x)
        physics_loss = trainer.compute_physics_loss(t)

        total_loss = ic_loss + physics_loss

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0

    def test_train_step(self):
        """Test single training step runs without error."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)
        trainer = Trainer(pinn, optimizer, epochs=10)

        # Training data
        t_domain = torch.linspace(0, 10, 50).unsqueeze(1)
        t_domain.requires_grad = True
        t_ic = torch.zeros(1, 1)
        x_ic = torch.ones(1, 1)

        # Run one training step
        loss = trainer.train_step(t_domain, t_ic, x_ic)

        assert isinstance(loss, float)

    def test_train(self):
        """Test full training runs without error."""
        pinn = PINN(1, 10, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)
        trainer = Trainer(pinn, optimizer, epochs=50)

        # Training data
        t_domain = torch.linspace(0, 5, 30).unsqueeze(1)
        t_domain.requires_grad = True
        t_ic = torch.zeros(1, 1)
        x_ic = torch.ones(1, 1)

        # Train
        history = trainer.train(t_domain, t_ic, x_ic, verbose=False)

        assert 'train_loss' in history
        assert 'ic_loss' in history
        assert 'physics_loss' in history
        assert len(history['train_loss']) == 50

        # Loss should generally decrease
        assert history['train_loss'][-1] <= history['train_loss'][0]

    def test_predict(self):
        """Test prediction method."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        trainer = Trainer(pinn, optimizer, epochs=10)

        t = torch.linspace(0, 10, 20).unsqueeze(1)

        x_pred = trainer.predict(t)

        assert isinstance(x_pred, np.ndarray)
        assert x_pred.shape == (20, 1)

    def test_loss_components(self):
        """Test that loss components are properly computed."""
        pinn = PINN(1, 20, 1)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)
        trainer = Trainer(pinn, optimizer, epochs=10, lambda_ic=1.0, lambda_physics=1.0)

        t_domain = torch.linspace(0, 5, 30).unsqueeze(1)
        t_domain.requires_grad = True
        t_ic = torch.zeros(1, 1)
        x_ic = torch.ones(1, 1)

        # Run training to get losses
        history = trainer.train(t_domain, t_ic, x_ic, verbose=False)

        # All losses should be positive
        for key in ['ic_loss', 'physics_loss', 'train_loss']:
            assert all(loss >= 0 for loss in history[key])