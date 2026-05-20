"""Tests for PINN model module."""
import numpy as np
import pytest
import torch

from pinn_harmonic_oscillator.models.pinn import PINN


class TestPINN:
    """Test cases for PINN model."""

    def test_init(self):
        """Test PINN initialization."""
        input_dim = 1
        hidden_dim = 20
        output_dim = 1

        pinn = PINN(input_dim, hidden_dim, output_dim)

        assert pinn.input_dim == input_dim
        assert pinn.output_dim == output_dim
        assert len(pinn.layers) == 5  # input Linear, Tanh, hidden Linear, Tanh, output Linear

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 32
        input_dim = 1
        hidden_dim = 20
        output_dim = 1

        pinn = PINN(input_dim, hidden_dim, output_dim)

        t = torch.randn(batch_size, input_dim)
        x = pinn.forward(t)

        assert x.shape == (batch_size, output_dim)

    def test_forward_torch_tensor(self):
        """Test forward pass returns torch tensor."""
        pinn = PINN(1, 20, 1)
        t = torch.randn(10, 1)

        x = pinn.forward(t)

        assert isinstance(x, torch.Tensor)

    def test_forward_positive_output(self):
        """Test that forward can produce various outputs (not constrained)."""
        pinn = PINN(1, 20, 1)
        pinn.eval()

        t = torch.linspace(-10, 10, 20).unsqueeze(1)

        with torch.no_grad():
            x = pinn.forward(t)

        # Output should be continuous and not constant
        assert not torch.allclose(x[0], x[-1], atol=1e-3)

    def test_parameter_count(self):
        """Test that model has trainable parameters."""
        pinn = PINN(1, 20, 1)

        params = list(pinn.parameters())
        assert len(params) > 0

        # Check that parameters have gradients enabled
        for p in params:
            assert p.requires_grad

    def test_different_hidden_layers(self):
        """Test PINN with different hidden layer configurations."""
        for hidden_dim in [10, 50, 100]:
            pinn = PINN(1, hidden_dim, 1)
            t = torch.randn(5, 1)

            with torch.no_grad():
                x = pinn.forward(t)

            assert x.shape == (5, 1)

    def test_multi_input_dim(self):
        """Test PINN with multi-dimensional input (time + initial conditions)."""
        input_dim = 3  # t, x0, v0
        pinn = PINN(input_dim, 20, 1)
        t = torch.randn(10, input_dim)

        with torch.no_grad():
            x = pinn.forward(t)

        assert x.shape == (10, 1)