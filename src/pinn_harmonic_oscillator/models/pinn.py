"""PINN (Physics-Informed Neural Network) model module."""
import torch
import torch.nn as nn


class PINN(nn.Module):
    """Physics-Informed Neural Network for solving ODEs.

    This network takes time (and optionally initial conditions) as input
    and outputs the solution x(t) at those time points.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize PINN.

        Args:
            input_dim: Dimension of input (e.g., 1 for just time, 3 for time + ICs)
            hidden_dim: Number of hidden units in each hidden layer
            output_dim: Dimension of output (e.g., 1 for scalar position)
        """
        super(PINN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define network layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            t: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.layers(t)

    def predict(self, t: torch.Tensor) -> torch.Tensor:
        """Make prediction without gradient tracking.

        Args:
            t: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        with torch.no_grad():
            return self.forward(t)

    def get_param_count(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)