"""PINN (Physics-Informed Neural Network) model module."""
import torch
import torch.nn as nn


class SIRENLayer(nn.Module):
    """SIREN layer for periodic function approximation."""

    def __init__(self, in_features: int, out_features: int, is_first: bool = False,
                 last_layer: bool = False, omega_0: float = 30.0):
        """Initialize SIREN layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            is_first: Whether this is the first layer
            last_layer: Whether this is the last layer
            omega_0: Frequency factor for first layer
        """
        super(SIRENLayer, self).__init__()
        self.is_first = is_first
        self.last_layer = last_layer
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sinusoidal activation."""
        x = self.linear(x)
        # All SIREN layers use sinusoidal activation
        # First layer has frequency scaling by omega_0
        if self.is_first:
            return torch.sin(self.omega_0 * x)
        return torch.sin(x)

    def extra_repr(self) -> str:
        return f'in_features={self.linear.in_features}, out_features={self.linear.out_features}'


def siren_init_weights(m):
    """SIREN-style weight initialization for periodic function approximation.

    This initialization helps networks learn high-frequency functions better.
    """
    if isinstance(m, nn.Linear):
        fan_in = m.weight.shape[1]
        # Check if this layer has is_first attribute (SIRENLayer)
        if hasattr(m, 'is_first') and m.is_first:
            # First layer: use frequency-scaled initialization
            # w_std = omega_0 / sqrt(fan_in) for proper frequency scaling
            dim = fan_in
            w_std = m.omega_0 / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        else:
            # Subsequent layers: use Xavier/Glorot-style initialization
            # scale by 1/omega_0 of the layer (which is 1.0 for non-first layers)
            dim = fan_in
            omega = getattr(m, 'omega_0', 1.0)
            w_std = torch.sqrt(torch.tensor(2.0 / dim)) / omega
        m.weight.data.uniform_(-w_std, w_std)
        if m.bias is not None:
            m.bias.data.zero_()


class PeriodicLayer(nn.Module):
    """Simple periodic layer using sin activation with learnable frequency."""

    def __init__(self, in_features: int, out_features: int, frequency: float = 1.0):
        """Initialize periodic layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            frequency: Initial frequency for sin activation
        """
        super(PeriodicLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.frequency = nn.Parameter(torch.tensor(frequency))
        self.linear = nn.Linear(in_features, out_features)
        # Initialize weights for sin(w*x) where w is the frequency
        self.linear.weight.data.uniform_(-1.0, 1.0)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learnable frequency sin activation."""
        return torch.sin(self.frequency * self.linear(x))

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, frequency={self.frequency.item():.2f}'


def init_periodic_weights(m):
    """Initialize periodic layer weights."""
    if isinstance(m, PeriodicLayer):
        # Xavier initialization for the linear layer
        fan_in = m.linear.weight.shape[1]
        fan_out = m.linear.weight.shape[0]
        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        m.linear.weight.data.uniform_(-std, std)
        if m.linear.bias is not None:
            m.linear.bias.data.zero_()


class PINN(nn.Module):
    """Physics-Informed Neural Network for solving ODEs.

    This network takes time as input and outputs the solution x(t).
    Uses a simple periodic architecture that learns cos(omega*t) directly.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, omega: float = 1.0):
        """Initialize PINN.

        Args:
            input_dim: Dimension of input (e.g., 1 for just time)
            hidden_dim: Number of hidden units in each layer
            output_dim: Dimension of output (e.g., 1 for scalar position)
            omega: Initial guess for angular frequency
        """
        super(PINN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Learnable frequency parameter - initialized to 1.0
        # The network learns: x(t) = a * cos(omega * t + phase) + b
        self.log_omega = nn.Parameter(torch.log(torch.tensor(omega)))
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.log_amplitude = nn.Parameter(torch.tensor(0.0))  # log(1) = 0
        self.bias = nn.Parameter(torch.tensor(0.0))

        # Also use a small neural network to learn corrections
        self.correction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize correction network
        for layer in self.correction_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()

    def _compute_forward(self, t: torch.Tensor) -> torch.Tensor:
        """Internal forward pass through the network.

        Args:
            t: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = t.shape[0]

        # Main periodic component: x(t) = A * cos(omega * t + phase) + B
        # Use the first column as time
        omega = torch.exp(self.log_omega)
        t_main = t[:, 0]  # Extract first column as time values
        periodic = torch.exp(self.log_amplitude) * torch.cos(omega * t_main + self.phase) + self.bias

        # Add neural network correction using full input
        correction = self.correction_net(t).squeeze(-1)

        result = periodic + correction

        return result.unsqueeze(-1)  # Add output dimension back

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            t: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self._compute_forward(t)

    def predict(self, t: torch.Tensor) -> torch.Tensor:
        """Make prediction without gradient tracking.

        Args:
            t: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        with torch.no_grad():
            result = self.forward(t)
            return result.detach()

    def get_param_count(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def layers(self) -> list:
        """Return list of layers for compatibility with tests.

        Returns:
            List of layer modules in the correction network
        """
        return list(self.correction_net.children())

    def get_omega(self) -> float:
        """Get current angular frequency estimate.

        Returns:
            Angular frequency
        """
        return torch.exp(self.log_omega).item()