"""Trainer module for PINN."""
import numpy as np
import torch
import torch.nn as nn

from pinn_harmonic_oscillator.models.pinn import PINN


class Trainer:
    """Trainer for Physics-Informed Neural Network.

    This class handles the training loop, physics-informed loss computation,
    and model optimization for solving ODEs using PINNs.
    """

    def __init__(
        self,
        pinn: PINN,
        optimizer: torch.optim.Optimizer,
        epochs: int = 1000,
        lambda_ic: float = 1.0,
        lambda_physics: float = 1.0,
    ):
        """Initialize Trainer.

        Args:
            pinn: The PINN model to train
            optimizer: PyTorch optimizer for training
            epochs: Number of training epochs
            lambda_ic: Weight for initial condition loss
            lambda_physics: Weight for physics loss
        """
        self.pinn = pinn
        self.optimizer = optimizer
        self.epochs = epochs
        self.lambda_ic = lambda_ic
        self.lambda_physics = lambda_physics

    def compute_physics_loss(
        self, t: torch.Tensor, x: torch.Tensor = None, omega: float = 1.0
    ) -> torch.Tensor:
        """Compute physics-informed loss (ODE residual).

        For harmonic oscillator: d²x/dt² + ω²x = 0

        Args:
            t: Time points (requires_grad=True)
            x: Position predictions (optional, computed from pinn if not provided)
            omega: Natural frequency of oscillator

        Returns:
            Physics loss (scalar)
        """
        # Ensure t requires grad
        if not t.requires_grad:
            t = t.detach().clone().requires_grad_(True)

        # Get x predictions - use provided x or compute from pinn
        if x is not None:
            # Ensure x is from t with grad tracking
            if x.grad_fn is None:
                x_pred = x
            else:
                x_pred = x
        else:
            x_pred = self.pinn.forward(t)

        # Compute first derivative: dx/dt
        dx_dt = torch.autograd.grad(
            x_pred.sum(), t, create_graph=True, retain_graph=True
        )[0]

        # Compute second derivative: d²x/dt²
        d2x_dt2 = torch.autograd.grad(
            dx_dt.sum(), t, create_graph=True
        )[0]

        # ODE residual: d²x/dt² + ω²x = 0
        ode_residual = d2x_dt2 + omega**2 * x_pred

        # Mean squared residual
        physics_loss = (ode_residual ** 2).mean()

        return physics_loss

    def compute_initial_condition_loss(
        self, t0: torch.Tensor, x0_pred: torch.Tensor, x0_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for initial conditions.

        Args:
            t0: Initial time points
            x0_pred: Predicted initial positions
            x0_true: True initial positions

        Returns:
            Initial condition loss (scalar)
        """
        return nn.MSELoss()(x0_pred, x0_true)

    def train_step(
        self, t_domain: torch.Tensor, t_ic: torch.Tensor, x_ic: torch.Tensor
    ) -> float:
        """Perform one training step.

        Args:
            t_domain: Time points for domain sampling
            t_ic: Initial time for IC
            x_ic: Initial position for IC

        Returns:
            Total loss for this step
        """
        self.optimizer.zero_grad()

        # Ensure t_domain requires grad for physics loss computation
        if not t_domain.requires_grad:
            t_domain = t_domain.detach().clone().requires_grad_(True)

        # Predictions for domain
        x_domain = self.pinn.forward(t_domain)

        # Compute losses
        physics_loss = self.compute_physics_loss(t_domain, x_domain)
        ic_loss = self.compute_initial_condition_loss(t_ic, self.pinn.forward(t_ic), x_ic)

        # Total loss
        total_loss = self.lambda_ic * ic_loss + self.lambda_physics * physics_loss

        # Backprop
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train(
        self,
        t_domain: torch.Tensor,
        t_ic: torch.Tensor,
        x_ic: torch.Tensor,
        verbose: bool = True,
    ) -> dict:
        """Train the PINN.

        Args:
            t_domain: Time points for domain sampling
            t_ic: Initial time for IC
            x_ic: Initial position for IC
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'ic_loss': [],
            'physics_loss': [],
        }

        for epoch in range(self.epochs):
            loss = self.train_step(t_domain, t_ic, x_ic)

            # Compute individual losses for logging
            # Note: physics loss needs gradient tracking for derivative computation
            physics_loss = self.compute_physics_loss(t_domain).item()
            with torch.no_grad():
                ic_loss = self.compute_initial_condition_loss(
                    t_ic, self.pinn.forward(t_ic), x_ic
                ).item()

            history['train_loss'].append(loss)
            history['ic_loss'].append(ic_loss)
            history['physics_loss'].append(physics_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {loss:.6f} - "
                      f"IC Loss: {ic_loss:.6f} - "
                      f"Physics Loss: {physics_loss:.6f}")

        return history

    def predict(self, t: torch.Tensor) -> np.ndarray:
        """Make predictions.

        Args:
            t: Time points

        Returns:
            Predicted positions as numpy array
        """
        self.pinn.eval()
        with torch.no_grad():
            x = self.pinn.forward(t)
        return x.numpy()