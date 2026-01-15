"""
Nonlinear encoder: β → (p, cos q, sin q)

Replaces linear W for systems where β doesn't linearly map to state.
The manifold diagnostics showed both SHO and Pendulum β-manifolds are flat 2D,
but the coordinate relationship is nonlinear for pendulum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BetaToStateEncoder(nn.Module):
    """
    Nonlinear encoder: β → (p, cos q, sin q)

    Replaces linear W for systems where β doesn't linearly map to state.
    Uses S¹-aware output (cos q, sin q) to handle periodic angle properly.
    """

    def __init__(self, n_beta=8, hidden_dim=32, n_layers=2):
        super().__init__()

        self.n_beta = n_beta

        layers = []
        in_dim = n_beta
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh()
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))  # Output: (p, cos q, sin q)

        self.net = nn.Sequential(*layers)

        # Initialize small weights for stable training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, beta):
        """
        Args:
            beta: (batch, n_beta) or (n_beta,)
        Returns:
            p: (batch,)
            cos_q: (batch,)
            sin_q: (batch,)
        """
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)

        out = self.net(beta)
        return out[:, 0], out[:, 1], out[:, 2]

    def forward_pcs(self, beta):
        """Return stacked (p, cos_q, sin_q) tensor."""
        p, cos_q, sin_q = self.forward(beta)
        return torch.stack([p, cos_q, sin_q], dim=1)

    def get_pq(self, beta):
        """Convenience: return (p, q) instead of (p, cos q, sin q)."""
        p, cos_q, sin_q = self.forward(beta)
        q = torch.atan2(sin_q, cos_q)
        return p, q


class BetaToStateEncoderWithUncertainty(nn.Module):
    """
    Encoder with uncertainty estimation: β → (p, cos q, sin q, σ_p, σ_q)

    Outputs both predictions and uncertainty estimates.
    Useful for downstream tasks that need confidence.
    """

    def __init__(self, n_beta=8, hidden_dim=32, n_layers=2):
        super().__init__()

        self.n_beta = n_beta

        # Shared trunk
        trunk_layers = []
        in_dim = n_beta
        for i in range(n_layers - 1):
            trunk_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh()
            ])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # Mean head: (p, cos q, sin q)
        self.mean_head = nn.Linear(hidden_dim, 3)

        # Log-variance head: (log σ_p², log σ_cosq², log σ_sinq²)
        self.logvar_head = nn.Linear(hidden_dim, 3)

        # Initialize
        for m in [self.trunk, self.mean_head, self.logvar_head]:
            if isinstance(m, nn.Sequential):
                for layer in m.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)
                        nn.init.zeros_(layer.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, beta):
        """
        Returns:
            mean: (batch, 3) - (p, cos q, sin q)
            logvar: (batch, 3) - log variances
        """
        if beta.dim() == 1:
            beta = beta.unsqueeze(0)

        h = self.trunk(beta)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        return mean, logvar

    def get_pq_with_uncertainty(self, beta):
        """Return (p, q, σ_p, σ_q)."""
        mean, logvar = self.forward(beta)
        p = mean[:, 0]
        cos_q, sin_q = mean[:, 1], mean[:, 2]
        q = torch.atan2(sin_q, cos_q)

        # Uncertainties
        sigma = torch.exp(0.5 * logvar)
        sigma_p = sigma[:, 0]
        # Propagate uncertainty through atan2 (approximate)
        sigma_q = torch.sqrt(sigma[:, 1]**2 + sigma[:, 2]**2)

        return p, q, sigma_p, sigma_q


def train_beta_encoder(encoder, beta_train, pcs_train, epochs=1000, lr=1e-3,
                       verbose=True, device='cpu'):
    """
    Train encoder to predict (p, cos q, sin q) from β.

    Args:
        encoder: BetaToStateEncoder
        beta_train: (n_samples, n_beta) numpy array
        pcs_train: (n_samples, 3) numpy array of (p, cos q, sin q)
        epochs: number of training epochs
        lr: learning rate
        verbose: print progress
        device: 'cpu' or 'cuda'

    Returns:
        losses: list of training losses
    """
    encoder = encoder.to(device)
    beta_t = torch.tensor(beta_train, dtype=torch.float32, device=device)
    pcs_t = torch.tensor(pcs_train, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

    losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        encoder.train()

        p_pred, cos_q_pred, sin_q_pred = encoder(beta_t)
        pcs_pred = torch.stack([p_pred, cos_q_pred, sin_q_pred], dim=1)

        loss = F.mse_loss(pcs_pred, pcs_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        loss_val = loss.item()
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val

        if verbose and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss_val:.6f}")

    return losses


def train_beta_encoder_with_uncertainty(encoder, beta_train, pcs_train,
                                        epochs=1000, lr=1e-3, verbose=True):
    """
    Train encoder with uncertainty using negative log-likelihood loss.

    NLL loss: -log p(y|x) = 0.5 * (log σ² + (y - μ)²/σ²)
    """
    beta_t = torch.tensor(beta_train, dtype=torch.float32)
    pcs_t = torch.tensor(pcs_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        encoder.train()

        mean, logvar = encoder(beta_t)

        # NLL loss (heteroscedastic)
        precision = torch.exp(-logvar)
        nll = 0.5 * (logvar + precision * (pcs_t - mean)**2)
        loss = nll.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (epoch + 1) % 200 == 0:
            # Also compute MSE for comparison
            mse = F.mse_loss(mean, pcs_t)
            print(f"  Epoch {epoch+1}: NLL = {loss.item():.4f}, MSE = {mse.item():.6f}")

    return losses


def evaluate_encoder(encoder, beta_test, pcs_test, device='cpu'):
    """
    Evaluate encoder on test set.

    Returns:
        dict with error metrics
    """
    encoder = encoder.to(device)
    encoder.eval()

    beta_t = torch.tensor(beta_test, dtype=torch.float32, device=device)
    pcs_t = torch.tensor(pcs_test, dtype=torch.float32)

    with torch.no_grad():
        p_pred, cos_q_pred, sin_q_pred = encoder(beta_t)
        pcs_pred = torch.stack([p_pred, cos_q_pred, sin_q_pred], dim=1).cpu().numpy()

    # Overall error
    mae = np.mean(np.abs(pcs_pred - pcs_test))
    mse = np.mean((pcs_pred - pcs_test)**2)

    # Per-component errors
    p_error = np.mean(np.abs(pcs_pred[:, 0] - pcs_test[:, 0]))
    cos_q_error = np.mean(np.abs(pcs_pred[:, 1] - pcs_test[:, 1]))
    sin_q_error = np.mean(np.abs(pcs_pred[:, 2] - pcs_test[:, 2]))

    # Angular error (in radians)
    q_pred = np.arctan2(pcs_pred[:, 2], pcs_pred[:, 1])
    q_true = np.arctan2(pcs_test[:, 2], pcs_test[:, 1])
    # Wrap angle difference to [-π, π]
    q_diff = np.arctan2(np.sin(q_pred - q_true), np.cos(q_pred - q_true))
    q_error = np.mean(np.abs(q_diff))

    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'p_error': p_error,
        'cos_q_error': cos_q_error,
        'sin_q_error': sin_q_error,
        'q_error_rad': q_error,
        'q_error_deg': np.degrees(q_error),
    }
