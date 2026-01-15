"""
Hybrid Nullspace Decoder for β reconstruction.

Instead of storing β_⊥ (which is window-specific and doesn't work for forecasting),
learn β_⊥ as a function of (P, sin(Q), cos(Q)) - the action-angle coordinates.

Key insight: β_⊥ encodes conserved quantities (like energy/amplitude) that should
depend only on the action P, plus phase-dependent shape corrections that depend on Q.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNullspaceDecoder(nn.Module):
    """
    Decode β from (p,q) using:
      β̂ = W⁺(p,q) + P_⊥ · g_θ(P, sin(Q), cos(Q))

    The linear part (W⁺) handles what's already captured by the pseudo-inverse.
    The MLP (g_θ) learns the nullspace correction using action-angle coords.

    This enables forecasting because g_θ depends on (P, Q), not the original signal.
    """

    def __init__(self, W_beta_to_pq, hjb_encoder, hidden_dim=32):
        """
        Args:
            W_beta_to_pq: Linear map β → (p,q), shape (n_beta, 2)
            hjb_encoder: Trained ImprovedHJB_MLP to get (P,Q) from (p,q)
            hidden_dim: Size of correction MLP
        """
        super().__init__()

        n_beta = W_beta_to_pq.shape[0]

        # Convert to torch tensors
        self.register_buffer('W', torch.tensor(W_beta_to_pq, dtype=torch.float32))
        W_pinv = np.linalg.pinv(W_beta_to_pq)
        self.register_buffer('W_pinv', torch.tensor(W_pinv, dtype=torch.float32))

        # Nullspace projector: P_⊥ = I - W @ W⁺
        P_parallel = W_beta_to_pq @ W_pinv
        P_perp = np.eye(n_beta) - P_parallel
        self.register_buffer('P_perp', torch.tensor(P_perp, dtype=torch.float32))

        # Store HJB encoder (frozen, used to get P, Q)
        self.hjb_encoder = hjb_encoder
        for param in self.hjb_encoder.parameters():
            param.requires_grad = False

        self.n_beta = n_beta

        # Correction MLP: features → β_correction
        # Input: (p, q, p², q², pq, P, sin(Q), cos(Q)) = 8 features
        # This gives the MLP full information about the state
        self.g = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.Tanh(),  # Tanh for bounded activations
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_beta)
        )

        # Initialize small weights for stability
        for m in self.g.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, p, q):
        """
        Decode (p,q) → β̂

        Args:
            p, q: Tensors of shape (batch,) or scalars

        Returns:
            beta_hat: Tensor of shape (batch, n_beta)
        """
        # Ensure batch dimension
        if p.dim() == 0:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)

        # Stack for batch processing
        pq = torch.stack([p, q], dim=-1)  # (batch, 2)

        # Linear part: (p,q) @ W⁺ᵀ = (batch, 2) @ (2, n_beta) = (batch, n_beta)
        # W_pinv is (2, n_beta), so we need pq @ W_pinv
        beta_linear = pq @ self.W_pinv  # (batch, n_beta)

        # Get action-angle coords (P, Q) from frozen encoder
        with torch.no_grad():
            P, Q = self.hjb_encoder.encode(p, q)

        # Correction input: (p, q, p², q², pq, P, sin(Q), cos(Q))
        # Includes both (p,q) features and action-angle features
        correction_input = torch.stack([
            p, q, p**2, q**2, p*q,  # Phase space features
            P, torch.sin(Q), torch.cos(Q)  # Action-angle features
        ], dim=-1)

        # Learned correction
        g_output = self.g(correction_input)  # (batch, n_beta)

        # Project to nullspace to ensure we only add orthogonal correction
        beta_correction = g_output @ self.P_perp.T  # (batch, n_beta)

        return beta_linear + beta_correction

    def decode_from_PQ(self, P, Q):
        """
        Decode directly from action-angle coordinates (P, Q).

        This is used for forecasting: propagate Q, then decode.

        Args:
            P, Q: Tensors of shape (batch,)

        Returns:
            beta_hat: Tensor of shape (batch, n_beta)
        """
        if P.dim() == 0:
            P = P.unsqueeze(0)
            Q = Q.unsqueeze(0)

        # First need to get (p, q) from (P, Q) via HJB decoder
        with torch.no_grad():
            p, q = self.hjb_encoder.decode(P, Q)

        # Now use standard forward
        return self.forward(p, q)

    def get_correction_magnitude(self, p, q):
        """
        Diagnostic: Return the relative magnitude of the correction term.

        Returns ||g_θ output|| / ||β̂||
        """
        if p.dim() == 0:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)

        with torch.no_grad():
            P, Q = self.hjb_encoder.encode(p, q)
            correction_input = torch.stack([
                p, q, p**2, q**2, p*q,
                P, torch.sin(Q), torch.cos(Q)
            ], dim=-1)
            g_output = self.g(correction_input)
            beta_correction = g_output @ self.P_perp.T

            pq = torch.stack([p, q], dim=-1)
            beta_linear = pq @ self.W_pinv
            beta_hat = beta_linear + beta_correction

            correction_norm = torch.norm(beta_correction, dim=-1)
            total_norm = torch.norm(beta_hat, dim=-1)

            return (correction_norm / (total_norm + 1e-8)).mean().item()


def train_nullspace_decoder(decoder, p_train, q_train, beta_train,
                            epochs=500, lr=1e-3, verbose=True):
    """
    Train g_θ to minimize ||β̂ - β_true||²

    Note: Only trains g (the correction MLP).
    W_pinv and hjb_encoder are frozen.

    Args:
        decoder: HybridNullspaceDecoder instance
        p_train, q_train: Ground truth (p,q) coordinates, shape (n_samples,)
        beta_train: Ground truth β from HST_ROM, shape (n_samples, n_beta)
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress

    Returns:
        losses: List of loss values per epoch
    """
    # Convert to tensors
    p_train = torch.tensor(p_train, dtype=torch.float32)
    q_train = torch.tensor(q_train, dtype=torch.float32)
    beta_train = torch.tensor(beta_train, dtype=torch.float32)

    # Only optimize the correction MLP
    optimizer = torch.optim.Adam(decoder.g.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    losses = []

    for epoch in range(epochs):
        decoder.train()

        # Forward pass
        beta_pred = decoder(p_train, q_train)

        # MSE loss
        loss = F.mse_loss(beta_pred, beta_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())

        if verbose and (epoch + 1) % 100 == 0:
            correction_mag = decoder.get_correction_magnitude(p_train, q_train)
            print(f"  Epoch {epoch+1:4d}: loss = {loss.item():.6f}, "
                  f"correction = {correction_mag:.3f}")

    return losses
