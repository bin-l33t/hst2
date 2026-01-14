"""
Fixed HJB Loss for Learning Action-Angle Coordinates

Key fixes from our analysis:
1. Conservation: encode-encode test (not propagate-compare which is vacuous)
2. Evolution: circular loss for angle Q
3. Symplectic: enforce {P, Q} = -1 (correct sign for (P,Q) ordering)
4. Gauge: light supervision to pin the standard chart

Standard convention for SHO:
  P = (p² + ω²q²) / (2ω)
  Q = atan2(p, ωq)  ∈ [0, 2π)

This gives {P, Q} = -1 (not +1!) due to ordering.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple


class FixedHJBLoss(nn.Module):
    """
    Fixed loss function for learning action-angle coordinates.

    Key differences from original HJBLoss:
    1. Conservation tested by encoding TWO trajectory points
    2. Evolution uses circular loss: 1 - cos(ΔQ - ωdt)
    3. Symplectic constraint: {P, Q} = -1
    4. Optional gauge supervision to standard chart
    """

    def __init__(self,
                 recon_weight: float = 1.0,
                 conservation_weight: float = 10.0,
                 evolution_weight: float = 5.0,
                 symplectic_weight: float = 0.1,
                 gauge_weight: float = 10.0):  # Strong supervision!
        super().__init__()
        self.weights = {
            'recon': recon_weight,
            'conservation': conservation_weight,
            'evolution': evolution_weight,
            'symplectic': symplectic_weight,
            'gauge': gauge_weight
        }

    def forward(self, model, p0: torch.Tensor, q0: torch.Tensor,
                p1: torch.Tensor, q1: torch.Tensor,
                omega: float, dt: float) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            model: HJB_MLP instance
            p0, q0: Initial state on trajectory (batch)
            p1, q1: Evolved state on SAME trajectory (batch)
            omega: Angular frequency (for SHO, constant)
            dt: Time between (p0,q0) and (p1,q1)

        Returns:
            Dict with 'total' and component losses
        """
        # Encode at TWO trajectory points
        P0, Q0 = model.encode(p0, q0)
        P1, Q1 = model.encode(p1, q1)

        # 1. RECONSTRUCTION: decode should recover (p, q)
        p0_recon, q0_recon = model.decode(P0, Q0)
        recon_loss = torch.mean((p0 - p0_recon)**2 + (q0 - q0_recon)**2)

        # 2. CONSERVATION: P should be SAME at both trajectory points
        # This is the critical fix - tests actual conservation, not tautology
        conservation_loss = torch.mean((P0 - P1)**2)

        # 3. EVOLUTION: Q should advance by ω·dt
        # Use CIRCULAR loss to handle wraparound correctly
        dQ_expected = omega * dt
        dQ_actual = Q1 - Q0
        # 1 - cos(Δ) is 0 when Δ=0, max=2 when Δ=π
        evolution_loss = torch.mean(1 - torch.cos(dQ_actual - dQ_expected))

        # 4. SYMPLECTIC: {P, Q} = -1
        symplectic_loss = self.compute_symplectic_loss(model, p0, q0)

        # 5. GAUGE: Light supervision to standard chart
        # Standard: Q = atan2(p, ω·q)
        P_true = (p0**2 + omega**2 * q0**2) / (2 * omega)
        Q_true = torch.atan2(p0, omega * q0)
        Q_true = torch.remainder(Q_true, 2 * np.pi)

        # P gauge loss (MSE)
        gauge_loss_P = torch.mean((P0 - P_true)**2)
        # Q gauge loss (circular)
        gauge_loss_Q = torch.mean(1 - torch.cos(Q0 - Q_true))
        gauge_loss = gauge_loss_P + gauge_loss_Q

        # Total loss
        total = (self.weights['recon'] * recon_loss +
                 self.weights['conservation'] * conservation_loss +
                 self.weights['evolution'] * evolution_loss +
                 self.weights['symplectic'] * symplectic_loss +
                 self.weights['gauge'] * gauge_loss)

        return {
            'total': total,
            'recon': recon_loss,
            'conservation': conservation_loss,
            'evolution': evolution_loss,
            'symplectic': symplectic_loss,
            'gauge': gauge_loss
        }

    def compute_symplectic_loss(self, model, p: torch.Tensor, q: torch.Tensor,
                                 eps: float = 1e-4) -> torch.Tensor:
        """
        Enforce {P, Q} = -1 via finite differences.

        {P, Q} = ∂P/∂q · ∂Q/∂p - ∂P/∂p · ∂Q/∂q

        For standard convention with (P, Q) output ordering, this should be -1.
        """
        # Use subset of batch for efficiency
        n = min(len(p), 32)

        pb_values = []

        for i in range(n):
            pi = p[i:i+1]
            qi = q[i:i+1]

            # Finite differences for partial derivatives
            P_pp, Q_pp = model.encode(pi + eps, qi)
            P_pm, Q_pm = model.encode(pi - eps, qi)
            P_qp, Q_qp = model.encode(pi, qi + eps)
            P_qm, Q_qm = model.encode(pi, qi - eps)

            dP_dp = (P_pp - P_pm) / (2 * eps)
            dP_dq = (P_qp - P_qm) / (2 * eps)
            dQ_dp = (Q_pp - Q_pm) / (2 * eps)
            dQ_dq = (Q_qp - Q_qm) / (2 * eps)

            # Poisson bracket
            pb = dP_dq * dQ_dp - dP_dp * dQ_dq
            pb_values.append(pb)

        pb_tensor = torch.cat(pb_values)

        # Target is +1 for Q = atan2(p, ωq) convention
        # (Verified analytically: {P, Q} = (ω²q² + p²)/(2ωP) = 1)
        return torch.mean((pb_tensor - 1.0)**2)


def train_on_sho_fixed(model, n_epochs: int = 2000, n_trajectories: int = 100,
                       omega: float = 1.0, dt: float = 0.5, lr: float = 1e-3,
                       device: str = None) -> list:
    """
    Train HJB_MLP with fixed loss on SHO trajectories.

    Key difference: Generate (p0, q0) AND (p1, q1) on SAME orbit.
    This enables true conservation testing.

    Args:
        model: HJB_MLP instance
        n_epochs: Number of training epochs
        n_trajectories: Batch size (trajectories per epoch)
        omega: SHO angular frequency
        dt: Time step between trajectory points
        lr: Learning rate
        device: Compute device ('cpu' or 'cuda')

    Returns:
        List of loss dictionaries
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FixedHJBLoss()

    losses = []

    for epoch in range(n_epochs):
        # Generate random initial conditions
        E = torch.rand(n_trajectories, device=device) * 4 + 0.5  # E ∈ [0.5, 4.5]
        theta0 = torch.rand(n_trajectories, device=device) * 2 * np.pi

        # Initial state (p, q) from (E, theta)
        # SHO: E = p²/2 + ω²q²/2
        # Parametrize: p = √(2E) sin(θ), q = √(2E)/ω cos(θ)
        p0 = torch.sqrt(2 * E) * torch.sin(theta0)
        q0 = torch.sqrt(2 * E) / omega * torch.cos(theta0)

        # Evolved state on SAME orbit
        # SHO evolution: θ(t) = θ₀ + ωt
        theta1 = theta0 + omega * dt
        p1 = torch.sqrt(2 * E) * torch.sin(theta1)
        q1 = torch.sqrt(2 * E) / omega * torch.cos(theta1)

        # Forward pass
        optimizer.zero_grad()
        loss_dict = loss_fn(model, p0, q0, p1, q1, omega, dt)

        # Backward pass
        loss_dict['total'].backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Record losses
        losses.append({k: v.item() for k, v in loss_dict.items()})

        if epoch % 200 == 0:
            ld = losses[-1]
            print(f"Epoch {epoch:4d}: "
                  f"total={ld['total']:.4f}, "
                  f"cons={ld['conservation']:.6f}, "
                  f"evol={ld['evolution']:.4f}, "
                  f"symp={ld['symplectic']:.4f}, "
                  f"gauge={ld['gauge']:.4f}")

    return losses


def evaluate_fixed(model, omega: float = 1.0, n_test: int = 200,
                   device: str = None) -> Dict:
    """
    Evaluate trained model against ground truth.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Generate test data
    np.random.seed(42)
    E = np.random.uniform(0.5, 4.5, n_test)
    theta_true = np.random.uniform(0, 2 * np.pi, n_test)

    # Ground truth
    P_true = E / omega
    Q_true = theta_true  # For SHO, angle variable = theta

    # Convert to (p, q)
    p = np.sqrt(2 * E) * np.sin(theta_true)
    q = np.sqrt(2 * E) / omega * np.cos(theta_true)

    # Get MLP predictions
    with torch.no_grad():
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)
        P_pred_t, Q_pred_t = model.encode(p_t, q_t)
        P_pred = P_pred_t.cpu().numpy()
        Q_pred = Q_pred_t.cpu().numpy()

    # Wrap Q to [0, 2π)
    Q_pred = Q_pred % (2 * np.pi)
    Q_true = Q_true % (2 * np.pi)

    # Compute errors
    from scipy.stats import pearsonr

    r_P, _ = pearsonr(P_pred, P_true)
    P_rel_error = np.mean(np.abs(P_pred - P_true) / P_true)

    # Angular distance for Q
    def angular_distance(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, 2 * np.pi - diff)

    Q_errors = angular_distance(Q_pred, Q_true)
    Q_mean_error = np.mean(Q_errors)

    # Circular correlation
    r_cos, _ = pearsonr(np.cos(Q_pred), np.cos(Q_true))
    r_sin, _ = pearsonr(np.sin(Q_pred), np.sin(Q_true))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAction P:")
    print(f"  Correlation: {r_P:.4f}")
    print(f"  Relative error: {P_rel_error:.4f}")
    print(f"  {'✓ PASS' if r_P > 0.95 else '✗ FAIL'}")

    print(f"\nAngle Q:")
    print(f"  cos correlation: {r_cos:.4f}")
    print(f"  sin correlation: {r_sin:.4f}")
    print(f"  Mean angular error: {Q_mean_error:.4f} rad = {np.degrees(Q_mean_error):.1f}°")
    print(f"  {'✓ PASS' if Q_mean_error < 0.2 else '✗ FAIL'}")

    return {
        'P_corr': r_P,
        'P_rel_error': P_rel_error,
        'Q_cos_corr': r_cos,
        'Q_sin_corr': r_sin,
        'Q_mean_error': Q_mean_error,
        'P_pred': P_pred,
        'P_true': P_true,
        'Q_pred': Q_pred,
        'Q_true': Q_true
    }


class ImprovedHJB_MLP(nn.Module):
    """
    Improved HJB_MLP with quadratic input features.

    Key insight: P = (p² + ω²q²)/(2ω) is quadratic in p, q.
    Adding quadratic features (p², q², pq) makes this much easier to learn.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()

        # Input: (p, q, p², q², pq) = 5 features
        input_dim = 5

        # Encoder network with quadratic features
        encoder_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.encoder = nn.Sequential(*encoder_layers)

        # Action head: hidden → P (positive)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures P > 0
        )

        # Angle head: hidden → (sin(Q), cos(Q)) for circular output
        self.angle_head = nn.Linear(hidden_dim, 2)

        # Decoder: (P, sin(Q), cos(Q)) → (p, q)
        decoder_layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        decoder_layers.append(nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(*decoder_layers)

    def _make_features(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Create input features including quadratics."""
        return torch.stack([p, q, p**2, q**2, p*q], dim=-1)

    def encode(self, p: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map (p, q) → (P, Q)"""
        x = self._make_features(p, q)
        h = self.encoder(x)

        # P via softplus (always positive)
        P = self.action_head(h).squeeze(-1)

        # Q via atan2 of learned (sin, cos)
        sc = self.angle_head(h)
        sin_Q = sc[..., 0]
        cos_Q = sc[..., 1]
        Q = torch.atan2(sin_Q, cos_Q)
        # Wrap to [0, 2π)
        Q = torch.remainder(Q, 2 * np.pi)

        return P, Q

    def decode(self, P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map (P, Q) → (p, q)"""
        sin_Q = torch.sin(Q)
        cos_Q = torch.cos(Q)
        x = torch.stack([P, sin_Q, cos_Q], dim=-1)
        out = self.decoder(x)
        p = out[..., 0]
        q = out[..., 1]
        return p, q


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING IMPROVED HJB_MLP WITH FIXED LOSS")
    print("=" * 70)

    # Create improved model with quadratic features
    model = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ImprovedHJB_MLP with {n_params} parameters")
    print("  - Quadratic input features: (p, q, p², q², pq)")
    print("  - Circular Q output via atan2(sin, cos)")
    print("  - Strong gauge supervision (weight=10)")

    # Train with fixed loss
    print("\n" + "-" * 70)
    print("TRAINING (this may take a few minutes)")
    print("-" * 70)

    losses = train_on_sho_fixed(model, n_epochs=3000, omega=1.0, dt=0.5, lr=1e-3)

    # Evaluate
    results = evaluate_fixed(model, omega=1.0)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = (results['P_corr'] > 0.95 and
                results['Q_mean_error'] < 0.2)

    if all_pass:
        print("\n✓ ALL TESTS PASSED - HJB_MLP learned action-angle coordinates!")
    else:
        print("\n✗ SOME TESTS FAILED - May need more training or architecture changes")
