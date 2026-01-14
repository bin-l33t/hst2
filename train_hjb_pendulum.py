"""
Train Improved HJB_MLP on Pendulum (Libration Regime)

Key differences from SHO:
1. ω = ω(E) depends on energy (nonlinear)
2. P = J(E) involves elliptic integrals
3. Behavior changes near separatrix (E → 1)

Uses validated formulas from pendulum_action_angle.py.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

from pendulum_action_angle import (
    pendulum_energy,
    pendulum_action_from_energy,
    pendulum_omega_from_energy,
    pendulum_action_angle,
    pendulum_from_action_angle,
    J_SEPARATRIX
)
from action_angle_utils import wrap_to_2pi


class PendulumHJB_MLP(nn.Module):
    """
    HJB_MLP adapted for pendulum.

    Key insight: E = p²/2 - cos(q) is the natural coordinate.
    Network learns J(E) and Q from (p, q, E, cos(q), sin(q)).
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()

        # Input: (p, q, p², q², pq, E, cos(q), sin(q)) = 8 features
        input_dim = 8

        # Deeper network for nonlinear pendulum
        encoder_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        self.encoder = nn.Sequential(*encoder_layers)

        # Action head: hidden → P (positive, bounded by J_sep)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs [0, 1], scale to [0, J_sep]
        )
        self.j_sep = J_SEPARATRIX

        # Angle head: hidden → (sin(Q), cos(Q))
        self.angle_head = nn.Linear(hidden_dim, 2)

        # Decoder: (P, sin(Q), cos(Q)) → (p, q)
        decoder_layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        decoder_layers.append(nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(*decoder_layers)

    def _make_features(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Create input features."""
        E = p**2 / 2 - torch.cos(q)  # Pendulum energy
        return torch.stack([
            p, q, p**2, q**2, p*q,
            E, torch.cos(q), torch.sin(q)
        ], dim=-1)

    def encode(self, p: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map (p, q) → (P, Q)"""
        x = self._make_features(p, q)
        h = self.encoder(x)

        # P via sigmoid scaled to [0, J_sep)
        P_raw = self.action_head(h).squeeze(-1)
        P = P_raw * self.j_sep * 0.99  # Stay slightly below separatrix

        # Q via atan2
        sc = self.angle_head(h)
        sin_Q = sc[..., 0]
        cos_Q = sc[..., 1]
        Q = torch.atan2(sin_Q, cos_Q)
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


class PendulumHJBLoss(nn.Module):
    """
    Loss for learning pendulum action-angle coordinates.

    Key difference from SHO: ω is per-trajectory (not constant).
    """

    def __init__(self,
                 recon_weight: float = 1.0,
                 conservation_weight: float = 10.0,
                 evolution_weight: float = 5.0,
                 symplectic_weight: float = 0.1,
                 gauge_weight: float = 5.0):
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
                omega_values: torch.Tensor, dt: float,
                P_true: torch.Tensor = None,
                Q_true: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            omega_values: Per-trajectory frequencies (tensor)
            P_true, Q_true: Ground truth for gauge loss (optional)
        """
        P0, Q0 = model.encode(p0, q0)
        P1, Q1 = model.encode(p1, q1)

        # 1. RECONSTRUCTION
        p0_recon, q0_recon = model.decode(P0, Q0)
        recon_loss = torch.mean((p0 - p0_recon)**2 + (q0 - q0_recon)**2)

        # 2. CONSERVATION: P same at both trajectory points
        conservation_loss = torch.mean((P0 - P1)**2)

        # 3. EVOLUTION: dQ = ω(P) · dt (per-trajectory ω!)
        dQ_expected = omega_values * dt
        dQ_actual = Q1 - Q0
        evolution_loss = torch.mean(1 - torch.cos(dQ_actual - dQ_expected))

        # 4. SYMPLECTIC (computed on subset)
        symplectic_loss = self.compute_symplectic_loss(model, p0, q0)

        # 5. GAUGE: supervision to ground truth
        if P_true is not None and Q_true is not None:
            gauge_loss_P = torch.mean((P0 - P_true)**2)
            gauge_loss_Q = torch.mean(1 - torch.cos(Q0 - Q_true))
            gauge_loss = gauge_loss_P + gauge_loss_Q
        else:
            gauge_loss = torch.tensor(0.0, device=p0.device)

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

    def compute_symplectic_loss(self, model, p, q, eps=1e-4):
        """Enforce |{P, Q}| = 1."""
        n = min(len(p), 32)
        pb_values = []

        for i in range(n):
            pi, qi = p[i:i+1], q[i:i+1]

            P_pp, Q_pp = model.encode(pi + eps, qi)
            P_pm, Q_pm = model.encode(pi - eps, qi)
            P_qp, Q_qp = model.encode(pi, qi + eps)
            P_qm, Q_qm = model.encode(pi, qi - eps)

            dP_dp = (P_pp - P_pm) / (2 * eps)
            dP_dq = (P_qp - P_qm) / (2 * eps)
            dQ_dp = (Q_pp - Q_pm) / (2 * eps)
            dQ_dq = (Q_qp - Q_qm) / (2 * eps)

            pb = dP_dq * dQ_dp - dP_dp * dQ_dq
            pb_values.append(pb)

        pb_tensor = torch.cat(pb_values)
        # Target |{P,Q}| = 1 (either sign is valid)
        return torch.mean((torch.abs(pb_tensor) - 1.0)**2)


def generate_pendulum_batch(n_traj: int, dt: float,
                            E_range: Tuple[float, float] = (-0.8, 0.8),
                            device: str = 'cpu'):
    """
    Generate pendulum trajectories in libration regime.

    E ∈ [-1, 1) for libration. Stay away from E = 1 (separatrix).
    """
    # Sample energies
    E = np.random.uniform(*E_range, n_traj)

    # Get action and frequency
    J = np.array([pendulum_action_from_energy(e) for e in E])
    omega = np.array([pendulum_omega_from_energy(e) for e in E])

    # Sample initial angles
    Q0 = np.random.uniform(0, 2*np.pi, n_traj)

    # Convert to (p, q) using ground truth
    p0 = np.zeros(n_traj)
    q0 = np.zeros(n_traj)
    P_true = np.zeros(n_traj)
    Q_true = np.zeros(n_traj)

    for i in range(n_traj):
        try:
            q0[i], p0[i] = pendulum_from_action_angle(J[i], Q0[i])
            P_true[i] = J[i]
            Q_true[i] = Q0[i]
        except Exception:
            # Fallback for edge cases
            q0[i], p0[i] = 0.5, 0.5
            P_true[i] = 0.5
            Q_true[i] = Q0[i]

    # Evolve angle
    Q1 = wrap_to_2pi(Q0 + omega * dt)

    p1 = np.zeros(n_traj)
    q1 = np.zeros(n_traj)

    for i in range(n_traj):
        try:
            q1[i], p1[i] = pendulum_from_action_angle(J[i], Q1[i])
        except Exception:
            q1[i], p1[i] = q0[i], p0[i]

    # Convert to tensors
    return {
        'p0': torch.tensor(p0, dtype=torch.float32, device=device),
        'q0': torch.tensor(q0, dtype=torch.float32, device=device),
        'p1': torch.tensor(p1, dtype=torch.float32, device=device),
        'q1': torch.tensor(q1, dtype=torch.float32, device=device),
        'omega': torch.tensor(omega, dtype=torch.float32, device=device),
        'P_true': torch.tensor(P_true, dtype=torch.float32, device=device),
        'Q_true': torch.tensor(Q_true, dtype=torch.float32, device=device),
    }


def train_on_pendulum(model, n_epochs: int = 3000, n_trajectories: int = 100,
                      dt: float = 0.5, lr: float = 1e-3,
                      E_range: Tuple[float, float] = (-0.8, 0.8),
                      device: str = None) -> list:
    """Train PendulumHJB_MLP on pendulum trajectories."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PendulumHJBLoss()

    losses = []

    for epoch in range(n_epochs):
        batch = generate_pendulum_batch(n_trajectories, dt, E_range, device)

        optimizer.zero_grad()
        loss_dict = loss_fn(model,
                           batch['p0'], batch['q0'],
                           batch['p1'], batch['q1'],
                           batch['omega'], dt,
                           batch['P_true'], batch['Q_true'])

        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append({k: v.item() for k, v in loss_dict.items()})

        if epoch % 300 == 0:
            ld = losses[-1]
            print(f"Epoch {epoch:4d}: "
                  f"total={ld['total']:.4f}, "
                  f"cons={ld['conservation']:.6f}, "
                  f"evol={ld['evolution']:.4f}, "
                  f"gauge={ld['gauge']:.4f}")

    return losses


def evaluate_pendulum(model, n_test: int = 200,
                      E_range: Tuple[float, float] = (-0.8, 0.8),
                      device: str = None):
    """Evaluate trained pendulum model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Generate test data
    np.random.seed(42)
    E = np.random.uniform(*E_range, n_test)

    J_true = np.array([pendulum_action_from_energy(e) for e in E])
    omega_true = np.array([pendulum_omega_from_energy(e) for e in E])
    Q_true = np.random.uniform(0, 2*np.pi, n_test)

    p = np.zeros(n_test)
    q = np.zeros(n_test)

    for i in range(n_test):
        try:
            q[i], p[i] = pendulum_from_action_angle(J_true[i], Q_true[i])
        except Exception:
            q[i], p[i] = 0.5, 0.3

    # Get predictions
    with torch.no_grad():
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)
        P_pred_t, Q_pred_t = model.encode(p_t, q_t)
        P_pred = P_pred_t.cpu().numpy()
        Q_pred = Q_pred_t.cpu().numpy()

    Q_pred = wrap_to_2pi(Q_pred)
    Q_true = wrap_to_2pi(Q_true)

    # Compute errors
    from scipy.stats import pearsonr
    from action_angle_utils import angular_distance

    r_P, _ = pearsonr(P_pred, J_true)
    P_rel_error = np.mean(np.abs(P_pred - J_true) / J_true)

    Q_errors = angular_distance(Q_pred, Q_true)
    Q_mean_error = np.mean(Q_errors)

    r_cos, _ = pearsonr(np.cos(Q_pred), np.cos(Q_true))
    r_sin, _ = pearsonr(np.sin(Q_pred), np.sin(Q_true))

    print("\n" + "=" * 60)
    print("PENDULUM EVALUATION")
    print("=" * 60)
    print(f"\nEnergy range: E ∈ [{E_range[0]:.1f}, {E_range[1]:.1f}]")
    print(f"J range: [{J_true.min():.3f}, {J_true.max():.3f}] (J_sep = {J_SEPARATRIX:.3f})")

    print(f"\nAction J (P):")
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
        'J_true': J_true,
        'Q_pred': Q_pred,
        'Q_true': Q_true
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING PENDULUM HJB_MLP")
    print("=" * 70)
    print(f"\nPendulum libration regime: E ∈ [-1, 1)")
    print(f"Separatrix action: J_sep = 8/π = {J_SEPARATRIX:.4f}")

    # Create model
    model = PendulumHJB_MLP(hidden_dim=128, num_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: PendulumHJB_MLP with {n_params} parameters")
    print("  - Features: (p, q, p², q², pq, E, cos(q), sin(q))")
    print("  - Circular Q output")
    print("  - Bounded P output (0, J_sep)")

    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    losses = train_on_pendulum(model, n_epochs=4000, E_range=(-0.8, 0.8))

    # Evaluate on same range
    print("\n" + "-" * 70)
    print("EVALUATION (E ∈ [-0.8, 0.8])")
    print("-" * 70)
    results = evaluate_pendulum(model, E_range=(-0.8, 0.8))

    # Test near separatrix
    print("\n" + "-" * 70)
    print("EVALUATION NEAR SEPARATRIX (E ∈ [0.5, 0.9])")
    print("-" * 70)
    results_sep = evaluate_pendulum(model, E_range=(0.5, 0.9))
