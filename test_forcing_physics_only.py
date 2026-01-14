"""
Test HJB_MLP with Deterministic Forcing (No Gauge Supervision)

Key Insight:
- Conservative data: P = constant → network can collapse to trivial P
- Forced data: dP = F_ext·dt → network MUST learn actual P

This tests Glinsky's actual framework: forcing + observation → action-angle
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass

from fixed_hjb_loss import ImprovedHJB_MLP
from action_angle_utils import wrap_to_2pi, angular_distance


@dataclass
class ForcingResult:
    """Results from one training run with forcing."""
    seed: int
    P_spearman: float
    P_pearson: float
    action_response_error: float
    evolution_error: float
    symplectic_error: float
    final_loss: float
    converged: bool


def generate_forced_sho_batch(n_traj: int, dt: float, omega: float = 1.0,
                               F_scale: float = 0.3, device: str = 'cpu'):
    """
    Generate SHO trajectories with deterministic forcing.

    The forcing F_ext causes action to change: dP = F_ext · dt
    This breaks the P = constant collapse mode.

    Args:
        n_traj: Number of trajectories
        dt: Time step
        omega: Angular frequency
        F_scale: Scale of forcing (randomized per trajectory)
        device: Compute device

    Returns:
        p0, q0, p1, q1: Phase space points at t=0 and t=dt
        F_ext: External forcing (causes dP = F_ext · dt)
        P0, Q0, P1, Q1: True action-angle coordinates
    """
    # Initial action-angle coordinates
    P0 = torch.rand(n_traj, device=device) * 2 + 0.5  # P ∈ [0.5, 2.5]
    Q0 = torch.rand(n_traj, device=device) * 2 * np.pi  # Q ∈ [0, 2π)

    # Random forcing per trajectory (can be positive or negative)
    F_ext = (torch.rand(n_traj, device=device) - 0.5) * 2 * F_scale  # F ∈ [-F_scale, F_scale]

    # Action-angle evolution with forcing
    # dP/dt = F_ext (action changes!)
    # dQ/dt = ω (angle evolves normally)
    P1 = P0 + F_ext * dt
    P1 = torch.clamp(P1, min=0.1)  # Keep P positive
    Q1 = Q0 + omega * dt
    Q1 = torch.remainder(Q1, 2 * np.pi)

    # Convert to phase space (p, q)
    # For SHO: P = E/ω = (p² + ω²q²)/(2ω)
    # Parametrization: p = √(2Pω) sin(Q), q = √(2P/ω) cos(Q)
    p0 = torch.sqrt(2 * P0 * omega) * torch.sin(Q0)
    q0 = torch.sqrt(2 * P0 / omega) * torch.cos(Q0)

    p1 = torch.sqrt(2 * P1 * omega) * torch.sin(Q1)
    q1 = torch.sqrt(2 * P1 / omega) * torch.cos(Q1)

    return p0, q0, p1, q1, F_ext, P0, Q0, P1, Q1


class ForcingLoss(nn.Module):
    """
    Loss function for learning action-angle with forcing.

    Key difference from conservative case:
    - Action CHANGES: dP = F_ext · dt (not conservation!)
    - This breaks the P = constant collapse mode

    NO GAUGE SUPERVISION - tests if physics + forcing determines (P, Q)
    """

    def __init__(self,
                 recon_weight: float = 1.0,
                 action_weight: float = 10.0,  # Key: action response
                 evolution_weight: float = 5.0,
                 symplectic_weight: float = 0.1):
        super().__init__()
        self.weights = {
            'recon': recon_weight,
            'action': action_weight,
            'evolution': evolution_weight,
            'symplectic': symplectic_weight
        }

    def forward(self, model, p0: torch.Tensor, q0: torch.Tensor,
                p1: torch.Tensor, q1: torch.Tensor,
                F_ext: torch.Tensor, omega: float, dt: float) -> Dict[str, torch.Tensor]:
        """
        Compute loss with forcing.

        Args:
            model: HJB_MLP instance
            p0, q0: Initial phase space point
            p1, q1: Evolved phase space point (with forcing!)
            F_ext: External forcing applied (per trajectory)
            omega: Angular frequency
            dt: Time step
        """
        # Encode at both points
        P0, Q0 = model.encode(p0, q0)
        P1, Q1 = model.encode(p1, q1)

        # 1. RECONSTRUCTION
        p0_rec, q0_rec = model.decode(P0, Q0)
        recon_loss = torch.mean((p0 - p0_rec)**2 + (q0 - q0_rec)**2)

        # 2. ACTION RESPONSE (not conservation!)
        # The network must predict dP = F_ext · dt
        dP_expected = F_ext * dt
        dP_actual = P1 - P0
        action_loss = torch.mean((dP_actual - dP_expected)**2)

        # 3. EVOLUTION (circular loss)
        dQ_expected = omega * dt
        dQ_actual = Q1 - Q0
        evolution_loss = torch.mean(1 - torch.cos(dQ_actual - dQ_expected))

        # 4. SYMPLECTIC
        symplectic_loss = self.compute_symplectic_loss(model, p0, q0)

        # Total (NO GAUGE!)
        total = (self.weights['recon'] * recon_loss +
                 self.weights['action'] * action_loss +
                 self.weights['evolution'] * evolution_loss +
                 self.weights['symplectic'] * symplectic_loss)

        return {
            'total': total,
            'recon': recon_loss,
            'action': action_loss,
            'evolution': evolution_loss,
            'symplectic': symplectic_loss
        }

    def compute_symplectic_loss(self, model, p: torch.Tensor, q: torch.Tensor,
                                 eps: float = 1e-4) -> torch.Tensor:
        """Enforce |{P, Q}| = 1."""
        n = min(len(p), 32)
        pb_values = []

        for i in range(n):
            pi = p[i:i+1]
            qi = q[i:i+1]

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
        return torch.mean((torch.abs(pb_tensor) - 1.0)**2)


def train_with_forcing(model, n_epochs: int = 3000, n_traj: int = 100,
                        omega: float = 1.0, dt: float = 0.3, F_scale: float = 0.3,
                        lr: float = 1e-3, device: str = None,
                        verbose: bool = False) -> List[Dict]:
    """Train with forcing, NO gauge supervision."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ForcingLoss()

    losses = []

    for epoch in range(n_epochs):
        # Generate forced trajectories
        p0, q0, p1, q1, F_ext, P0_true, Q0_true, P1_true, Q1_true = \
            generate_forced_sho_batch(n_traj, dt, omega, F_scale, device)

        optimizer.zero_grad()
        loss_dict = loss_fn(model, p0, q0, p1, q1, F_ext, omega, dt)
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append({k: v.item() for k, v in loss_dict.items()})

        if verbose and epoch % 500 == 0:
            ld = losses[-1]
            print(f"Epoch {epoch:4d}: total={ld['total']:.4f}, "
                  f"action={ld['action']:.6f}, evol={ld['evolution']:.4f}")

    return losses


def evaluate_forcing_model(model, omega: float = 1.0, dt: float = 0.3,
                            F_scale: float = 0.3, n_test: int = 200,
                            device: str = 'cpu') -> Dict:
    """Evaluate model trained with forcing."""

    model.eval()

    # Generate test data with forcing
    torch.manual_seed(9999)  # Fixed seed for consistent testing
    np.random.seed(9999)

    p0, q0, p1, q1, F_ext, P0_true, Q0_true, P1_true, Q1_true = \
        generate_forced_sho_batch(n_test, dt, omega, F_scale, device)

    with torch.no_grad():
        P0_pred, Q0_pred = model.encode(p0, q0)
        P1_pred, Q1_pred = model.encode(p1, q1)

    # Convert to numpy
    P0_pred = P0_pred.cpu().numpy()
    P1_pred = P1_pred.cpu().numpy()
    Q0_pred = Q0_pred.cpu().numpy()
    Q1_pred = Q1_pred.cpu().numpy()
    P0_true = P0_true.cpu().numpy()
    P1_true = P1_true.cpu().numpy()
    Q0_true = Q0_true.cpu().numpy()
    Q1_true = Q1_true.cpu().numpy()
    F_ext = F_ext.cpu().numpy()

    # 1. Check action response: dP_pred ≈ F_ext · dt
    dP_pred = P1_pred - P0_pred
    dP_expected = F_ext * dt

    # Need to account for possible gauge transformation P_pred = a·P_true + b
    # Then dP_pred = a·dP_true, so we fit the scale
    if np.std(dP_expected) > 1e-6:
        scale_fit = np.polyfit(dP_expected, dP_pred, 1)[0]
        dP_pred_scaled = dP_pred / scale_fit if abs(scale_fit) > 0.1 else dP_pred
        action_response_error = np.mean(np.abs(dP_pred_scaled - dP_expected)) / np.mean(np.abs(dP_expected))
    else:
        action_response_error = float('inf')

    # 2. Correlation with ground truth P
    r_P, _ = pearsonr(P0_pred, P0_true)
    rho_P, _ = spearmanr(P0_pred, P0_true)

    # 3. Evolution: dQ/dt ≈ ω
    dQ_pred = Q1_pred - Q0_pred
    # Unwrap for proper difference
    dQ_pred = np.arctan2(np.sin(dQ_pred), np.cos(dQ_pred))
    dQ_expected = omega * dt
    evolution_error = np.mean(np.abs(dQ_pred - dQ_expected)) / dQ_expected

    # 4. Symplectic check
    eps = 1e-4
    pb_values = []

    with torch.no_grad():
        for i in range(min(50, len(p0))):
            pi = p0[i:i+1]
            qi = q0[i:i+1]

            P_pp, Q_pp = model.encode(pi + eps, qi)
            P_pm, Q_pm = model.encode(pi - eps, qi)
            P_qp, Q_qp = model.encode(pi, qi + eps)
            P_qm, Q_qm = model.encode(pi, qi - eps)

            dP_dp = (P_pp - P_pm).item() / (2 * eps)
            dP_dq = (P_qp - P_qm).item() / (2 * eps)
            dQ_dp = (Q_pp - Q_pm).item() / (2 * eps)
            dQ_dq = (Q_qp - Q_qm).item() / (2 * eps)

            pb = dP_dq * dQ_dp - dP_dp * dQ_dq
            pb_values.append(pb)

    pb_array = np.array(pb_values)
    symplectic_error = np.mean(np.abs(np.abs(pb_array) - 1.0))

    return {
        'P_pearson': r_P,
        'P_spearman': rho_P,
        'action_response_error': action_response_error,
        'evolution_error': evolution_error,
        'symplectic_error': symplectic_error,
        'P0_pred': P0_pred,
        'P0_true': P0_true
    }


def run_multi_seed_forcing_experiment(n_seeds: int = 5, n_epochs: int = 3000,
                                       omega: float = 1.0) -> List[ForcingResult]:
    """Run multiple training runs with forcing to test convergence."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    print("=" * 70)
    print("FORCING EXPERIMENT: Physics + Forcing (No Gauge)")
    print("=" * 70)
    print(f"\nTraining {n_seeds} models with forcing, NO gauge supervision")
    print("Key: Action changes (dP = F·dt) should break P=constant collapse\n")

    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)
        losses = train_with_forcing(model, n_epochs=n_epochs, omega=omega,
                                     device=device, verbose=True)

        # Evaluate
        eval_result = evaluate_forcing_model(model, omega=omega, device=device)

        # Determine convergence
        converged = (abs(eval_result['P_spearman']) > 0.9 and
                     eval_result['evolution_error'] < 0.1)

        result = ForcingResult(
            seed=seed,
            P_spearman=eval_result['P_spearman'],
            P_pearson=eval_result['P_pearson'],
            action_response_error=eval_result['action_response_error'],
            evolution_error=eval_result['evolution_error'],
            symplectic_error=eval_result['symplectic_error'],
            final_loss=losses[-1]['total'],
            converged=converged
        )
        results.append(result)

        print(f"\n  Results:")
        print(f"    P correlations: Pearson={eval_result['P_pearson']:.4f}, "
              f"Spearman={eval_result['P_spearman']:.4f}")
        print(f"    Action response error: {eval_result['action_response_error']:.4f}")
        print(f"    Evolution error: {eval_result['evolution_error']:.4f}")
        print(f"    Symplectic error: {eval_result['symplectic_error']:.4f}")
        print(f"    Final loss: {losses[-1]['total']:.4f}")
        print(f"    Converged: {'✓ YES' if converged else '✗ NO'}")

    return results


def analyze_forcing_results(results: List[ForcingResult]):
    """Analyze forcing experiment results."""

    print("\n" + "=" * 70)
    print("FORCING EXPERIMENT ANALYSIS")
    print("=" * 70)

    n_converged = sum(r.converged for r in results)
    n_total = len(results)

    spearmans = [r.P_spearman for r in results]
    action_errors = [r.action_response_error for r in results]
    evol_errors = [r.evolution_error for r in results]

    print(f"\n1. CONVERGENCE RATE")
    print("-" * 40)
    print(f"   {n_converged}/{n_total} seeds converged")

    print(f"\n2. P CORRELATION (should be ≈ ±1)")
    print("-" * 40)
    print(f"   Spearman: {[f'{s:.4f}' for s in spearmans]}")
    print(f"   Mean |Spearman|: {np.mean(np.abs(spearmans)):.4f}")

    print(f"\n3. ACTION RESPONSE (should predict dP = F·dt)")
    print("-" * 40)
    print(f"   Errors: {[f'{e:.4f}' for e in action_errors]}")
    print(f"   Mean: {np.mean(action_errors):.4f}")

    print(f"\n4. EVOLUTION (dQ/dt = ω)")
    print("-" * 40)
    print(f"   Errors: {[f'{e:.4f}' for e in evol_errors]}")
    print(f"   Mean: {np.mean(evol_errors):.4f}")

    # Comparison with physics-only (no forcing)
    print("\n" + "=" * 70)
    print("COMPARISON: Forcing vs Conservative (Physics-Only)")
    print("=" * 70)
    print("\n| Metric              | Conservative | Forcing |")
    print("|---------------------|--------------|---------|")
    print(f"| Convergence rate    | 1/5 (20%)    | {n_converged}/5 ({100*n_converged//5}%) |")
    print(f"| Mean |Spearman|     | ~0.2         | {np.mean(np.abs(spearmans)):.2f}    |")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if n_converged == n_total:
        print("\n✓ FORCING BREAKS IDENTIFIABILITY GAP")
        print("  All seeds converged - physics + forcing determines (P, Q)")
        print("  Glinsky's framework VALIDATED: dE ≠ 0 is necessary!")
    elif n_converged > 1:
        print(f"\n◐ FORCING HELPS ({n_converged}/{n_total} vs 1/5)")
        print("  More seeds converge with forcing, but not all")
        print("  May need stronger forcing or more epochs")
    else:
        print("\n✗ FORCING INSUFFICIENT")
        print("  Same failure rate as conservative case")
        print("  Need to investigate further")


def create_forcing_visualization(results: List[ForcingResult],
                                  save_path: str = 'forcing_comparison.png'):
    """Visualize forcing results vs conservative."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        n_seeds = len(results)
        x = np.arange(n_seeds)

        # 1. Spearman correlation comparison
        ax = axes[0]
        # Conservative results from previous run (hardcoded)
        conservative_spearman = [0.0739, -0.0216, 0.0158, -0.9999, -0.0482]
        forcing_spearman = [r.P_spearman for r in results]

        width = 0.35
        ax.bar(x - width/2, np.abs(conservative_spearman), width, label='Conservative', color='coral', alpha=0.7)
        ax.bar(x + width/2, np.abs(forcing_spearman), width, label='Forcing', color='steelblue', alpha=0.7)
        ax.axhline(y=0.95, color='green', linestyle='--', label='Threshold (0.95)')
        ax.set_xlabel('Seed')
        ax.set_ylabel('|Spearman|')
        ax.set_title('P Correlation: Conservative vs Forcing')
        ax.legend()
        ax.set_ylim(0, 1.1)

        # 2. Final loss
        ax = axes[1]
        conservative_loss = [0.75, 0.76, 0.73, 0.001, 0.73]  # Approximate
        forcing_loss = [r.final_loss for r in results]

        ax.bar(x - width/2, conservative_loss, width, label='Conservative', color='coral', alpha=0.7)
        ax.bar(x + width/2, forcing_loss, width, label='Forcing', color='steelblue', alpha=0.7)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Final Loss')
        ax.set_title('Training Convergence')
        ax.legend()

        # 3. Convergence count
        ax = axes[2]
        conservative_converged = 1
        forcing_converged = sum(r.converged for r in results)

        bars = ax.bar(['Conservative', 'Forcing'], [conservative_converged, forcing_converged],
                       color=['coral', 'steelblue'], alpha=0.7)
        ax.axhline(y=5, color='gray', linestyle='--', label='Total (5)')
        ax.set_ylabel('Seeds Converged')
        ax.set_title('Convergence Rate')
        ax.set_ylim(0, 6)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}/5',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n[Saved forcing comparison to {save_path}]")

    except ImportError:
        print("\n[matplotlib not available - skipping visualization]")


if __name__ == "__main__":
    # Run forcing experiment
    results = run_multi_seed_forcing_experiment(n_seeds=5, n_epochs=3000)

    # Analyze results
    analyze_forcing_results(results)

    # Create visualization
    create_forcing_visualization(results)

    # Detailed table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Seed':>4} | {'Spearman':>9} | {'Action Err':>10} | {'Evol Err':>9} | {'Loss':>8} | {'Conv':>6}")
    print("-" * 70)
    for r in results:
        conv = '✓' if r.converged else '✗'
        print(f"{r.seed:4d} | {r.P_spearman:9.4f} | {r.action_response_error:10.4f} | "
              f"{r.evolution_error:9.4f} | {r.final_loss:8.4f} | {conv:>6}")
