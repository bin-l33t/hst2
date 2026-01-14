"""
Test Physics-Only Loss (No Gauge Supervision)

Question: Do physics constraints ALONE determine (P, Q) uniquely?
Or is there gauge freedom (P → aP+b, Q → Q+c)?

Theory prediction:
- P should be determined up to affine transform (same level sets)
- Q should be determined up to constant offset
- Physics (conservation, evolution, symplectic) should hold regardless
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass

from fixed_hjb_loss import ImprovedHJB_MLP
from action_angle_utils import wrap_to_2pi, angular_distance


@dataclass
class GaugeAnalysis:
    """Results of gauge analysis for one run."""
    seed: int
    P_slope: float
    P_intercept: float
    Q_offset: float
    P_pearson: float
    P_spearman: float
    Q_cos_corr: float
    Q_sin_corr: float
    conservation_passed: bool
    evolution_passed: bool
    symplectic_passed: bool


class PhysicsOnlyLoss(nn.Module):
    """
    Loss function with NO gauge supervision.

    Only physics constraints:
    1. Reconstruction: (p,q) → (P,Q) → (p,q)
    2. Conservation: P(t=0) = P(t=dt) along trajectory
    3. Evolution: Q advances by ω·dt
    4. Symplectic: |{P, Q}| = 1
    """

    def __init__(self,
                 recon_weight: float = 1.0,
                 conservation_weight: float = 10.0,
                 evolution_weight: float = 5.0,
                 symplectic_weight: float = 0.1):
        super().__init__()
        self.weights = {
            'recon': recon_weight,
            'conservation': conservation_weight,
            'evolution': evolution_weight,
            'symplectic': symplectic_weight
        }

    def forward(self, model, p0: torch.Tensor, q0: torch.Tensor,
                p1: torch.Tensor, q1: torch.Tensor,
                omega: float, dt: float) -> Dict[str, torch.Tensor]:
        """Compute physics-only loss (no gauge supervision)."""

        # Encode at TWO trajectory points
        P0, Q0 = model.encode(p0, q0)
        P1, Q1 = model.encode(p1, q1)

        # 1. RECONSTRUCTION
        p0_recon, q0_recon = model.decode(P0, Q0)
        recon_loss = torch.mean((p0 - p0_recon)**2 + (q0 - q0_recon)**2)

        # 2. CONSERVATION: P should be same at both points
        conservation_loss = torch.mean((P0 - P1)**2)

        # 3. EVOLUTION: Q should advance by ω·dt (circular loss)
        dQ_expected = omega * dt
        dQ_actual = Q1 - Q0
        evolution_loss = torch.mean(1 - torch.cos(dQ_actual - dQ_expected))

        # 4. SYMPLECTIC: |{P, Q}| = 1
        symplectic_loss = self.compute_symplectic_loss(model, p0, q0)

        # NO GAUGE LOSS!

        total = (self.weights['recon'] * recon_loss +
                 self.weights['conservation'] * conservation_loss +
                 self.weights['evolution'] * evolution_loss +
                 self.weights['symplectic'] * symplectic_loss)

        return {
            'total': total,
            'recon': recon_loss,
            'conservation': conservation_loss,
            'evolution': evolution_loss,
            'symplectic': symplectic_loss
        }

    def compute_symplectic_loss(self, model, p: torch.Tensor, q: torch.Tensor,
                                 eps: float = 1e-4) -> torch.Tensor:
        """Enforce |{P, Q}| = 1 via finite differences."""
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
        # Target |{P, Q}| = 1
        return torch.mean((torch.abs(pb_tensor) - 1.0)**2)


def train_physics_only(model, n_epochs: int = 3000, n_trajectories: int = 100,
                       omega: float = 1.0, dt: float = 0.5, lr: float = 1e-3,
                       device: str = None, verbose: bool = False) -> List[Dict]:
    """Train with physics-only loss."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PhysicsOnlyLoss()

    losses = []

    for epoch in range(n_epochs):
        # Generate trajectory pairs
        E = torch.rand(n_trajectories, device=device) * 4 + 0.5
        theta0 = torch.rand(n_trajectories, device=device) * 2 * np.pi

        p0 = torch.sqrt(2 * E) * torch.sin(theta0)
        q0 = torch.sqrt(2 * E) / omega * torch.cos(theta0)

        theta1 = theta0 + omega * dt
        p1 = torch.sqrt(2 * E) * torch.sin(theta1)
        q1 = torch.sqrt(2 * E) / omega * torch.cos(theta1)

        optimizer.zero_grad()
        loss_dict = loss_fn(model, p0, q0, p1, q1, omega, dt)
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append({k: v.item() for k, v in loss_dict.items()})

        if verbose and epoch % 500 == 0:
            ld = losses[-1]
            print(f"Epoch {epoch:4d}: total={ld['total']:.4f}, "
                  f"cons={ld['conservation']:.6f}, evol={ld['evolution']:.4f}")

    return losses


def sho_ground_truth(p, q, omega=1.0):
    """Ground truth (P, Q) for SHO."""
    P = (p**2 + omega**2 * q**2) / (2 * omega)
    Q = np.arctan2(p, omega * q)
    Q = wrap_to_2pi(Q)
    return P, Q


def analyze_gauge(model, omega: float = 1.0, n_test: int = 200,
                  device: str = 'cpu') -> Dict:
    """Analyze the gauge relationship between predicted and true (P, Q)."""

    np.random.seed(42)  # Fixed seed for consistent test data
    E = np.random.uniform(0.5, 4.5, n_test)
    theta = np.random.uniform(0, 2 * np.pi, n_test)

    p = np.sqrt(2 * E) * np.sin(theta)
    q = np.sqrt(2 * E) / omega * np.cos(theta)

    P_true, Q_true = sho_ground_truth(p, q, omega)

    model.eval()
    with torch.no_grad():
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)
        P_pred_t, Q_pred_t = model.encode(p_t, q_t)
        P_pred = P_pred_t.cpu().numpy()
        Q_pred = Q_pred_t.cpu().numpy()

    Q_pred = wrap_to_2pi(Q_pred)

    # Fit affine transform for P: P_pred ≈ slope * P_true + intercept
    slope, intercept = np.polyfit(P_true, P_pred, 1)

    # Circular mean of Q offset
    Q_diff = Q_pred - Q_true
    Q_offset = np.arctan2(np.mean(np.sin(Q_diff)), np.mean(np.cos(Q_diff)))

    # Correlations
    r_P, _ = pearsonr(P_pred, P_true)
    rho_P, _ = spearmanr(P_pred, P_true)
    r_cos, _ = pearsonr(np.cos(Q_pred), np.cos(Q_true))
    r_sin, _ = pearsonr(np.sin(Q_pred), np.sin(Q_true))

    return {
        'P_slope': slope,
        'P_intercept': intercept,
        'Q_offset': Q_offset,
        'P_pearson': r_P,
        'P_spearman': rho_P,
        'Q_cos_corr': r_cos,
        'Q_sin_corr': r_sin,
        'P_pred': P_pred,
        'P_true': P_true,
        'Q_pred': Q_pred,
        'Q_true': Q_true
    }


def test_physics_properties(model, omega: float = 1.0, device: str = 'cpu') -> Dict[str, bool]:
    """Test physics properties regardless of gauge."""

    results = {}

    # Test conservation: P constant along trajectory
    np.random.seed(100)
    E = np.random.uniform(0.5, 4.5, 50)
    theta0 = np.random.uniform(0, 2 * np.pi, 50)

    p0 = np.sqrt(2 * E) * np.sin(theta0)
    q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

    model.eval()
    with torch.no_grad():
        p_t = torch.tensor(p0, dtype=torch.float32, device=device)
        q_t = torch.tensor(q0, dtype=torch.float32, device=device)
        P0_t, _ = model.encode(p_t, q_t)
        P0 = P0_t.cpu().numpy()

    # Check at multiple times
    max_rel_change = 0
    for dt in np.linspace(0, 2*np.pi, 20):
        theta_t = theta0 + omega * dt
        p_t_np = np.sqrt(2 * E) * np.sin(theta_t)
        q_t_np = np.sqrt(2 * E) / omega * np.cos(theta_t)

        with torch.no_grad():
            p_t = torch.tensor(p_t_np, dtype=torch.float32, device=device)
            q_t = torch.tensor(q_t_np, dtype=torch.float32, device=device)
            P_t_t, _ = model.encode(p_t, q_t)
            P_t = P_t_t.cpu().numpy()

        rel_change = np.abs(P_t - P0) / (np.abs(P0) + 1e-10)
        max_rel_change = max(max_rel_change, np.max(rel_change))

    results['conservation'] = max_rel_change < 0.1  # Allow 10% variation

    # Test evolution: dQ/dt = ω
    dt_values = np.linspace(0, 2*np.pi, 50)
    Q_traj = np.zeros((len(dt_values), 50))

    for i, dt in enumerate(dt_values):
        theta_t = theta0 + omega * dt
        p_t_np = np.sqrt(2 * E) * np.sin(theta_t)
        q_t_np = np.sqrt(2 * E) / omega * np.cos(theta_t)

        with torch.no_grad():
            p_t = torch.tensor(p_t_np, dtype=torch.float32, device=device)
            q_t = torch.tensor(q_t_np, dtype=torch.float32, device=device)
            _, Q_t_t = model.encode(p_t, q_t)
            Q_traj[i] = Q_t_t.cpu().numpy()

    Q_unwrapped = np.unwrap(Q_traj, axis=0)
    omega_measured = []
    for j in range(50):
        slope, _ = np.polyfit(dt_values, Q_unwrapped[:, j], 1)
        omega_measured.append(slope)

    omega_mean = np.mean(omega_measured)
    omega_error = abs(omega_mean - omega) / omega
    results['evolution'] = omega_error < 0.05  # 5% error

    # Test symplectic: |{P, Q}| ≈ 1
    np.random.seed(101)
    E_test = np.random.uniform(0.5, 4.5, 30)
    theta_test = np.random.uniform(0, 2 * np.pi, 30)
    p_test = np.sqrt(2 * E_test) * np.sin(theta_test)
    q_test = np.sqrt(2 * E_test) / omega * np.cos(theta_test)

    eps = 1e-4
    pb_values = []

    with torch.no_grad():
        for i in range(len(p_test)):
            pi, qi = p_test[i], q_test[i]

            def encode_pt(pi, qi):
                pt = torch.tensor([pi], dtype=torch.float32, device=device)
                qt = torch.tensor([qi], dtype=torch.float32, device=device)
                Pt, Qt = model.encode(pt, qt)
                return Pt.item(), Qt.item()

            P_pp, Q_pp = encode_pt(pi + eps, qi)
            P_pm, Q_pm = encode_pt(pi - eps, qi)
            P_qp, Q_qp = encode_pt(pi, qi + eps)
            P_qm, Q_qm = encode_pt(pi, qi - eps)

            dP_dp = (P_pp - P_pm) / (2 * eps)
            dP_dq = (P_qp - P_qm) / (2 * eps)
            dQ_dp = (Q_pp - Q_pm) / (2 * eps)
            dQ_dq = (Q_qp - Q_qm) / (2 * eps)

            pb = dP_dq * dQ_dp - dP_dp * dQ_dq
            pb_values.append(pb)

    pb_array = np.array(pb_values)
    pb_error = np.mean(np.abs(np.abs(pb_array) - 1.0))
    results['symplectic'] = pb_error < 0.2

    return results


def run_multi_seed_experiment(n_seeds: int = 5, n_epochs: int = 3000,
                               omega: float = 1.0) -> List[GaugeAnalysis]:
    """Run multiple training runs with different seeds."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    print("=" * 70)
    print("PHYSICS-ONLY LOSS: GAUGE FREEDOM EXPERIMENT")
    print("=" * 70)
    print(f"\nTraining {n_seeds} models with different seeds...")
    print("NO gauge supervision - only physics constraints\n")

    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create and train model
        model = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)
        losses = train_physics_only(model, n_epochs=n_epochs, omega=omega,
                                     device=device, verbose=True)

        # Analyze gauge
        gauge = analyze_gauge(model, omega=omega, device=device)

        # Test physics
        physics = test_physics_properties(model, omega=omega, device=device)

        # Store results
        result = GaugeAnalysis(
            seed=seed,
            P_slope=gauge['P_slope'],
            P_intercept=gauge['P_intercept'],
            Q_offset=gauge['Q_offset'],
            P_pearson=gauge['P_pearson'],
            P_spearman=gauge['P_spearman'],
            Q_cos_corr=gauge['Q_cos_corr'],
            Q_sin_corr=gauge['Q_sin_corr'],
            conservation_passed=physics['conservation'],
            evolution_passed=physics['evolution'],
            symplectic_passed=physics['symplectic']
        )
        results.append(result)

        print(f"\n  Gauge Analysis:")
        print(f"    P: slope={gauge['P_slope']:.3f}, intercept={gauge['P_intercept']:.3f}")
        print(f"    Q offset: {gauge['Q_offset']:.3f} rad = {np.degrees(gauge['Q_offset']):.1f}°")
        print(f"    P correlations: Pearson={gauge['P_pearson']:.4f}, Spearman={gauge['P_spearman']:.4f}")
        print(f"    Q correlations: cos={gauge['Q_cos_corr']:.4f}, sin={gauge['Q_sin_corr']:.4f}")
        print(f"\n  Physics Tests:")
        print(f"    Conservation: {'✓ PASS' if physics['conservation'] else '✗ FAIL'}")
        print(f"    Evolution: {'✓ PASS' if physics['evolution'] else '✗ FAIL'}")
        print(f"    Symplectic: {'✓ PASS' if physics['symplectic'] else '✗ FAIL'}")

    return results


def analyze_gauge_freedom(results: List[GaugeAnalysis]):
    """Analyze whether gauge is unique or has freedom."""

    print("\n" + "=" * 70)
    print("GAUGE FREEDOM ANALYSIS")
    print("=" * 70)

    # Extract parameters
    slopes = [r.P_slope for r in results]
    intercepts = [r.P_intercept for r in results]
    offsets = [r.Q_offset for r in results]

    print("\n1. P GAUGE PARAMETERS")
    print("-" * 40)
    print(f"   Slopes:     mean={np.mean(slopes):.3f}, std={np.std(slopes):.3f}")
    print(f"               range=[{np.min(slopes):.3f}, {np.max(slopes):.3f}]")
    print(f"   Intercepts: mean={np.mean(intercepts):.3f}, std={np.std(intercepts):.3f}")
    print(f"               range=[{np.min(intercepts):.3f}, {np.max(intercepts):.3f}]")

    print("\n2. Q GAUGE PARAMETERS")
    print("-" * 40)
    print(f"   Offsets:    mean={np.mean(offsets):.3f} rad = {np.degrees(np.mean(offsets)):.1f}°")
    print(f"               std={np.std(offsets):.3f} rad = {np.degrees(np.std(offsets)):.1f}°")
    print(f"               range=[{np.min(offsets):.3f}, {np.max(offsets):.3f}] rad")

    print("\n3. PHYSICS PROPERTIES (should pass regardless of gauge)")
    print("-" * 40)
    cons_pass = sum(r.conservation_passed for r in results)
    evol_pass = sum(r.evolution_passed for r in results)
    symp_pass = sum(r.symplectic_passed for r in results)
    print(f"   Conservation: {cons_pass}/{len(results)} passed")
    print(f"   Evolution:    {evol_pass}/{len(results)} passed")
    print(f"   Symplectic:   {symp_pass}/{len(results)} passed")

    print("\n4. MONOTONICITY (P ordering preserved?)")
    print("-" * 40)
    spearmans = [r.P_spearman for r in results]
    print(f"   Spearman correlations: {[f'{s:.4f}' for s in spearmans]}")
    print(f"   All monotonic: {'✓ YES' if all(s > 0.95 for s in spearmans) else '✗ NO'}")

    # Determine outcome
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    slope_variation = np.std(slopes) / np.mean(slopes) if np.mean(slopes) != 0 else float('inf')
    intercept_variation = np.std(intercepts)
    offset_variation = np.std(offsets)

    all_physics_pass = (cons_pass == len(results) and
                        evol_pass == len(results) and
                        symp_pass == len(results))
    all_monotonic = all(s > 0.95 for s in spearmans)

    gauge_varies = (slope_variation > 0.1 or
                    intercept_variation > 0.1 or
                    offset_variation > 0.3)

    if all_physics_pass and all_monotonic and not gauge_varies:
        print("\n✓ Physics UNIQUELY determines (P, Q)")
        print("  All runs converge to same gauge")
    elif all_physics_pass and all_monotonic and gauge_varies:
        print("\n◐ Physics determines (P, Q) up to GAUGE FREEDOM")
        print("  P → aP + b (affine transform)")
        print("  Q → Q + c (constant offset)")
        print("  But conservation, evolution, symplectic all satisfied!")
    elif all_monotonic:
        print("\n△ Partial success")
        print("  P ordering correct (monotonic)")
        print("  Some physics tests failed - may need longer training")
    else:
        print("\n✗ Physics constraints INSUFFICIENT")
        print("  Network not learning valid action-angle coordinates")

    return {
        'slope_variation': slope_variation,
        'intercept_variation': intercept_variation,
        'offset_variation': offset_variation,
        'all_physics_pass': all_physics_pass,
        'all_monotonic': all_monotonic
    }


def create_gauge_visualization(results: List[GaugeAnalysis],
                                save_path: str = 'gauge_variation.png'):
    """Create visualization of gauge variation across seeds."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        n_seeds = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: P gauge variation
        slopes = [r.P_slope for r in results]
        intercepts = [r.P_intercept for r in results]
        offsets = [r.Q_offset for r in results]

        # Slope bar chart
        ax = axes[0, 0]
        ax.bar(range(n_seeds), slopes, color='steelblue')
        ax.axhline(y=1.0, color='red', linestyle='--', label='slope=1 (no scaling)')
        ax.set_xlabel('Seed')
        ax.set_ylabel('P slope')
        ax.set_title('P Gauge: Slope')
        ax.legend()

        # Intercept bar chart
        ax = axes[0, 1]
        ax.bar(range(n_seeds), intercepts, color='darkorange')
        ax.axhline(y=0.0, color='red', linestyle='--', label='intercept=0')
        ax.set_xlabel('Seed')
        ax.set_ylabel('P intercept')
        ax.set_title('P Gauge: Intercept')
        ax.legend()

        # Q offset bar chart
        ax = axes[0, 2]
        offsets_deg = [np.degrees(o) for o in offsets]
        ax.bar(range(n_seeds), offsets_deg, color='forestgreen')
        ax.axhline(y=0.0, color='red', linestyle='--', label='offset=0')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Q offset (degrees)')
        ax.set_title('Q Gauge: Offset')
        ax.legend()

        # Row 2: Correlations and physics
        pearsons = [r.P_pearson for r in results]
        spearmans = [r.P_spearman for r in results]

        ax = axes[1, 0]
        x = np.arange(n_seeds)
        width = 0.35
        ax.bar(x - width/2, pearsons, width, label='Pearson', color='steelblue')
        ax.bar(x + width/2, spearmans, width, label='Spearman', color='coral')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Correlation')
        ax.set_title('P Correlations')
        ax.legend()
        ax.set_ylim(0.9, 1.02)

        # Q correlations
        ax = axes[1, 1]
        cos_corrs = [r.Q_cos_corr for r in results]
        sin_corrs = [r.Q_sin_corr for r in results]
        ax.bar(x - width/2, cos_corrs, width, label='cos(Q)', color='steelblue')
        ax.bar(x + width/2, sin_corrs, width, label='sin(Q)', color='coral')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Correlation')
        ax.set_title('Q Correlations')
        ax.legend()
        ax.set_ylim(0.9, 1.02)

        # Physics pass/fail
        ax = axes[1, 2]
        cons = [1 if r.conservation_passed else 0 for r in results]
        evol = [1 if r.evolution_passed else 0 for r in results]
        symp = [1 if r.symplectic_passed else 0 for r in results]

        x = np.arange(n_seeds)
        width = 0.25
        ax.bar(x - width, cons, width, label='Conservation', color='green', alpha=0.7)
        ax.bar(x, evol, width, label='Evolution', color='blue', alpha=0.7)
        ax.bar(x + width, symp, width, label='Symplectic', color='purple', alpha=0.7)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Pass (1) / Fail (0)')
        ax.set_title('Physics Tests')
        ax.legend()
        ax.set_ylim(0, 1.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n[Saved gauge visualization to {save_path}]")

    except ImportError:
        print("\n[matplotlib not available - skipping visualization]")


if __name__ == "__main__":
    # Run multi-seed experiment
    results = run_multi_seed_experiment(n_seeds=5, n_epochs=3000)

    # Analyze gauge freedom
    analysis = analyze_gauge_freedom(results)

    # Create visualization
    create_gauge_visualization(results)

    # Final summary table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Seed':>4} | {'Slope':>8} | {'Intercept':>9} | {'Q_offset':>8} | {'Spearman':>8} | {'Physics':>8}")
    print("-" * 70)
    for r in results:
        physics_ok = '✓' if (r.conservation_passed and r.evolution_passed and r.symplectic_passed) else '✗'
        print(f"{r.seed:4d} | {r.P_slope:8.3f} | {r.P_intercept:9.3f} | {np.degrees(r.Q_offset):7.1f}° | {r.P_spearman:8.4f} | {physics_ok:>8}")
