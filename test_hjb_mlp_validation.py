"""
Test HJB_MLP Against Ground Truth

This script:
1. Trains HJB_MLP on SHO trajectories
2. Validates against our established ground truth
3. Diagnoses issues and suggests fixes

Key insight from our work:
- Action P = (p² + ω²q²)/(2ω) should be conserved
- Angle Q = arctan2(ωq, p) should evolve at rate ω
- The transformation must be symplectic: {P, Q} = 1
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple

# Import our validated utilities
from action_angle_utils import wrap_to_2pi, angular_distance

# Import HJB_MLP
from hjb_mlp import HJB_MLP, train_on_sho, evaluate_action_angle

# Import validation framework
from validate_hjb_mlp import validate_hjb_mlp, sho_ground_truth


def create_hjb_mlp_wrapper(model: HJB_MLP, device: str = 'cpu'):
    """
    Create a wrapper that adapts HJB_MLP to our validation interface.

    HJB_MLP.encode expects (p, q) and returns (P, Q).
    Validation framework expects (q, p) → (P, Q).
    """
    model = model.to(device)
    model.eval()

    def wrapper(q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper: (q, p) → (P, Q)"""
        with torch.no_grad():
            p_t = torch.tensor(p, dtype=torch.float32, device=device)
            q_t = torch.tensor(q, dtype=torch.float32, device=device)

            # HJB_MLP.encode takes (p, q)
            P_t, Q_t = model.encode(p_t, q_t)

            P = P_t.cpu().numpy()
            Q = Q_t.cpu().numpy()

            # Wrap Q to [0, 2π)
            Q = wrap_to_2pi(Q)

        return P, Q

    return wrapper


def diagnose_hjb_mlp(model: HJB_MLP, omega: float = 1.0, n_test: int = 200):
    """
    Detailed diagnosis of HJB_MLP learning.
    """
    print("\n" + "=" * 70)
    print("DETAILED HJB_MLP DIAGNOSIS")
    print("=" * 70)

    device = next(model.parameters()).device

    # Generate test data with known ground truth
    np.random.seed(42)
    E_range = np.random.uniform(0.5, 5.0, n_test)
    Q_true = np.random.uniform(0, 2 * np.pi, n_test)
    P_true = E_range / omega

    # Get (p, q) from ground truth
    p = np.sqrt(2 * E_range) * np.sin(Q_true)
    q = np.sqrt(2 * E_range / omega**2) * np.cos(Q_true)

    # Get MLP predictions
    model.eval()
    with torch.no_grad():
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)
        P_pred_t, Q_pred_t = model.encode(p_t, q_t)
        P_pred = P_pred_t.cpu().numpy()
        Q_pred = Q_pred_t.cpu().numpy()

    # Wrap Q
    Q_pred = wrap_to_2pi(Q_pred)

    # 1. Analyze P learning
    print("\n[1] ACTION P ANALYSIS")
    print("-" * 40)

    from scipy.stats import pearsonr, spearmanr

    r_P, _ = pearsonr(P_pred, P_true)
    rho_P, _ = spearmanr(P_pred, P_true)

    # Check if P is a linear transform of true P
    slope, intercept = np.polyfit(P_true, P_pred, 1)

    print(f"  Pearson correlation: r = {r_P:.4f}")
    print(f"  Spearman correlation: ρ = {rho_P:.4f}")
    print(f"  Linear fit: P_pred = {slope:.3f} * P_true + {intercept:.3f}")
    print(f"  P_pred range: [{P_pred.min():.2f}, {P_pred.max():.2f}]")
    print(f"  P_true range: [{P_true.min():.2f}, {P_true.max():.2f}]")

    if r_P > 0.95:
        print("  ✓ P is well-learned (up to linear transform)")
    elif r_P > 0.8:
        print("  ~ P is partially learned, may need more training")
    else:
        print("  ✗ P is NOT learned correctly")

    # 2. Analyze Q learning
    print("\n[2] ANGLE Q ANALYSIS")
    print("-" * 40)

    # Circular correlation
    r_cos, _ = pearsonr(np.cos(Q_pred), np.cos(Q_true))
    r_sin, _ = pearsonr(np.sin(Q_pred), np.sin(Q_true))

    # Mean angular error
    Q_errors = angular_distance(Q_pred, Q_true)
    mean_Q_error = np.mean(Q_errors)

    print(f"  cos(Q) correlation: r = {r_cos:.4f}")
    print(f"  sin(Q) correlation: r = {r_sin:.4f}")
    print(f"  Mean angular error: {mean_Q_error:.4f} rad = {np.degrees(mean_Q_error):.1f}°")
    print(f"  Q_pred range: [{Q_pred.min():.2f}, {Q_pred.max():.2f}]")

    if r_cos > 0.9 and r_sin > 0.9:
        print("  ✓ Q is well-learned")
    elif mean_Q_error < 0.5:
        print("  ~ Q is partially learned")
    else:
        print("  ✗ Q is NOT learned correctly")
        print("     ISSUE: HJBLoss uses MSE for Q, but Q is circular!")
        print("     FIX: Use angular loss: 1 - cos(Q_pred - Q_true)")

    # 3. Analyze conservation
    print("\n[3] CONSERVATION ANALYSIS")
    print("-" * 40)

    # Evolve trajectories
    dt = 1.0
    p_evolved = p * np.cos(omega * dt) - omega * q * np.sin(omega * dt)
    q_evolved = q * np.cos(omega * dt) + p / omega * np.sin(omega * dt)

    with torch.no_grad():
        p_ev_t = torch.tensor(p_evolved, dtype=torch.float32, device=device)
        q_ev_t = torch.tensor(q_evolved, dtype=torch.float32, device=device)
        P_evolved_t, _ = model.encode(p_ev_t, q_ev_t)
        P_evolved = P_evolved_t.cpu().numpy()

    P_change = np.abs(P_evolved - P_pred)
    P_rel_change = P_change / (np.abs(P_pred) + 1e-10)

    print(f"  Mean |ΔP| after dt=1: {np.mean(P_change):.4f}")
    print(f"  Mean |ΔP|/|P|: {np.mean(P_rel_change):.4f}")

    if np.mean(P_rel_change) < 0.01:
        print("  ✓ P is conserved along trajectories")
    elif np.mean(P_rel_change) < 0.05:
        print("  ~ P is approximately conserved")
    else:
        print("  ✗ P is NOT conserved - varies along trajectory!")

    # 4. Analyze symplectic structure (Jacobian)
    print("\n[4] SYMPLECTIC STRUCTURE")
    print("-" * 40)

    # Compute Jacobian at a few points
    eps = 1e-4
    n_check = 20
    poisson_brackets = []

    for i in range(n_check):
        pi, qi = p[i], q[i]

        # Finite differences
        with torch.no_grad():
            def encode_point(pi, qi):
                pt = torch.tensor([pi], dtype=torch.float32, device=device)
                qt = torch.tensor([qi], dtype=torch.float32, device=device)
                Pt, Qt = model.encode(pt, qt)
                return Pt.item(), Qt.item()

            P_pp, Q_pp = encode_point(pi + eps, qi)
            P_pm, Q_pm = encode_point(pi - eps, qi)
            P_qp, Q_qp = encode_point(pi, qi + eps)
            P_qm, Q_qm = encode_point(pi, qi - eps)

        dP_dp = (P_pp - P_pm) / (2 * eps)
        dP_dq = (P_qp - P_qm) / (2 * eps)
        dQ_dp = (Q_pp - Q_pm) / (2 * eps)
        dQ_dq = (Q_qp - Q_qm) / (2 * eps)

        # Poisson bracket {P, Q} = ∂P/∂q · ∂Q/∂p - ∂P/∂p · ∂Q/∂q
        pb = dP_dq * dQ_dp - dP_dp * dQ_dq
        poisson_brackets.append(pb)

    pb_mean = np.mean(poisson_brackets)
    pb_std = np.std(poisson_brackets)
    pb_error = np.mean(np.abs(np.array(poisson_brackets) - 1.0))

    print(f"  Poisson bracket {{P, Q}} mean: {pb_mean:.4f}")
    print(f"  Poisson bracket std: {pb_std:.4f}")
    print(f"  Error from canonical (=1): {pb_error:.4f}")

    if pb_error < 0.1:
        print("  ✓ Transformation is approximately symplectic")
    else:
        print("  ✗ Transformation is NOT symplectic!")
        print("     ISSUE: HJBLoss doesn't enforce {P, Q} = 1")
        print("     FIX: Add Poisson bracket loss term")

    # Create diagnostic plots
    create_diagnostic_plots(P_true, P_pred, Q_true, Q_pred,
                           P_pred, P_evolved, poisson_brackets)

    return {
        'P_corr': r_P,
        'Q_cos_corr': r_cos,
        'Q_sin_corr': r_sin,
        'Q_mean_error': mean_Q_error,
        'P_rel_change': np.mean(P_rel_change),
        'poisson_error': pb_error
    }


def create_diagnostic_plots(P_true, P_pred, Q_true, Q_pred,
                           P_init, P_evolved, poisson_brackets):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. P_pred vs P_true
    ax = axes[0, 0]
    ax.scatter(P_true, P_pred, alpha=0.5, s=10)
    lims = [0, max(P_true.max(), P_pred.max()) * 1.1]
    ax.plot(lims, lims, 'r--', label='y=x')
    ax.set_xlabel('P_true (ground truth)')
    ax.set_ylabel('P_pred (MLP)')
    ax.set_title('Action Learning')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q on unit circle
    ax = axes[0, 1]
    ax.scatter(np.cos(Q_true), np.sin(Q_true), alpha=0.3, s=10, label='True')
    ax.scatter(np.cos(Q_pred), np.sin(Q_pred), alpha=0.3, s=10, label='Pred')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('cos(Q)')
    ax.set_ylabel('sin(Q)')
    ax.set_title('Angle on Unit Circle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Q_pred vs Q_true
    ax = axes[0, 2]
    ax.scatter(Q_true, Q_pred, alpha=0.5, s=10)
    ax.plot([0, 2*np.pi], [0, 2*np.pi], 'r--', label='y=x')
    ax.set_xlabel('Q_true')
    ax.set_ylabel('Q_pred')
    ax.set_title('Angle Learning\n(should follow y=x with wrapping)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. P conservation
    ax = axes[1, 0]
    P_change = np.abs(P_evolved - P_init) / (np.abs(P_init) + 1e-10)
    ax.hist(P_change, bins=30, edgecolor='black')
    ax.axvline(x=0.01, color='r', linestyle='--', label='1% threshold')
    ax.set_xlabel('|ΔP|/|P|')
    ax.set_ylabel('Count')
    ax.set_title('Action Conservation\n(should be near 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Poisson brackets
    ax = axes[1, 1]
    ax.hist(poisson_brackets, bins=20, edgecolor='black')
    ax.axvline(x=1.0, color='r', linestyle='--', label='Canonical: {P,Q}=1')
    ax.axvline(x=np.mean(poisson_brackets), color='g', linestyle='-',
               label=f'Mean: {np.mean(poisson_brackets):.2f}')
    ax.set_xlabel('{P, Q}')
    ax.set_ylabel('Count')
    ax.set_title('Symplectic Structure')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    from scipy.stats import pearsonr
    r_P, _ = pearsonr(P_pred, P_true)
    r_cos, _ = pearsonr(np.cos(Q_pred), np.cos(Q_true))
    r_sin, _ = pearsonr(np.sin(Q_pred), np.sin(Q_true))

    summary = f"""
    HJB_MLP VALIDATION SUMMARY

    ACTION P:
      Correlation: {r_P:.3f}
      (should be > 0.95)

    ANGLE Q:
      cos correlation: {r_cos:.3f}
      sin correlation: {r_sin:.3f}
      (should be > 0.9)

    CONSERVATION:
      Mean |ΔP|/|P|: {np.mean(P_change):.4f}
      (should be < 0.01)

    SYMPLECTIC:
      Mean {{P,Q}}: {np.mean(poisson_brackets):.3f}
      (should be ≈ 1.0)
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('hjb_mlp_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: hjb_mlp_diagnosis.png")
    plt.close()


def run_full_validation(n_epochs: int = 2000, omega: float = 1.0):
    """
    Run full HJB_MLP validation pipeline.
    """
    print("=" * 70)
    print("HJB_MLP VALIDATION AGAINST GROUND TRUTH")
    print("=" * 70)
    print(f"\nGround truth for SHO (ω = {omega}):")
    print("  P = (p² + ω²q²)/(2ω)")
    print("  Q = arctan2(ωq, p)")

    # Create model
    model = HJB_MLP(hidden_dim=64, num_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: HJB_MLP with {n_params} parameters")

    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    losses = train_on_sho(model, n_epochs=n_epochs, n_trajectories=100, omega0=omega)
    print(f"Final loss: {losses[-1]['total']:.6f}")

    # Diagnose
    diagnosis = diagnose_hjb_mlp(model, omega=omega)

    # Also run the formal validation suite
    print("\n" + "-" * 70)
    print("FORMAL VALIDATION SUITE")
    print("-" * 70)

    wrapper = create_hjb_mlp_wrapper(model)
    results = validate_hjb_mlp(wrapper, system='sho', omega=omega, verbose=True)

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    issues = []

    if diagnosis['P_corr'] < 0.95:
        issues.append("P accuracy")

    if diagnosis['Q_mean_error'] > 0.2:
        issues.append("Q accuracy (circular)")

    if diagnosis['P_rel_change'] > 0.01:
        issues.append("P conservation")

    if diagnosis['poisson_error'] > 0.1:
        issues.append("Symplectic structure")

    if not issues:
        print("\n✓ HJB_MLP successfully learns canonical transformation!")
    else:
        print(f"\nIdentified issues: {', '.join(issues)}")
        print("\nRecommended fixes for HJBLoss:")

        if "Q accuracy (circular)" in issues:
            print("""
1. ANGULAR LOSS FOR Q:
   Current: MSE loss (Q_pred - Q_true)²
   Problem: Doesn't handle wraparound (Q=0 and Q=2π are the same!)

   Fix: Use angular loss
   ```python
   angle_loss = 1 - torch.cos(Q_pred - Q_true)
   # Or use two outputs: (sin(Q), cos(Q))
   ```
""")

        if "Symplectic structure" in issues:
            print("""
2. POISSON BRACKET CONSTRAINT:
   Canonical transformation requires {P, Q} = 1

   Fix: Add Jacobian constraint to loss
   ```python
   # Compute Jacobian via autograd
   J = torch.autograd.functional.jacobian(model.encode, (p, q))
   det_J = J[0,0]*J[1,1] - J[0,1]*J[1,0]
   symplectic_loss = (det_J - 1)**2
   ```
""")

        if "P conservation" in issues:
            print("""
3. CONSERVATION ENFORCEMENT:
   Current conservation loss may need higher weight or better formulation.

   Consider:
   - Increase conservation_weight in HJBLoss
   - Train on longer trajectories
   - Add explicit P = const constraint for multiple time points
""")

    return model, diagnosis, results


if __name__ == "__main__":
    model, diagnosis, results = run_full_validation(n_epochs=2000)
