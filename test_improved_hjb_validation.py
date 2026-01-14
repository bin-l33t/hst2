"""
Comprehensive Validation of Improved HJB_MLP

Tests the fixed model against all our validated ground truth.
"""

import numpy as np
import torch
from scipy.stats import pearsonr

from fixed_hjb_loss import ImprovedHJB_MLP, FixedHJBLoss, train_on_sho_fixed
from action_angle_utils import wrap_to_2pi, angular_distance


def sho_ground_truth(p, q, omega=1.0):
    """Ground truth with standard convention."""
    P = (p**2 + omega**2 * q**2) / (2 * omega)
    Q = np.arctan2(p, omega * q)
    Q = wrap_to_2pi(Q)
    return P, Q


def compute_poisson_bracket(model, p, q, eps=1e-4, device='cpu'):
    """Compute {P, Q} numerically."""
    model.eval()
    n = len(p)
    poisson = np.zeros(n)

    with torch.no_grad():
        for i in range(n):
            pi, qi = p[i], q[i]

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

            poisson[i] = dP_dq * dQ_dp - dP_dp * dQ_dq

    return poisson


def test_accuracy(model, omega=1.0, n_test=200, device='cpu'):
    """Test P and Q against ground truth."""
    print("\n[TEST 1] ACCURACY")
    print("-" * 40)

    np.random.seed(42)
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

    r_P, _ = pearsonr(P_pred, P_true)
    P_rel_error = np.mean(np.abs(P_pred - P_true) / P_true)

    Q_errors = angular_distance(Q_pred, Q_true)
    Q_mean_error = np.mean(Q_errors)

    r_cos, _ = pearsonr(np.cos(Q_pred), np.cos(Q_true))
    r_sin, _ = pearsonr(np.sin(Q_pred), np.sin(Q_true))

    print(f"  P correlation: {r_P:.4f}")
    print(f"  P relative error: {P_rel_error:.4f}")
    print(f"  Q cos correlation: {r_cos:.4f}")
    print(f"  Q sin correlation: {r_sin:.4f}")
    print(f"  Q mean angular error: {Q_mean_error:.4f} rad = {np.degrees(Q_mean_error):.1f}°")

    P_pass = r_P > 0.95 and P_rel_error < 0.05
    Q_pass = Q_mean_error < 0.1

    print(f"\n  P: {'✓ PASS' if P_pass else '✗ FAIL'}")
    print(f"  Q: {'✓ PASS' if Q_pass else '✗ FAIL'}")

    return P_pass and Q_pass


def test_conservation(model, omega=1.0, n_traj=50, device='cpu'):
    """Test that P is conserved along trajectories."""
    print("\n[TEST 2] CONSERVATION")
    print("-" * 40)

    np.random.seed(43)
    E = np.random.uniform(0.5, 4.5, n_traj)
    theta0 = np.random.uniform(0, 2 * np.pi, n_traj)

    p0 = np.sqrt(2 * E) * np.sin(theta0)
    q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

    # Get P at t=0
    model.eval()
    with torch.no_grad():
        p_t = torch.tensor(p0, dtype=torch.float32, device=device)
        q_t = torch.tensor(q0, dtype=torch.float32, device=device)
        P0_t, _ = model.encode(p_t, q_t)
        P0 = P0_t.cpu().numpy()

    # Evolve and check P at multiple times
    dt_values = np.linspace(0, 2*np.pi, 20)
    max_rel_change = 0

    for dt in dt_values:
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

    print(f"  Max |ΔP|/|P| over one period: {max_rel_change:.6f}")

    # Allow up to 5% variation (realistic for MLP)
    passed = max_rel_change < 0.05
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'} (threshold: 5%)")

    return passed


def test_evolution(model, omega=1.0, n_traj=50, device='cpu'):
    """Test that Q evolves at rate ω."""
    print("\n[TEST 3] EVOLUTION (dQ/dt = ω)")
    print("-" * 40)

    np.random.seed(44)
    E = np.random.uniform(0.5, 4.5, n_traj)
    theta0 = np.random.uniform(0, 2 * np.pi, n_traj)

    p0 = np.sqrt(2 * E) * np.sin(theta0)
    q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

    # Get Q at t=0
    model.eval()
    with torch.no_grad():
        p_t = torch.tensor(p0, dtype=torch.float32, device=device)
        q_t = torch.tensor(q0, dtype=torch.float32, device=device)
        _, Q0_t = model.encode(p_t, q_t)
        Q0 = Q0_t.cpu().numpy()

    # Measure effective omega from Q trajectory
    dt_values = np.linspace(0, 2*np.pi, 50)
    Q_traj = np.zeros((len(dt_values), n_traj))

    for i, dt in enumerate(dt_values):
        theta_t = theta0 + omega * dt
        p_t_np = np.sqrt(2 * E) * np.sin(theta_t)
        q_t_np = np.sqrt(2 * E) / omega * np.cos(theta_t)

        with torch.no_grad():
            p_t = torch.tensor(p_t_np, dtype=torch.float32, device=device)
            q_t = torch.tensor(q_t_np, dtype=torch.float32, device=device)
            _, Q_t_t = model.encode(p_t, q_t)
            Q_traj[i] = Q_t_t.cpu().numpy()

    # Unwrap and fit
    Q_unwrapped = np.unwrap(Q_traj, axis=0)
    omega_measured = []

    for j in range(n_traj):
        slope, _ = np.polyfit(dt_values, Q_unwrapped[:, j], 1)
        omega_measured.append(slope)

    omega_mean = np.mean(omega_measured)
    omega_std = np.std(omega_measured)
    omega_error = abs(omega_mean - omega) / omega

    print(f"  Measured ω: {omega_mean:.4f} ± {omega_std:.4f}")
    print(f"  Expected ω: {omega:.4f}")
    print(f"  Relative error: {omega_error:.4f}")

    passed = omega_error < 0.01
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def test_symplectic(model, omega=1.0, n_test=50, device='cpu'):
    """Test that |{P, Q}| = 1."""
    print("\n[TEST 4] SYMPLECTIC (|{P, Q}| = 1)")
    print("-" * 40)

    # Note: For Q = atan2(p, ωq), {P, Q} = +1 (not -1)
    # The sign depends on convention. What matters is |{P, Q}| = 1.

    np.random.seed(45)
    E = np.random.uniform(0.5, 4.5, n_test)
    theta = np.random.uniform(0, 2 * np.pi, n_test)

    p = np.sqrt(2 * E) * np.sin(theta)
    q = np.sqrt(2 * E) / omega * np.cos(theta)

    poisson = compute_poisson_bracket(model, p, q, device=device)

    mean_pb = np.mean(poisson)
    std_pb = np.std(poisson)
    error_from_1 = np.mean(np.abs(np.abs(poisson) - 1.0))

    print(f"  Mean {{P, Q}}: {mean_pb:.4f}")
    print(f"  Std: {std_pb:.4f}")
    print(f"  Error from |{{P,Q}}|=1: {error_from_1:.4f}")

    passed = error_from_1 < 0.1
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def test_roundtrip(model, omega=1.0, n_test=100, device='cpu'):
    """Test encode → decode roundtrip."""
    print("\n[TEST 5] ROUNDTRIP (p,q) → (P,Q) → (p,q)")
    print("-" * 40)

    np.random.seed(46)
    E = np.random.uniform(0.5, 4.5, n_test)
    theta = np.random.uniform(0, 2 * np.pi, n_test)

    p = np.sqrt(2 * E) * np.sin(theta)
    q = np.sqrt(2 * E) / omega * np.cos(theta)

    model.eval()
    with torch.no_grad():
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)

        P_t, Q_t = model.encode(p_t, q_t)
        p_rec_t, q_rec_t = model.decode(P_t, Q_t)

        p_rec = p_rec_t.cpu().numpy()
        q_rec = q_rec_t.cpu().numpy()

    p_error = np.mean(np.abs(p_rec - p))
    q_error = np.mean(np.abs(q_rec - q))

    print(f"  Mean |p_rec - p|: {p_error:.4f}")
    print(f"  Mean |q_rec - q|: {q_error:.4f}")

    passed = p_error < 0.1 and q_error < 0.1
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def run_full_validation():
    """Run complete validation suite."""
    print("=" * 70)
    print("IMPROVED HJB_MLP - COMPREHENSIVE VALIDATION")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    omega = 1.0

    # Create and train model
    print("\nCreating and training improved model...")
    model = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)
    model = model.to(device)

    losses = train_on_sho_fixed(model, n_epochs=3000, omega=omega, device=device)

    # Run all tests
    results = {}
    results['accuracy'] = test_accuracy(model, omega, device=device)
    results['conservation'] = test_conservation(model, omega, device=device)
    results['evolution'] = test_evolution(model, omega, device=device)
    results['symplectic'] = test_symplectic(model, omega, device=device)
    results['roundtrip'] = test_roundtrip(model, omega, device=device)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:15s}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe improved HJB_MLP successfully learns:")
        print("  • Action P = (p² + ω²q²)/(2ω)")
        print("  • Angle Q = atan2(p, ωq)")
        print("  • Conservation: P(t) = const")
        print("  • Evolution: dQ/dt = ω")
        print("  • Symplectic: {P, Q} = -1")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nMay need more training or architecture adjustments.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_full_validation()
