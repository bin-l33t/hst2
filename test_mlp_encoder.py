"""
Test if MLP encoder fixes the pendulum forward error.

Compare:
- Linear W: β → (p, cos q, sin q)  [current: ~0.3 error]
- MLP encoder: β → (p, cos q, sin q)  [expected: much lower]

Also run seam and holonomy tests to understand the manifold structure.
"""

import numpy as np
import torch
import sys

from hst_rom import HST_ROM
from beta_encoder import BetaToStateEncoder, train_beta_encoder, evaluate_encoder
from manifold_diagnostics import seam_test, holonomy_test, run_extended_diagnostics


def generate_pendulum_signal(p0, q0, omega0=1.0, dt=0.01, n_points=128):
    """Generate pendulum signal via ODE integration."""
    from scipy.integrate import solve_ivp

    def pendulum_dynamics(t, y):
        q, p = y
        return [p, -omega0**2 * np.sin(q)]

    t_span = (0, (n_points - 1) * dt)
    t_eval = np.linspace(0, (n_points - 1) * dt, n_points)

    sol = solve_ivp(
        pendulum_dynamics, t_span, [q0, p0],
        t_eval=t_eval, method='DOP853', rtol=1e-10
    )

    q_t = sol.y[0]
    p_t = sol.y[1]
    z = q_t + 1j * p_t
    return z, p_t, q_t


def generate_rotation_ic(E_min=1.5, E_max=3.0, omega0=1.0):
    """Generate initial conditions in pendulum rotation regime."""
    E = np.random.uniform(E_min, E_max) * omega0**2
    q0 = np.random.uniform(0, 2 * np.pi)
    p0 = np.sqrt(2 * (E + omega0**2 * np.cos(q0)))
    if np.random.random() < 0.5:
        p0 = -p0
    return p0, q0, E


def test_mlp_encoder():
    """
    Test if MLP encoder fixes the pendulum forward error.

    Compare linear vs MLP encoder on β → (p, cos q, sin q).
    """
    print("=" * 70)
    print("MLP ENCODER TEST: Linear W vs MLP for β → (p, cos q, sin q)")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    omega0 = 1.0
    n_samples = 300
    window_size = 128
    n_pca = 8
    dt = 0.01

    # Train/test split
    n_train = int(0.8 * n_samples)
    all_idx = np.arange(n_samples)
    np.random.shuffle(all_idx)
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train:]

    print(f"\nData split: {n_train} train, {len(test_idx)} test")

    # Generate pendulum rotation data
    print("\n[1] Generating pendulum rotation data...")
    signals = []
    p_true = []
    q_true = []
    center_idx = window_size // 2

    for i in range(n_samples):
        p0, q0, E = generate_rotation_ic(E_min=1.5, E_max=3.0, omega0=omega0)
        z, p_t, q_t = generate_pendulum_signal(p0, q0, omega0=omega0,
                                                dt=dt, n_points=window_size)
        signals.append(z)
        p_true.append(p_t[center_idx])
        q_true.append(q_t[center_idx])

    signals = np.array(signals)
    p_true = np.array(p_true)
    q_true = np.array(q_true)

    # Fit HST_ROM
    print("\n[2] Fitting HST_ROM...")
    hst_rom = HST_ROM(n_components=n_pca, J=3, window_size=window_size)
    train_signals = [signals[i] for i in train_idx]
    beta_train = hst_rom.fit(train_signals, extract_windows=False)
    beta_test = np.array([hst_rom.transform(signals[i]) for i in test_idx])

    print(f"  β_train shape: {beta_train.shape}")
    print(f"  β_test shape: {beta_test.shape}")

    # Prepare target: (p, cos q, sin q)
    pcs_train = np.stack([
        p_true[train_idx],
        np.cos(q_true[train_idx]),
        np.sin(q_true[train_idx])
    ], axis=1)
    pcs_test = np.stack([
        p_true[test_idx],
        np.cos(q_true[test_idx]),
        np.sin(q_true[test_idx])
    ], axis=1)

    # ========================================
    # LINEAR BASELINE
    # ========================================
    print("\n[3] Linear baseline: β → (p, cos q, sin q)...")
    W, _, _, _ = np.linalg.lstsq(beta_train, pcs_train, rcond=None)
    pcs_pred_linear_train = beta_train @ W
    pcs_pred_linear_test = beta_test @ W

    linear_train_error = np.mean(np.abs(pcs_pred_linear_train - pcs_train))
    linear_test_error = np.mean(np.abs(pcs_pred_linear_test - pcs_test))

    # Per-component errors
    linear_p_error = np.mean(np.abs(pcs_pred_linear_test[:, 0] - pcs_test[:, 0]))
    linear_cos_error = np.mean(np.abs(pcs_pred_linear_test[:, 1] - pcs_test[:, 1]))
    linear_sin_error = np.mean(np.abs(pcs_pred_linear_test[:, 2] - pcs_test[:, 2]))

    # Angular error
    q_pred_linear = np.arctan2(pcs_pred_linear_test[:, 2], pcs_pred_linear_test[:, 1])
    q_true_test = np.arctan2(pcs_test[:, 2], pcs_test[:, 1])
    q_diff_linear = np.arctan2(np.sin(q_pred_linear - q_true_test),
                               np.cos(q_pred_linear - q_true_test))
    linear_q_error = np.mean(np.abs(q_diff_linear))

    print(f"  Linear TRAIN error: {linear_train_error:.4f}")
    print(f"  Linear TEST error: {linear_test_error:.4f}")
    print(f"    p error: {linear_p_error:.4f}")
    print(f"    cos q error: {linear_cos_error:.4f}")
    print(f"    sin q error: {linear_sin_error:.4f}")
    print(f"    q error: {linear_q_error:.4f} rad ({np.degrees(linear_q_error):.1f} deg)")

    # ========================================
    # MLP ENCODER
    # ========================================
    print("\n[4] MLP encoder: β → (p, cos q, sin q)...")
    encoder = BetaToStateEncoder(n_beta=n_pca, hidden_dim=64, n_layers=3)
    train_beta_encoder(encoder, beta_train, pcs_train, epochs=2000, lr=1e-3, verbose=True)

    # Evaluate
    mlp_metrics = evaluate_encoder(encoder, beta_test, pcs_test)

    print(f"\n  MLP TEST error (MAE): {mlp_metrics['mae']:.4f}")
    print(f"    p error: {mlp_metrics['p_error']:.4f}")
    print(f"    cos q error: {mlp_metrics['cos_q_error']:.4f}")
    print(f"    sin q error: {mlp_metrics['sin_q_error']:.4f}")
    print(f"    q error: {mlp_metrics['q_error_rad']:.4f} rad ({mlp_metrics['q_error_deg']:.1f} deg)")

    # Also evaluate on train
    mlp_train_metrics = evaluate_encoder(encoder, beta_train, pcs_train)
    print(f"\n  MLP TRAIN error (MAE): {mlp_train_metrics['mae']:.4f}")

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON: LINEAR vs MLP")
    print("=" * 70)

    improvement = linear_test_error / mlp_metrics['mae']
    print(f"\n{'Metric':<20} | {'Linear':<12} | {'MLP':<12} | {'Improvement'}")
    print("-" * 60)
    print(f"{'MAE (test)':<20} | {linear_test_error:<12.4f} | {mlp_metrics['mae']:<12.4f} | {improvement:.1f}x")
    print(f"{'p error':<20} | {linear_p_error:<12.4f} | {mlp_metrics['p_error']:<12.4f} | {linear_p_error/mlp_metrics['p_error']:.1f}x")
    print(f"{'q error (rad)':<20} | {linear_q_error:<12.4f} | {mlp_metrics['q_error_rad']:<12.4f} | {linear_q_error/mlp_metrics['q_error_rad']:.1f}x")

    # ========================================
    # ADDITIONAL DIAGNOSTICS
    # ========================================
    print("\n" + "=" * 70)
    print("ADDITIONAL DIAGNOSTICS: Seam and Holonomy")
    print("=" * 70)

    # Seam test
    print("\n[5] Seam test (branch-cut detection)...")
    seam = seam_test(beta_train)
    print(f"  Max angle jump: {seam['max_jump']:.3f} rad ({np.degrees(seam['max_jump']):.1f} deg)")
    print(f"  95th percentile: {seam['p95_jump']:.3f} rad ({np.degrees(seam['p95_jump']):.1f} deg)")
    print(f"  Has seam: {seam['has_seam']}")

    # Holonomy test
    print("\n[6] Holonomy test (orientation consistency)...")
    holonomy = holonomy_test(beta_train)
    print(f"  Mean inconsistency: {holonomy['mean_inconsistency']:.4f}")
    print(f"  Max inconsistency: {holonomy['max_inconsistency']:.4f}")
    print(f"  Needs atlas: {holonomy['needs_atlas']}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mlp_wins = mlp_metrics['mae'] < linear_test_error
    significant_improvement = improvement > 2.0

    print(f"\nLinear W error: {linear_test_error:.4f}")
    print(f"MLP encoder error: {mlp_metrics['mae']:.4f}")
    print(f"Improvement: {improvement:.1f}x")

    print(f"\nSeam test: has_seam={seam['has_seam']}")
    print(f"Holonomy test: needs_atlas={holonomy['needs_atlas']}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if significant_improvement:
        print(f"  MLP encoder provides {improvement:.1f}x improvement over linear W")
        print("  → The β → (p, cos q, sin q) relationship is nonlinear")
        print("  → Use MLP encoder in the full pipeline")
    else:
        print("  MLP encoder does not significantly improve over linear W")
        print("  → The issue may be elsewhere (decoder, HJB encoder, etc.)")

    if seam['has_seam']:
        print("  Seam detected → Using (sin, cos) embedding is correct")
    else:
        print("  No seam detected → Simple angle embedding might work too")

    if holonomy['needs_atlas']:
        print("  Holonomy inconsistency → Consider multi-chart approach")
    else:
        print("  No holonomy issues → Single-chart MLP is sufficient")

    print("-" * 70)

    # Success criteria
    success = significant_improvement and not holonomy['needs_atlas']

    print("\n" + "=" * 70)
    if success:
        print("TEST PASSED: MLP encoder significantly improves forward path")
        print("NEXT STEP: Integrate MLP encoder into full pendulum pipeline")
    else:
        print("TEST RESULT: Further investigation needed")
    print("=" * 70)

    return {
        'linear_error': linear_test_error,
        'mlp_error': mlp_metrics['mae'],
        'improvement': improvement,
        'seam': seam,
        'holonomy': holonomy,
        'success': success,
    }


if __name__ == "__main__":
    results = test_mlp_encoder()
    sys.exit(0 if results['success'] else 1)
