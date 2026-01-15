"""
Test: Can we connect HST_ROM output to ImprovedHJB_MLP?

Hypothesis: PCA of HST coefficients extracts (p, q)-like coordinates
that ImprovedHJB_MLP can transform to (P, Q).

Key question: Does β from HST_ROM correlate with phase space (p, q)?
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import orthogonal_procrustes
import warnings
warnings.filterwarnings('ignore')

from hst_rom import HST_ROM
from action_angle_utils import wrap_to_2pi


def generate_sho_signal(p0: float, q0: float, omega: float = 1.0,
                        dt: float = 0.01, n_points: int = 256) -> np.ndarray:
    """
    Generate complex SHO signal z(t) = q(t) + i*p(t)/omega.

    This is the analytic signal representation.
    """
    t = np.arange(n_points) * dt

    # SHO evolution
    E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
    theta0 = np.arctan2(p0, omega * q0)

    # Evolve
    theta_t = theta0 + omega * t
    p_t = np.sqrt(2 * E) * np.sin(theta_t)
    q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)

    # Analytic signal (complex representation)
    z = q_t + 1j * p_t / omega

    return z, p_t, q_t


def test_beta_pq_correlation():
    """
    Test correlation between HST_ROM output β and phase space (p, q).

    Steps:
    1. Generate SHO trajectories with known (p, q)
    2. Compute HST → PCA → β
    3. Check correlation: r(β[:, i], p), r(β[:, i], q)
    """
    print("=" * 70)
    print("TEST: β vs (p, q) CORRELATION")
    print("=" * 70)

    omega = 1.0
    n_trajectories = 100
    window_size = 128

    # Generate trajectories
    np.random.seed(42)
    trajectories = []
    ground_truth = []  # (p, q, P, Q) at window center

    for i in range(n_trajectories):
        # Random initial conditions
        E = np.random.uniform(0.5, 4.5)
        theta0 = np.random.uniform(0, 2 * np.pi)

        p0 = np.sqrt(2 * E) * np.sin(theta0)
        q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

        # Generate signal
        z, p_t, q_t = generate_sho_signal(p0, q0, omega=omega, n_points=window_size)
        trajectories.append(z)

        # Ground truth at window center
        center_idx = window_size // 2
        p_center = p_t[center_idx]
        q_center = q_t[center_idx]
        P_true = E / omega
        Q_true = wrap_to_2pi(theta0 + omega * center_idx * 0.01)

        ground_truth.append([p_center, q_center, P_true, Q_true])

    ground_truth = np.array(ground_truth)
    p_true = ground_truth[:, 0]
    q_true = ground_truth[:, 1]
    P_true = ground_truth[:, 2]
    Q_true = ground_truth[:, 3]

    print(f"\nGenerated {n_trajectories} SHO trajectories")
    print(f"Window size: {window_size}")

    # Fit HST_ROM
    print("\nFitting HST_ROM...")
    rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=window_size)

    # Use full signals (not sliding windows)
    betas = rom.fit(trajectories, extract_windows=False)

    print(f"β shape: {betas.shape}")
    print(f"Variance explained: {rom.pca.explained_variance_ratio_[:4]}")
    print(f"Total variance (4 comp): {sum(rom.pca.explained_variance_ratio_[:4]):.3f}")

    # Correlation analysis
    print("\n" + "-" * 70)
    print("CORRELATION MATRIX: β components vs ground truth")
    print("-" * 70)

    print(f"\n{'β_i':>6} | {'p':>8} | {'q':>8} | {'P':>8} | {'Q':>8} | {'cos(Q)':>8} | {'sin(Q)':>8}")
    print("-" * 70)

    correlations = {
        'p': [], 'q': [], 'P': [], 'Q': [], 'cos_Q': [], 'sin_Q': []
    }

    for i in range(min(8, betas.shape[1])):
        beta_i = betas[:, i]

        r_p, _ = pearsonr(beta_i, p_true)
        r_q, _ = pearsonr(beta_i, q_true)
        r_P, _ = pearsonr(beta_i, P_true)
        r_Q, _ = pearsonr(beta_i, Q_true)
        r_cos_Q, _ = pearsonr(beta_i, np.cos(Q_true))
        r_sin_Q, _ = pearsonr(beta_i, np.sin(Q_true))

        correlations['p'].append(r_p)
        correlations['q'].append(r_q)
        correlations['P'].append(r_P)
        correlations['cos_Q'].append(r_cos_Q)
        correlations['sin_Q'].append(r_sin_Q)

        print(f"β_{i:d}    | {r_p:8.4f} | {r_q:8.4f} | {r_P:8.4f} | {r_Q:8.4f} | {r_cos_Q:8.4f} | {r_sin_Q:8.4f}")

    # Find best correlations
    print("\n" + "-" * 70)
    print("BEST CORRELATIONS")
    print("-" * 70)

    best_p_idx = np.argmax(np.abs(correlations['p']))
    best_q_idx = np.argmax(np.abs(correlations['q']))
    best_P_idx = np.argmax(np.abs(correlations['P']))

    print(f"\nBest β for p: β_{best_p_idx} (r = {correlations['p'][best_p_idx]:.4f})")
    print(f"Best β for q: β_{best_q_idx} (r = {correlations['q'][best_q_idx]:.4f})")
    print(f"Best β for P: β_{best_P_idx} (r = {correlations['P'][best_P_idx]:.4f})")

    # Check if β[:, 0:2] can proxy for (p, q)
    max_pq_corr = max(
        max(np.abs(correlations['p'][:2])),
        max(np.abs(correlations['q'][:2]))
    )

    print("\n" + "-" * 70)
    print("VERDICT")
    print("-" * 70)

    if max_pq_corr > 0.9:
        print("\n✓ HIGH CORRELATION: β[:, 0:2] can directly proxy for (p, q)")
        print("  → Simple connection to ImprovedHJB_MLP possible")
        can_direct = True
    elif max_pq_corr > 0.7:
        print("\n◐ MODERATE CORRELATION: β partially captures (p, q)")
        print("  → May work with some adaptation")
        can_direct = True
    else:
        print("\n✗ LOW CORRELATION: β does NOT directly correspond to (p, q)")
        print("  → Need learned (β → p,q) mapping or different approach")
        can_direct = False

    return {
        'betas': betas,
        'ground_truth': ground_truth,
        'correlations': correlations,
        'rom': rom,
        'can_direct': can_direct
    }


def test_optimal_rotation():
    """
    Test if there's a linear transformation from β to (p, q).

    Even if β[:, 0:2] != (p, q) directly, there might be a rotation
    R such that R @ β[:, 0:2] ≈ (p, q).
    """
    print("\n" + "=" * 70)
    print("TEST: OPTIMAL LINEAR TRANSFORMATION β → (p, q)")
    print("=" * 70)

    omega = 1.0
    n_trajectories = 200
    window_size = 128

    # Generate data
    np.random.seed(123)
    trajectories = []
    pq_true = []

    for i in range(n_trajectories):
        E = np.random.uniform(0.5, 4.5)
        theta0 = np.random.uniform(0, 2 * np.pi)

        p0 = np.sqrt(2 * E) * np.sin(theta0)
        q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

        z, p_t, q_t = generate_sho_signal(p0, q0, omega=omega, n_points=window_size)
        trajectories.append(z)

        center_idx = window_size // 2
        pq_true.append([p_t[center_idx], q_t[center_idx]])

    pq_true = np.array(pq_true)

    # Fit ROM
    rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=window_size)
    betas = rom.fit(trajectories, extract_windows=False)

    # Find optimal rotation/scaling from β[:, 0:2] to (p, q)
    beta_2d = betas[:, :2]

    # Standardize both
    beta_2d_std = (beta_2d - beta_2d.mean(axis=0)) / beta_2d.std(axis=0)
    pq_std = (pq_true - pq_true.mean(axis=0)) / pq_true.std(axis=0)

    # Procrustes: find R such that β_std @ R ≈ pq_std
    R, scale = orthogonal_procrustes(beta_2d_std, pq_std)

    # Apply transformation
    beta_transformed = beta_2d_std @ R

    # Measure fit quality
    residual = np.linalg.norm(beta_transformed - pq_std) / np.linalg.norm(pq_std)

    # Correlations after transformation
    r_p_after, _ = pearsonr(beta_transformed[:, 0], pq_std[:, 0])
    r_q_after, _ = pearsonr(beta_transformed[:, 1], pq_std[:, 1])

    print(f"\nOptimal rotation matrix R:")
    print(f"  [{R[0, 0]:7.4f}  {R[0, 1]:7.4f}]")
    print(f"  [{R[1, 0]:7.4f}  {R[1, 1]:7.4f}]")
    print(f"\nResidual after transformation: {residual:.4f}")
    print(f"Correlation after transformation:")
    print(f"  r(R@β[:, 0], p) = {r_p_after:.4f}")
    print(f"  r(R@β[:, 1], q) = {r_q_after:.4f}")

    # Try with more components
    print("\n" + "-" * 70)
    print("TRYING WITH MORE β COMPONENTS")
    print("-" * 70)

    for n_comp in [2, 4, 6, 8]:
        beta_n = betas[:, :n_comp]

        # Linear regression: β @ W ≈ pq
        # W = (β^T β)^-1 β^T pq
        W, residuals, rank, s = np.linalg.lstsq(beta_n, pq_true, rcond=None)

        pq_pred = beta_n @ W

        # Errors
        p_error = np.mean(np.abs(pq_pred[:, 0] - pq_true[:, 0]))
        q_error = np.mean(np.abs(pq_pred[:, 1] - pq_true[:, 1]))

        r_p, _ = pearsonr(pq_pred[:, 0], pq_true[:, 0])
        r_q, _ = pearsonr(pq_pred[:, 1], pq_true[:, 1])

        print(f"\nn_components = {n_comp}:")
        print(f"  Linear fit: β @ W → (p, q)")
        print(f"  p: error={p_error:.4f}, r={r_p:.4f}")
        print(f"  q: error={q_error:.4f}, r={r_q:.4f}")

    return {
        'R': R,
        'residual': residual,
        'r_p_after': r_p_after,
        'r_q_after': r_q_after
    }


def test_direct_beta_action_angle():
    """
    Test: Can we learn (P, Q) directly from β without going through (p, q)?

    This tests if HST_ROM output β can be transformed directly to action-angle
    coordinates, bypassing the (p, q) intermediate step.
    """
    print("\n" + "=" * 70)
    print("TEST: DIRECT β → (P, Q) MAPPING")
    print("=" * 70)

    omega = 1.0
    n_trajectories = 200
    window_size = 128

    # Generate data with ground truth (P, Q)
    np.random.seed(456)
    trajectories = []
    PQ_true = []

    for i in range(n_trajectories):
        E = np.random.uniform(0.5, 4.5)
        theta0 = np.random.uniform(0, 2 * np.pi)

        p0 = np.sqrt(2 * E) * np.sin(theta0)
        q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

        z, p_t, q_t = generate_sho_signal(p0, q0, omega=omega, n_points=window_size)
        trajectories.append(z)

        P_true = E / omega
        Q_true = wrap_to_2pi(theta0 + omega * (window_size // 2) * 0.01)
        PQ_true.append([P_true, Q_true])

    PQ_true = np.array(PQ_true)
    P_true = PQ_true[:, 0]
    Q_true = PQ_true[:, 1]

    # Fit ROM
    rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=window_size)
    betas = rom.fit(trajectories, extract_windows=False)

    # Test: Can β predict P?
    print("\n1. LINEAR REGRESSION: β → P")
    print("-" * 40)

    W_P, _, _, _ = np.linalg.lstsq(betas, P_true, rcond=None)
    P_pred = betas @ W_P

    r_P, _ = pearsonr(P_pred, P_true)
    P_error = np.mean(np.abs(P_pred - P_true) / P_true)

    print(f"  Correlation: {r_P:.4f}")
    print(f"  Relative error: {P_error:.4f}")
    print(f"  {'✓ GOOD' if r_P > 0.95 else '✗ POOR'}")

    # Test: Can β predict Q? (Need circular handling)
    print("\n2. LINEAR REGRESSION: β → (cos(Q), sin(Q))")
    print("-" * 40)

    cos_Q_true = np.cos(Q_true)
    sin_Q_true = np.sin(Q_true)

    W_cos, _, _, _ = np.linalg.lstsq(betas, cos_Q_true, rcond=None)
    W_sin, _, _, _ = np.linalg.lstsq(betas, sin_Q_true, rcond=None)

    cos_Q_pred = betas @ W_cos
    sin_Q_pred = betas @ W_sin

    Q_pred = np.arctan2(sin_Q_pred, cos_Q_pred)
    Q_pred = wrap_to_2pi(Q_pred)

    # Angular error
    Q_diff = Q_pred - Q_true
    Q_error = np.mean(np.abs(np.arctan2(np.sin(Q_diff), np.cos(Q_diff))))

    r_cos, _ = pearsonr(cos_Q_pred, cos_Q_true)
    r_sin, _ = pearsonr(sin_Q_pred, sin_Q_true)

    print(f"  cos(Q) correlation: {r_cos:.4f}")
    print(f"  sin(Q) correlation: {r_sin:.4f}")
    print(f"  Angular error: {np.degrees(Q_error):.1f}°")
    print(f"  {'✓ GOOD' if Q_error < 0.3 else '✗ POOR'}")

    # Conclusion
    print("\n" + "-" * 70)
    print("CONCLUSION: DIRECT β → (P, Q)")
    print("-" * 70)

    direct_works = r_P > 0.95 and Q_error < 0.3

    if direct_works:
        print("\n✓ DIRECT MAPPING WORKS")
        print("  β can be linearly transformed to (P, Q)")
        print("  → Could train ImprovedHJB_MLP variant on β directly")
    else:
        print("\n◐ PARTIAL SUCCESS")
        print(f"  P prediction: {'✓' if r_P > 0.95 else '✗'}")
        print(f"  Q prediction: {'✓' if Q_error < 0.3 else '✗'}")
        print("  → May need nonlinear mapping (MLP)")

    return {
        'P_corr': r_P,
        'Q_error_deg': np.degrees(Q_error),
        'direct_works': direct_works
    }


def run_full_bridge_test():
    """Run all bridge tests."""

    print("\n" + "=" * 70)
    print("HST_ROM → ImprovedHJB_MLP BRIDGE ANALYSIS")
    print("=" * 70)

    # Test 1: Direct correlation
    result1 = test_beta_pq_correlation()

    # Test 2: Optimal rotation
    result2 = test_optimal_rotation()

    # Test 3: Direct β → (P, Q)
    result3 = test_direct_beta_action_angle()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BRIDGE OPTIONS")
    print("=" * 70)

    print("\n| Approach | Feasibility | Notes |")
    print("|----------|-------------|-------|")
    print(f"| β[:, 0:2] ≈ (p, q) | {'✓' if result1['can_direct'] else '✗'} | Direct correlation |")
    print(f"| R @ β[:, 0:2] → (p, q) | {'✓' if result2['residual'] < 0.3 else '✗'} | With rotation |")
    print(f"| β → (P, Q) directly | {'✓' if result3['direct_works'] else '◐'} | Skip (p, q) |")

    print("\n" + "-" * 70)
    print("RECOMMENDED APPROACH:")
    print("-" * 70)

    if result3['direct_works']:
        print("\n→ DIRECT: Train on β → (P, Q)")
        print("  HST_ROM already extracts action-angle-like coordinates")
        print("  No need for intermediate (p, q) step")
    elif result2['residual'] < 0.3:
        print("\n→ WITH ROTATION: Use R @ β[:, :2] as (p, q) proxy")
        print("  Then feed to ImprovedHJB_MLP")
    else:
        print("\n→ LEARNED MAPPING: Train MLP to map β → (p, q)")
        print("  Then chain with ImprovedHJB_MLP")

    return {
        'correlation_test': result1,
        'rotation_test': result2,
        'direct_test': result3
    }


if __name__ == "__main__":
    results = run_full_bridge_test()
