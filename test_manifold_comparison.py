"""
Compare manifold diagnostics between SHO and Pendulum.

This test runs the ChatGPT-suggested manifold diagnostics on both systems
to understand their structural differences, with focus on the TWIST metric
that measures manifold curvature.

Expected findings:
- SHO: Clean 2D manifold with LOW twist (nearly flat)
- Pendulum: 2D manifold but HIGH twist (curved) → explains linear map failure

Key insight: If twist_ratio (pendulum/SHO) > 2, the linear β→(p,q) map
fails due to manifold curvature, and an MLP encoder is needed.
"""

import numpy as np
import sys

from hst_rom import HST_ROM
from manifold_diagnostics import (
    run_manifold_diagnostics, compare_manifolds,
    run_full_diagnostics, compare_and_recommend
)


def generate_sho_signal(p0, q0, omega=1.0, dt=0.01, n_points=128):
    """Generate SHO signal z(t) = q(t) + i*p(t)/omega."""
    t = np.arange(n_points) * dt
    E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
    theta0 = np.arctan2(p0, omega * q0)
    theta_t = theta0 + omega * t
    p_t = np.sqrt(2 * E) * np.sin(theta_t)
    q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)
    z = q_t + 1j * p_t / omega
    return z, p_t, q_t


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


def test_manifold_comparison():
    """Compare manifold structure of SHO vs Pendulum with TWIST diagnostic."""
    print("=" * 70)
    print("MANIFOLD STRUCTURE COMPARISON: SHO vs PENDULUM")
    print("(with TWIST diagnostic for curvature detection)")
    print("=" * 70)

    np.random.seed(42)

    # Parameters
    omega0 = 1.0
    n_samples = 300
    window_size = 128
    n_pca = 8
    dt = 0.01

    # ============================
    # Generate SHO data
    # ============================
    print("\n[1] Generating SHO data...")
    E_samples = np.random.uniform(0.5, 2.0, n_samples)
    phase_samples = np.random.uniform(0, 2 * np.pi, n_samples)

    p0_sho = np.sqrt(2 * E_samples) * np.sin(phase_samples)
    q0_sho = np.sqrt(2 * E_samples) / omega0 * np.cos(phase_samples)

    sho_signals = []
    for i in range(n_samples):
        z, _, _ = generate_sho_signal(p0_sho[i], q0_sho[i], omega=omega0,
                                       dt=dt, n_points=window_size)
        sho_signals.append(z)

    # Fit HST_ROM to SHO
    hst_sho = HST_ROM(n_components=n_pca, J=3, window_size=window_size)
    beta_sho = hst_sho.fit(sho_signals, extract_windows=False)

    print(f"  SHO β shape: {beta_sho.shape}")
    print(f"  SHO PCA variance: {sum(hst_sho.pca.explained_variance_ratio_):.3f}")

    # Run FULL diagnostics on SHO (includes twist)
    sho_results = run_full_diagnostics(beta_sho, name="SHO")

    # ============================
    # Generate Pendulum data
    # ============================
    print("\n[2] Generating Pendulum rotation data...")
    pend_signals = []
    for i in range(n_samples):
        p0, q0, E = generate_rotation_ic(E_min=1.5, E_max=3.0, omega0=omega0)
        z, _, _ = generate_pendulum_signal(p0, q0, omega0=omega0,
                                            dt=dt, n_points=window_size)
        pend_signals.append(z)

    # Fit HST_ROM to Pendulum
    hst_pend = HST_ROM(n_components=n_pca, J=3, window_size=window_size)
    beta_pend = hst_pend.fit(pend_signals, extract_windows=False)

    print(f"  Pendulum β shape: {beta_pend.shape}")
    print(f"  Pendulum PCA variance: {sum(hst_pend.pca.explained_variance_ratio_):.3f}")

    # Run FULL diagnostics on Pendulum (includes twist)
    pend_results = run_full_diagnostics(beta_pend, name="Pendulum Rotation")

    # ============================
    # Compare and recommend
    # ============================
    recommendation = compare_and_recommend(sho_results, pend_results)

    # ============================
    # Summary
    # ============================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    twist_ratio = recommendation['twist_ratio']
    print(f"\nTwist ratio (Pendulum/SHO): {twist_ratio:.2f}x")

    if twist_ratio > 2:
        print(f"  → CURVATURE HYPOTHESIS CONFIRMED!")
        print(f"  → Pendulum manifold is {twist_ratio:.1f}x more curved than SHO")
        print(f"  → This explains the ~0.3 forward error with linear β→(p,q) map")
        print(f"  → NEXT STEP: Replace linear W with small MLP encoder")
        curvature_confirmed = True
    else:
        print(f"  → Twist ratio < 2, curvature may not be the main issue")
        curvature_confirmed = False

    print("\n" + "=" * 70)
    print(f"Recommendation: {recommendation['recommendation']}")
    print("=" * 70)

    return {
        'sho': sho_results,
        'pendulum': pend_results,
        'recommendation': recommendation,
        'curvature_confirmed': curvature_confirmed,
    }


if __name__ == "__main__":
    results = test_manifold_comparison()

    # Success criteria:
    # 1. Twist ratio > 2 (confirms curvature hypothesis)
    # 2. Recommendation is 'mlp_encoder' (Case B)
    curvature_confirmed = results['curvature_confirmed']
    correct_recommendation = results['recommendation']['recommendation'] == 'mlp_encoder'

    success = curvature_confirmed and correct_recommendation

    if success:
        print("\nTEST PASSED: Curvature hypothesis confirmed, MLP encoder recommended")
    else:
        print("\nTEST RESULT: Curvature hypothesis not confirmed")
        print("  This is informational - may need to investigate other factors")

    # Exit 0 regardless - this is a diagnostic test
    sys.exit(0)
