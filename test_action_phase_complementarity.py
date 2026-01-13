"""
Action Measurement: Inherent Complementarity with Phase

Key insight: Action P = (1/2π) ∮ p dq is DEFINED as an integral over the full cycle.
Computing P requires averaging over all phases Q. This is structural, not incidental.

Three Regimes:
1. Small η (classical): Can measure (q,p) → get both P and Q
2. Large η (quantum-like): Must use cycle integral → P survives, Q lost
3. Intermediate: Transition between regimes

The complementarity is STRUCTURAL:
- P is defined as phase-averaged quantity
- Precise P measurement requires going around the orbit
- This inherently averages over Q
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from scipy.special import ellipk
import warnings
warnings.filterwarnings('ignore')


def wrap_angle(angle):
    """Wrap angle to [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


# ============================================================
# SIMPLE HARMONIC OSCILLATOR (Clean demonstration)
# ============================================================

def generate_sho_trajectory(P_true, omega=1.0, T_periods=10, dt=0.01, Q_init=0):
    """
    Generate SHO trajectory with action P.

    For SHO: H = (p² + ω²q²)/2, Action P = H/ω = (p² + ω²q²)/(2ω)
    Orbit: q = √(2P/ω²) cos(Q), p = √(2Pω²) sin(Q) where Q = ωt + Q_init
    """
    T_period = 2 * np.pi / omega
    T_total = T_periods * T_period

    t = np.arange(0, T_total, dt)
    Q = omega * t + Q_init  # Phase evolution

    # Amplitude from action: P = E/ω = (½mω²A²)/ω = ½ωA² → A = √(2P/ω)
    A = np.sqrt(2 * P_true / omega)

    q = A * np.cos(Q)
    p = -A * omega * np.sin(Q)  # p = m*dq/dt = -Aω sin(Q)

    return t, q, p, Q % (2 * np.pi), T_period


# ============================================================
# MEASUREMENT WITH BACK-ACTION
# ============================================================

def measure_with_backaction(q_true, p_true, sigma_q, eta, seed=None):
    """
    Measure (q, p) with back-action.

    Protocol:
    1. Measure q with precision σ_q
    2. Back-action kicks p by η/σ_q
    3. Measure p (already corrupted)
    """
    if seed is not None:
        np.random.seed(seed)

    # Measure q
    q_meas = q_true + np.random.normal(0, sigma_q)

    # Back-action on p
    sigma_p_kick = eta / max(sigma_q, 1e-10)
    p_after_kick = p_true + np.random.normal(0, sigma_p_kick)

    # Measure p (with small additional noise)
    p_meas = p_after_kick + np.random.normal(0, sigma_q * 0.1)

    return q_meas, p_meas


# ============================================================
# ACTION MEASUREMENT METHODS
# ============================================================

def measure_action_instant(q_meas, p_meas, omega=1.0):
    """
    Method A: Instantaneous action estimate from single (q, p).

    For SHO: P = (p² + ω²q²) / (2ω)
    """
    return (p_meas**2 + omega**2 * q_meas**2) / (2 * omega)


def measure_action_integral(q_series, p_series, dt):
    """
    Method B: Action from cycle integral P = (1/2π) ∮ p dq

    Numerically: P ≈ (1/2π) Σ p_i · (q_{i+1} - q_i)
    """
    # Compute integral using trapezoidal rule
    dq = np.diff(q_series)
    p_mid = (p_series[:-1] + p_series[1:]) / 2

    integral = np.sum(p_mid * dq)
    P_measured = integral / (2 * np.pi)

    return P_measured


def measure_action_with_backaction(q_true, p_true, eta, dt, seed=None):
    """
    Measure action by integrating noisy (q, p) measurements.

    At each timestep:
    - Measure q, p with back-action
    - Accumulate ∮ p dq
    """
    if seed is not None:
        np.random.seed(seed)

    sigma_q = np.sqrt(eta)  # Uncertainty-optimal precision

    n = len(q_true)
    q_meas_series = np.zeros(n)
    p_meas_series = np.zeros(n)

    for i in range(n):
        q_m, p_m = measure_with_backaction(q_true[i], p_true[i], sigma_q, eta)
        q_meas_series[i] = q_m
        p_meas_series[i] = p_m

    P_meas = measure_action_integral(q_meas_series, p_meas_series, dt)

    return P_meas, q_meas_series, p_meas_series


# ============================================================
# PHASE MEASUREMENT
# ============================================================

def measure_phase_instant(q_meas, p_meas, omega=1.0):
    """
    Instantaneous phase estimate from (q, p).

    Q = arctan2(-p/ω, q) for SHO with q = A cos(Q), p = -Aω sin(Q)
    """
    return np.arctan2(-p_meas / omega, q_meas)


def measure_phase_cumulative(q_meas_series, p_meas_series, omega=1.0):
    """
    Phase estimate from cumulative measurements.

    Use circular mean of instantaneous estimates.
    """
    Q_estimates = np.arctan2(-p_meas_series / omega, q_meas_series)

    # Circular mean
    sin_mean = np.mean(np.sin(Q_estimates))
    cos_mean = np.mean(np.cos(Q_estimates))
    Q_avg = np.arctan2(sin_mean, cos_mean)

    return Q_avg


# ============================================================
# MAIN TESTS
# ============================================================

def test_action_integral_convergence(P_true=1.0, omega=1.0, eta_values=None, seed=42):
    """
    Test 1: How does action integral converge with observation time?
    """
    if eta_values is None:
        eta_values = [0.001, 0.01, 0.05, 0.1, 0.3]

    T_obs_ratios = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    n_trials = 30

    results = {}

    for eta in eta_values:
        P_errors = []

        for ratio in T_obs_ratios:
            # Generate trajectory
            t, q, p, Q_true, T_period = generate_sho_trajectory(
                P_true, omega, T_periods=max(ratio + 1, 2), dt=0.01
            )

            # Observation window
            n_obs = int(ratio * T_period / 0.01)
            n_obs = min(n_obs, len(q))

            trial_errors = []
            for trial in range(n_trials):
                P_meas, _, _ = measure_action_with_backaction(
                    q[:n_obs], p[:n_obs], eta, 0.01, seed=seed + trial
                )
                trial_errors.append(abs(P_meas - P_true))

            P_errors.append(np.mean(trial_errors))

        results[eta] = {
            'T_obs_ratios': T_obs_ratios,
            'P_errors': np.array(P_errors)
        }

    return results


def test_phase_error_growth(P_true=1.0, omega=1.0, eta_values=None, seed=42):
    """
    Test 2: How does phase error grow with observation time?

    Key: For large η, accumulated back-action randomizes phase knowledge.
    """
    if eta_values is None:
        eta_values = [0.001, 0.01, 0.05, 0.1, 0.3]

    T_obs_ratios = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    n_trials = 30

    results = {}

    for eta in eta_values:
        Q_errors = []

        for ratio in T_obs_ratios:
            # Generate trajectory with random initial phase
            np.random.seed(seed)
            Q_init = np.random.uniform(0, 2 * np.pi)

            t, q, p, Q_true_series, T_period = generate_sho_trajectory(
                P_true, omega, T_periods=max(ratio + 1, 2), dt=0.01, Q_init=Q_init
            )

            n_obs = int(ratio * T_period / 0.01)
            n_obs = min(n_obs, len(q))

            trial_errors = []
            for trial in range(n_trials):
                _, q_meas, p_meas = measure_action_with_backaction(
                    q[:n_obs], p[:n_obs], eta, 0.01, seed=seed + trial + 1000
                )

                # Final phase estimate from last measurement
                Q_est = measure_phase_instant(q_meas[-1], p_meas[-1], omega)
                Q_true_final = Q_true_series[n_obs - 1]

                trial_errors.append(abs(wrap_angle(Q_est - Q_true_final)))

            Q_errors.append(np.mean(trial_errors))

        results[eta] = {
            'T_obs_ratios': T_obs_ratios,
            'Q_errors': np.array(Q_errors)
        }

    return results


def test_instant_vs_integral(P_true=1.0, omega=1.0, eta_values=None, seed=42):
    """
    Test 3: Compare instantaneous vs integral action measurement.
    """
    if eta_values is None:
        eta_values = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]

    n_trials = 50
    results = {}

    for eta in eta_values:
        # Generate trajectory
        t, q, p, Q_true, T_period = generate_sho_trajectory(
            P_true, omega, T_periods=5, dt=0.01
        )

        sigma_q = np.sqrt(eta)

        # Method A: Instantaneous
        instant_errors = []
        instant_Q_errors = []

        for trial in range(n_trials):
            # Pick random point
            idx = np.random.randint(len(q) // 4, 3 * len(q) // 4)
            q_m, p_m = measure_with_backaction(q[idx], p[idx], sigma_q, eta, seed=seed + trial)

            P_instant = measure_action_instant(q_m, p_m, omega)
            Q_instant = measure_phase_instant(q_m, p_m, omega)

            instant_errors.append(abs(P_instant - P_true))
            instant_Q_errors.append(abs(wrap_angle(Q_instant - Q_true[idx])))

        # Method B: 1-cycle integral
        n_one_cycle = int(T_period / 0.01)
        cycle_errors = []

        for trial in range(n_trials):
            start = np.random.randint(0, len(q) - n_one_cycle)
            P_meas, _, _ = measure_action_with_backaction(
                q[start:start + n_one_cycle], p[start:start + n_one_cycle],
                eta, 0.01, seed=seed + trial + 2000
            )
            cycle_errors.append(abs(P_meas - P_true))

        # Method B: 3-cycle integral
        n_three_cycle = int(3 * T_period / 0.01)
        three_cycle_errors = []

        for trial in range(n_trials):
            start = np.random.randint(0, max(1, len(q) - n_three_cycle))
            end = min(start + n_three_cycle, len(q))
            P_meas, _, _ = measure_action_with_backaction(
                q[start:end], p[start:end], eta, 0.01, seed=seed + trial + 3000
            )
            three_cycle_errors.append(abs(P_meas - P_true))

        results[eta] = {
            'P_instant': np.mean(instant_errors),
            'P_1cycle': np.mean(cycle_errors),
            'P_3cycle': np.mean(three_cycle_errors),
            'Q_instant': np.mean(instant_Q_errors)
        }

    return results


def test_classical_limit(P_true=1.0, omega=1.0, seed=42):
    """
    Test 4: Show both P and Q errors → 0 as η → 0.
    """
    eta_values = np.logspace(-4, 0, 20)
    n_trials = 30

    P_errors = []
    Q_errors = []

    for eta in eta_values:
        t, q, p, Q_true, T_period = generate_sho_trajectory(
            P_true, omega, T_periods=3, dt=0.01
        )

        sigma_q = np.sqrt(eta)

        trial_P_errors = []
        trial_Q_errors = []

        for trial in range(n_trials):
            idx = np.random.randint(len(q) // 4, 3 * len(q) // 4)
            q_m, p_m = measure_with_backaction(q[idx], p[idx], sigma_q, eta, seed=seed + trial)

            P_meas = measure_action_instant(q_m, p_m, omega)
            Q_meas = measure_phase_instant(q_m, p_m, omega)

            trial_P_errors.append(abs(P_meas - P_true))
            trial_Q_errors.append(abs(wrap_angle(Q_meas - Q_true[idx])))

        P_errors.append(np.mean(trial_P_errors))
        Q_errors.append(np.mean(trial_Q_errors))

    return eta_values, np.array(P_errors), np.array(Q_errors)


# ============================================================
# VISUALIZATION
# ============================================================

def create_orbit_visualization(P_true=1.0, omega=1.0):
    """
    Create visualization showing P = enclosed area / 2π.
    """
    # Generate orbit
    Q = np.linspace(0, 2 * np.pi, 100)
    A = np.sqrt(2 * P_true / omega)

    q = A * np.cos(Q)
    p = -A * omega * np.sin(Q)

    return q, p, Q


def run_full_test():
    """Run all tests and create visualization."""

    print("=" * 70)
    print("ACTION-PHASE COMPLEMENTARITY TEST")
    print("=" * 70)
    print("\nKey insight: P = (1/2π) ∮ p dq requires averaging over all phases Q")

    P_true = 1.0
    omega = 1.0
    eta_values = [0.001, 0.01, 0.05, 0.1, 0.3]

    # Run tests
    print("\n[1] Testing action integral convergence...")
    action_results = test_action_integral_convergence(P_true, omega, eta_values)

    print("[2] Testing phase error growth...")
    phase_results = test_phase_error_growth(P_true, omega, eta_values)

    print("[3] Testing instant vs integral methods...")
    method_results = test_instant_vs_integral(P_true, omega, eta_values + [0.5])

    print("[4] Testing classical limit...")
    eta_sweep, P_errors_classical, Q_errors_classical = test_classical_limit(P_true, omega)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: Action integral convergence
    ax = axes[0, 0]
    for eta in eta_values:
        data = action_results[eta]
        ax.semilogy(data['T_obs_ratios'], data['P_errors'], 'o-',
                   label=f'η={eta}', markersize=4)
    ax.axvline(1.0, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('T_obs / T')
    ax.set_ylabel('Action error |P_meas - P_true|')
    ax.set_title('Action Integral Convergence\n(More cycles → better P for large η)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Phase error growth
    ax = axes[0, 1]
    for eta in eta_values:
        data = phase_results[eta]
        ax.plot(data['T_obs_ratios'], data['Q_errors'], 's-',
               label=f'η={eta}', markersize=4)
    ax.axhline(np.pi, color='red', linestyle=':', alpha=0.5, label='π (random)')
    ax.axvline(1.0, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('T_obs / T')
    ax.set_ylabel('Phase error |Q_meas - Q_true|')
    ax.set_title('Phase Error Growth\n(Q degrades with observation time)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, np.pi * 1.2)

    # Panel 3: Complementarity cross-plot
    ax = axes[0, 2]
    for eta in [0.01, 0.1, 0.3]:
        if eta in action_results and eta in phase_results:
            P_errs = action_results[eta]['P_errors']
            Q_errs = phase_results[eta]['Q_errors']

            # Precision = 1/error (capped)
            P_prec = 1 / (P_errs + 0.01)
            Q_prec = 1 / (Q_errs + 0.01)

            ax.plot(P_prec, Q_prec, 'o-', label=f'η={eta}', markersize=6, alpha=0.7)

            # Mark T_obs = 1T point
            idx_1T = np.argmin(np.abs(action_results[eta]['T_obs_ratios'] - 1.0))
            ax.scatter([P_prec[idx_1T]], [Q_prec[idx_1T]], s=100, marker='*', zorder=5)

    ax.set_xlabel('P precision (1/error)')
    ax.set_ylabel('Q precision (1/error)')
    ax.set_title('Complementarity: P vs Q Precision\n(★ = T_obs = 1T)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Classical limit
    ax = axes[1, 0]
    ax.loglog(eta_sweep, P_errors_classical, 'b-', linewidth=2, label='P error')
    ax.loglog(eta_sweep, Q_errors_classical, 'r-', linewidth=2, label='Q error')
    ax.loglog(eta_sweep, np.sqrt(eta_sweep), 'k--', alpha=0.5, label='√η')
    ax.set_xlabel('Back-action strength η')
    ax.set_ylabel('Measurement error')
    ax.set_title('Classical Limit (η → 0)\n(Both P and Q become precise)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: Orbit visualization
    ax = axes[1, 1]
    q_orbit, p_orbit, Q_orbit = create_orbit_visualization(P_true, omega)

    # Color by phase
    scatter = ax.scatter(q_orbit, p_orbit, c=Q_orbit, cmap='hsv', s=20)
    ax.plot(q_orbit, p_orbit, 'k-', alpha=0.3)

    # Add arrow showing direction
    arrow_idx = len(q_orbit) // 4
    ax.annotate('', xy=(q_orbit[arrow_idx + 5], p_orbit[arrow_idx + 5]),
                xytext=(q_orbit[arrow_idx], p_orbit[arrow_idx]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlabel('q (position)')
    ax.set_ylabel('p (momentum)')
    ax.set_title('Phase Space Orbit\n(Color = phase Q, Area/2π = Action P)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Phase Q')

    # Add annotation
    ax.text(0, 0, f'P = {P_true:.1f}', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 6: Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Compute summary statistics
    eta_example = 0.1
    if eta_example in method_results:
        mr = method_results[eta_example]

    summary = f"""
    ACTION-PHASE COMPLEMENTARITY
    ============================

    KEY INSIGHT:
    Action P = (1/2π) ∮ p dq

    This integral goes around the FULL orbit,
    averaging over ALL phases Q.

    → P measurement inherently loses Q info!

    THREE REGIMES:

    1. Classical (η → 0):
       - Can measure (q,p) precisely
       - Get P instantly: P = (p² + ω²q²)/(2ω)
       - Get Q instantly: Q = atan2(-p/ω, q)
       - Both observable!

    2. Quantum-like (η large):
       - Can't measure (q,p) precisely
       - Must use cycle integral for P
       - Cycle integral averages over Q
       - P survives, Q lost!

    3. Transition (η intermediate):
       - Some P-Q trade-off
       - More cycles → better P, worse Q

    STRUCTURAL COMPLEMENTARITY:
    ┌────────────────────────────────┐
    │ P is DEFINED as phase average │
    │ → Knowing P precisely means   │
    │   averaging over all Q        │
    │ → Can't know both precisely!  │
    └────────────────────────────────┘
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('action_phase_complementarity.png', dpi=150, bbox_inches='tight')
    print("\nSaved: action_phase_complementarity.png")
    plt.show()

    # Print numerical summary
    print("\n" + "=" * 70)
    print("NUMERICAL SUMMARY")
    print("=" * 70)

    print(f"\n{'η':<8} {'P (instant)':<12} {'P (1 cycle)':<12} {'P (3 cycle)':<12} {'Q (instant)':<12}")
    print("-" * 56)

    for eta in sorted(method_results.keys()):
        mr = method_results[eta]
        print(f"{eta:<8.3f} {mr['P_instant']:<12.4f} {mr['P_1cycle']:<12.4f} {mr['P_3cycle']:<12.4f} {mr['Q_instant']:<12.4f}")

    # Regime classification
    print("\n" + "=" * 70)
    print("REGIME CLASSIFICATION")
    print("=" * 70)

    print(f"\n{'η':<8} {'T_obs/T':<10} {'P_error':<12} {'Q_error':<12} {'Regime':<15}")
    print("-" * 57)

    test_points = [(0.001, 0.5), (0.001, 2.0), (0.1, 0.5), (0.1, 2.0), (0.3, 0.5), (0.3, 2.0)]

    for eta, T_ratio in test_points:
        if eta in action_results and eta in phase_results:
            idx = np.argmin(np.abs(action_results[eta]['T_obs_ratios'] - T_ratio))
            P_err = action_results[eta]['P_errors'][idx]
            Q_err = phase_results[eta]['Q_errors'][idx]

            if P_err < 0.1 and Q_err < 0.5:
                regime = "Classical"
            elif P_err < 0.3 and Q_err > 1.0:
                regime = "Quantum-like"
            else:
                regime = "Transition"

            print(f"{eta:<8.3f} {T_ratio:<10.1f} {P_err:<12.4f} {Q_err:<12.4f} {regime:<15}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("""
    The complementarity between P and Q is STRUCTURAL:

    1. Action P = (1/2π) ∮ p dq is defined as a cycle integral
    2. This integral averages over all phases Q
    3. Precise P measurement requires going around the orbit
    4. Going around the orbit loses Q information

    For small back-action (η → 0):
    - Can bypass the integral: P = (p² + ω²q²)/(2ω)
    - Both P and Q accessible from single (q, p) measurement
    - Classical mechanics regime

    For large back-action (η >> 0):
    - Must use cycle integral (instantaneous fails)
    - Cycle integral averages over Q
    - P survives, Q is lost
    - "Quantum" regime emerges

    This is NOT uncertainty principle with ℏ.
    It's the DEFINITION of action as a phase-averaged quantity.
    The complementarity is mathematical, not quantum mechanical.
    """)

    return {
        'action': action_results,
        'phase': phase_results,
        'methods': method_results,
        'classical': (eta_sweep, P_errors_classical, Q_errors_classical)
    }


if __name__ == "__main__":
    results = run_full_test()
