"""
Prediction with Measurement Error - Operational Test of Glinsky's Claim

Key insight: Measurement error + time evolution → phase unpredictability

The mechanism:
1. Small error δP in action measurement → error in frequency estimate
2. Over time Δτ, frequency error accumulates: δQ ~ (dω/dP)·δP·Δτ
3. When δQ ~ π, phase is effectively random
4. But action error stays bounded at ~ε (doesn't grow)

This demonstrates Glinsky's quantization operationally:
- P remains predictable at all timescales
- Q becomes unpredictable when Δτ > Δτ_critical
- The critical timescale depends on measurement precision ε

Expected:
| Δτ/T | δP | δQ | Regime |
|------|----|----|--------|
| 0.1 | ~ε | ~ε | Classical (both predictable) |
| 1.0 | ~ε | grows | Transition |
| 10+ | ~ε | ~π (random) | "Quantum" (only P predictable) |
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian


def pendulum_omega(E):
    """Frequency ω(E) for pendulum: H = p²/2 - cos(q), so E = -cos(q_max)"""
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 <= 0 or k2 >= 1:
        return 1.0
    k = np.sqrt(k2)
    return np.pi / (2 * ellipk(k**2))


def pendulum_omega_derivative(E, dE=0.001):
    """Numerical derivative dω/dE"""
    if E >= 0.99 or E <= -0.99:
        return np.nan
    return (pendulum_omega(E + dE) - pendulum_omega(E - dE)) / (2 * dE)


def energy_to_action(E):
    """Action J(E) for pendulum"""
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 <= 0 or k2 >= 1:
        return max(E + 1, 0.01)
    k = np.sqrt(k2)
    K = ellipk(k**2)
    E_ellip = ellipe(k**2)
    # J = (4/π)[E·K - (1-k²)·K] for pendulum
    J = (8/np.pi) * (E_ellip - (1 - k**2) * K)
    return J


def wrap_angle(angle):
    """Wrap angle to [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def prediction_error_single(E_true, Q_true, delta_tau, epsilon, seed=None):
    """
    Compute prediction error for a single initial condition.

    Parameters
    ----------
    E_true : float
        True energy
    Q_true : float
        True initial phase
    delta_tau : float
        Prediction horizon (time)
    epsilon : float
        Measurement error (std) as fraction of typical scale
    seed : int
        Random seed

    Returns
    -------
    delta_P : float
        Action prediction error
    delta_Q : float
        Phase prediction error (wrapped to [-π, π])
    """
    if seed is not None:
        np.random.seed(seed)

    # True frequency
    omega_true = pendulum_omega(E_true)
    if np.isnan(omega_true):
        return np.nan, np.nan

    T_period = 2 * np.pi / omega_true

    # True evolution
    Q_actual = (Q_true + omega_true * delta_tau) % (2 * np.pi)
    P_actual = E_true  # Energy conserved (using E as action proxy)

    # Measurement with error
    # epsilon is relative to |E+1| (distance from minimum)
    E_scale = abs(E_true + 1) + 0.1  # Add small offset to avoid zero
    E_meas = E_true + np.random.normal(0, epsilon * E_scale)
    Q_meas = Q_true + np.random.normal(0, epsilon)  # Small angle noise

    # Clamp measured E to valid range
    E_meas = np.clip(E_meas, -0.999, 0.999)

    # Predicted frequency from measured E
    omega_meas = pendulum_omega(E_meas)
    if np.isnan(omega_meas):
        return np.nan, np.nan

    # Predicted evolution using measured values
    Q_pred = (Q_meas + omega_meas * delta_tau) % (2 * np.pi)
    P_pred = E_meas  # Prediction of conserved quantity

    # Errors
    delta_P = abs(P_pred - P_actual)
    delta_Q = abs(wrap_angle(Q_pred - Q_actual))

    return delta_P, delta_Q


def run_prediction_error_sweep(E_true, epsilon, delta_tau_ratios, n_samples=200, seed=42):
    """
    Sweep over prediction horizons and compute mean errors.
    """
    np.random.seed(seed)

    omega_true = pendulum_omega(E_true)
    T_period = 2 * np.pi / omega_true

    results = []

    for ratio in delta_tau_ratios:
        delta_tau = ratio * T_period

        delta_P_samples = []
        delta_Q_samples = []

        for i in range(n_samples):
            # Random true initial phase
            Q_true = np.random.uniform(0, 2 * np.pi)

            dP, dQ = prediction_error_single(
                E_true, Q_true, delta_tau, epsilon,
                seed=seed + i + int(ratio * 1000)
            )

            if not np.isnan(dP):
                delta_P_samples.append(dP)
                delta_Q_samples.append(dQ)

        if delta_P_samples:
            results.append({
                'ratio': ratio,
                'delta_tau': delta_tau,
                'T_period': T_period,
                'mean_delta_P': np.mean(delta_P_samples),
                'std_delta_P': np.std(delta_P_samples),
                'mean_delta_Q': np.mean(delta_Q_samples),
                'std_delta_Q': np.std(delta_Q_samples),
                'n_samples': len(delta_P_samples)
            })

    return results


def theoretical_critical_time(E, epsilon):
    """
    Theoretical estimate of Δτ_critical where δQ ~ π.

    δQ ~ |dω/dE| · δE · Δτ
    Set δQ = π, δE = ε·E_scale:
    Δτ_critical = π / (|dω/dE| · ε · E_scale)

    Returns Δτ_critical / T
    """
    omega = pendulum_omega(E)
    T_period = 2 * np.pi / omega

    domega_dE = pendulum_omega_derivative(E)
    if np.isnan(domega_dE) or domega_dE == 0:
        return np.nan

    E_scale = abs(E + 1) + 0.1
    delta_E = epsilon * E_scale

    # Time for phase error to reach π
    delta_tau_critical = np.pi / (abs(domega_dE) * delta_E)

    return delta_tau_critical / T_period


def run_full_test():
    """
    Main test: Prediction error vs time horizon.
    """
    print("=" * 70)
    print("PREDICTION WITH MEASUREMENT ERROR")
    print("=" * 70)
    print("\nMechanism: measurement error ε → frequency error → phase drift")
    print("δQ grows with Δτ while δP stays bounded")

    # Parameters
    epsilons = [0.01, 0.05, 0.10]  # Measurement error levels
    E_values = [-0.5, 0.0, 0.5]   # Different energies
    delta_tau_ratios = np.logspace(-1, 2, 30)  # 0.1T to 100T

    all_results = {}

    for epsilon in epsilons:
        print(f"\n[Test] Measurement error ε = {epsilon:.0%}")
        print("-" * 60)

        for E in E_values:
            omega = pendulum_omega(E)
            T = 2 * np.pi / omega
            domega_dE = pendulum_omega_derivative(E)

            print(f"\n  E = {E:.2f}: T = {T:.2f}, dω/dE = {domega_dE:.4f}")

            results = run_prediction_error_sweep(
                E, epsilon, delta_tau_ratios, n_samples=200
            )

            # Find critical time (where δQ first exceeds π/2)
            critical_ratio = None
            for r in results:
                if r['mean_delta_Q'] > np.pi / 2:
                    critical_ratio = r['ratio']
                    break

            # Theoretical prediction
            theory_critical = theoretical_critical_time(E, epsilon)

            print(f"    Δτ_critical (empirical):   {critical_ratio:.1f}T" if critical_ratio else "    Δτ_critical (empirical):   > 100T")
            print(f"    Δτ_critical (theoretical): {theory_critical:.1f}T" if not np.isnan(theory_critical) else "    Δτ_critical (theoretical): N/A")

            all_results[(epsilon, E)] = {
                'results': results,
                'critical_ratio': critical_ratio,
                'theory_critical': theory_critical,
                'omega': omega,
                'T_period': T,
                'domega_dE': domega_dE
            }

    # Detailed output for one case
    print("\n" + "=" * 70)
    print("DETAILED RESULTS (ε = 5%, E = 0.0)")
    print("=" * 70)

    key = (0.05, 0.0)
    if key in all_results:
        res = all_results[key]['results']
        print(f"\n{'Δτ/T':>10} {'δP (mean)':>12} {'δQ (mean)':>12} {'δQ/π':>10} {'Regime':>15}")
        print("-" * 65)

        for r in res[::3]:  # Every 3rd result
            ratio = r['ratio']
            dP = r['mean_delta_P']
            dQ = r['mean_delta_Q']
            dQ_ratio = dQ / np.pi

            if dQ_ratio < 0.2:
                regime = "Classical"
            elif dQ_ratio < 0.6:
                regime = "Transition"
            else:
                regime = "Quantum-like"

            print(f"{ratio:>10.2f} {dP:>12.4f} {dQ:>12.4f} {dQ_ratio:>10.2f} {regime:>15}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: δP and δQ vs Δτ/T for each ε
    for idx, epsilon in enumerate(epsilons):
        ax = axes[0, idx]

        for E in E_values:
            key = (epsilon, E)
            if key not in all_results:
                continue

            res = all_results[key]['results']
            ratios = [r['ratio'] for r in res]
            dP = [r['mean_delta_P'] for r in res]
            dQ = [r['mean_delta_Q'] for r in res]

            ax.semilogx(ratios, dP, 'o-', label=f'δP (E={E})', markersize=3, alpha=0.7)
            ax.semilogx(ratios, dQ, 's--', label=f'δQ (E={E})', markersize=3, alpha=0.7)

        ax.axhline(np.pi, color='red', linestyle=':', alpha=0.5, label='π (random)')
        ax.axhline(np.pi/2, color='orange', linestyle=':', alpha=0.5, label='π/2')
        ax.set_xlabel('Δτ / T')
        ax.set_ylabel('Prediction error')
        ax.set_title(f'ε = {epsilon:.0%}')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, np.pi * 1.2)

    # Row 2: Analysis plots

    # Plot 2a: Critical time vs ε
    ax = axes[1, 0]
    for E in E_values:
        critical_ratios = []
        for epsilon in epsilons:
            key = (epsilon, E)
            if key in all_results and all_results[key]['critical_ratio']:
                critical_ratios.append((epsilon, all_results[key]['critical_ratio']))

        if critical_ratios:
            eps_vals, crit_vals = zip(*critical_ratios)
            ax.loglog(eps_vals, crit_vals, 'o-', label=f'E = {E}', markersize=8)

    # Theoretical line: Δτ_crit ∝ 1/ε
    eps_line = np.array([0.005, 0.2])
    ax.loglog(eps_line, 10 / eps_line, 'k--', alpha=0.5, label='∝ 1/ε')

    ax.set_xlabel('Measurement error ε')
    ax.set_ylabel('Δτ_critical / T')
    ax.set_title('Critical time vs measurement error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2b: δP stays bounded
    ax = axes[1, 1]
    key = (0.05, 0.0)
    if key in all_results:
        res = all_results[key]['results']
        ratios = [r['ratio'] for r in res]
        dP = [r['mean_delta_P'] for r in res]
        dQ = [r['mean_delta_Q'] for r in res]

        ax.semilogx(ratios, dP, 'b-', linewidth=2, label='δP (action)')
        ax.semilogx(ratios, dQ, 'r-', linewidth=2, label='δQ (phase)')
        ax.axhline(np.pi, color='red', linestyle=':', alpha=0.5)

        # Annotate
        ax.annotate('P stays bounded!', xy=(50, dP[-1]), fontsize=10, color='blue')
        ax.annotate('Q randomizes', xy=(50, np.pi), fontsize=10, color='red')

    ax.set_xlabel('Δτ / T')
    ax.set_ylabel('Prediction error')
    ax.set_title('δP bounded, δQ grows (ε=5%, E=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, np.pi * 1.2)

    # Plot 2c: Summary text
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
    PREDICTION WITH MEASUREMENT ERROR
    ==================================

    Mechanism:
    1. Measure (P, Q) with error ε
    2. Estimate ω from measured P
    3. Predict Q(Δτ) = Q + ω·Δτ

    Key result:
    - δP stays bounded at ~ε (action conserved)
    - δQ grows until ~π (phase randomizes)

    Critical timescale:
    Δτ_crit / T ~ π / (|dω/dP|·ε·T)

    At Δτ > Δτ_crit:
    - Phase is effectively random
    - Only action remains predictable
    - System appears "quantized"

    This is Glinsky's mechanism:
    Finite measurement precision + time evolution
    → phase unpredictability → effective quantization

    The discrete structure emerges from
    FINITE PRECISION, not from dynamics.
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('prediction_measurement_error.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to prediction_measurement_error.png")
    plt.show()

    # Final summary
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check if pattern holds
    pattern_confirmed = True
    for epsilon in epsilons:
        for E in E_values:
            key = (epsilon, E)
            if key in all_results:
                res = all_results[key]['results']
                if res:
                    # Check δP bounded
                    dP_vals = [r['mean_delta_P'] for r in res]
                    dP_bounded = max(dP_vals) < 0.5  # Should stay small

                    # Check δQ grows to ~π
                    dQ_vals = [r['mean_delta_Q'] for r in res]
                    dQ_saturates = max(dQ_vals) > 0.8 * np.pi

                    if not (dP_bounded and dQ_saturates):
                        pattern_confirmed = False

    if pattern_confirmed:
        print("""
    GLINSKY MECHANISM CONFIRMED!

    Measurement error + time evolution produces:
    1. δP bounded: Action remains predictable at all timescales
    2. δQ → π: Phase becomes unpredictable after Δτ_critical

    Key formula:
        Δτ_critical / T ≈ π / (|dω/dE| · ε)

    Physical interpretation:
    - Finite precision ε sets "effective ℏ" for the system
    - At timescales >> Δτ_critical, phase is gauge variable
    - Only action (quantum number) remains observable

    This is NOT quantum mechanics - it's classical mechanics
    with finite measurement precision. The "quantization"
    emerges from information limits, not from ℏ.
        """)
    else:
        print("""
    Results require interpretation.
    Check individual energy/epsilon combinations.
        """)

    return all_results


if __name__ == "__main__":
    results = run_full_test()
