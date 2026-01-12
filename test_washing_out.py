"""
Non-Integer P "Washing Out" Test

Tests whether non-integer action values (in units of I₀) are less stable
than integer values under coarse observation.

Hypothesis: If Bohr-Sommerfeld quantization is physical (not just a counting convention),
then:
- Integer P/I₀: stable "standing waves", low estimation variance
- Non-integer P/I₀: unstable, "wash out" under repeated observations

This would be analogous to how non-eigenstate superpositions decohere faster
than eigenstates in quantum mechanics.

Note for future work:
    Interaction/measurement perturbation mechanism not yet tested.
    Glinsky suggests measurement requires force application, implying
    action exchange in units of I₀. This may be the mechanism that
    enforces discretization. Needs separate investigation with coupled
    system model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian
from hst import extract_features


def pendulum_omega(E):
    """Theoretical frequency"""
    if E >= 1:
        return 0.01
    k2 = (1 + E) / 2
    if k2 <= 0:
        return 1.0
    k = np.sqrt(k2)
    if k >= 1:
        return 0.01
    return np.pi / (2 * ellipk(k**2))


def energy_to_action(E):
    """
    Approximate action for pendulum.
    For small oscillations: J ≈ E/ω₀ = E (since ω₀ = 1 at bottom)
    For larger oscillations: use the actual formula.
    """
    from scipy.special import ellipe
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 <= 0 or k2 >= 1:
        return max(E + 1, 0.01)
    k = np.sqrt(k2)
    K = ellipk(k**2)
    E_ellip = ellipe(k**2)
    J = (8/np.pi) * (E_ellip - (1 - k**2) * K)
    return J


def action_to_energy(J_target, tol=1e-6, max_iter=100):
    """
    Invert action to energy via bisection.
    """
    E_low, E_high = -0.999, 0.999

    for _ in range(max_iter):
        E_mid = (E_low + E_high) / 2
        J_mid = energy_to_action(E_mid)

        if np.isnan(J_mid):
            E_high = E_mid
            continue

        if abs(J_mid - J_target) < tol:
            return E_mid
        elif J_mid < J_target:
            E_low = E_mid
        else:
            E_high = E_mid

    return (E_low + E_high) / 2


def compute_I0():
    """
    Characteristic scale I₀ = E₀/ω₀

    Use E = 0 (amplitude = 90°) as characteristic.
    """
    E_char = 0.0
    omega_char = pendulum_omega(E_char)
    I0 = (E_char + 1) / omega_char  # Shift E so minimum is 0
    return I0, omega_char


def generate_at_target_action(J_target, n_observations=20, delta_tau_ratio=5.0, seed=None):
    """
    Generate multiple observations of a trajectory with target action J.

    Each observation:
    1. Starts at random time offset (observation uncertainty)
    2. Extracts features
    3. Estimates P from features

    Returns list of P estimates.
    """
    if seed is not None:
        np.random.seed(seed)

    pendulum = PendulumOscillator()

    # Find energy for this action
    E_target = action_to_energy(J_target)
    if np.isnan(E_target) or E_target >= 0.99:
        return None, None

    # Compute period
    omega = pendulum_omega(E_target)
    T_period = 2 * np.pi / omega
    dt = 0.01

    # Initial condition at turning point
    q_max = np.arccos(-E_target) if E_target > -1 else np.pi * 0.95
    q0 = q_max
    p0 = 0.0

    # Generate long trajectory
    T_sim = max(50 * T_period, 200)
    try:
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=dt)
    except:
        return None, None

    if len(z) < 1024:
        return None, None

    # True action (constant along orbit)
    J_actual = energy_to_action(E_actual)
    P_true = np.mean(np.abs(z))  # Simple proxy

    # Multiple observations with coarse timing uncertainty
    delta_tau = delta_tau_ratio * T_period
    max_offset_samples = max(1, int(delta_tau / dt))

    P_estimates = []
    features_list = []

    window_size = 512

    for obs_idx in range(n_observations):
        # Random observation start
        offset_samples = np.random.randint(0, max_offset_samples)
        start_idx = offset_samples % max(1, len(z) - window_size - 100)

        if start_idx + window_size >= len(z):
            start_idx = max(0, len(z) - window_size - 1)

        z_obs = z[start_idx:start_idx + window_size]

        if len(z_obs) < window_size:
            continue

        try:
            feat = extract_features(z_obs, J=3, wavelet_name='db8')
            features_list.append(feat)

            # Simple P estimate: mean magnitude
            P_est = np.mean(np.abs(z_obs))
            P_estimates.append(P_est)
        except:
            pass

    return {
        'J_target': J_target,
        'J_actual': J_actual,
        'E_actual': E_actual,
        'P_true': P_true,
        'P_estimates': np.array(P_estimates),
        'features': np.array(features_list) if features_list else None
    }


def run_washing_out_test():
    """
    Main test: Do non-integer action values wash out under coarse observation?
    """
    print("=" * 70)
    print("NON-INTEGER P WASHING OUT TEST")
    print("=" * 70)

    # Compute I₀
    I0, omega_char = compute_I0()
    print(f"\nCharacteristic scale: I₀ = {I0:.4f}, ω_char = {omega_char:.4f}")

    # Target action values spanning continuous range
    # Include both integer and non-integer in units of I₀
    J_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    J_targets = [m * I0 for m in J_multipliers]

    print(f"\nTarget actions (in units of I₀): {J_multipliers}")
    print(f"Target actions (absolute): {[f'{J:.3f}' for J in J_targets]}")

    # Generate data
    print("\n[1] Generating observations at each target action...")

    n_observations = 50
    delta_tau_ratio = 5.0  # Coarse observation

    results = []

    for J_target, J_mult in zip(J_targets, J_multipliers):
        data = generate_at_target_action(
            J_target,
            n_observations=n_observations,
            delta_tau_ratio=delta_tau_ratio,
            seed=42 + int(J_mult * 1000)
        )

        if data is None or not isinstance(data, dict) or len(data.get('P_estimates', [])) < 10:
            print(f"  J/I₀ = {J_mult:.1f}: Failed to generate")
            continue

        P_est = data['P_estimates']
        P_true = data['P_true']

        # Metrics
        error_mean = np.mean(np.abs(P_est - P_true))
        error_std = np.std(np.abs(P_est - P_true))
        variance = np.var(P_est)
        cv = np.std(P_est) / np.mean(P_est) if np.mean(P_est) > 0 else 0

        is_integer = abs(J_mult - round(J_mult)) < 0.01

        results.append({
            'J_mult': J_mult,
            'J_target': J_target,
            'J_actual': data['J_actual'],
            'P_true': P_true,
            'P_estimates': P_est,
            'error_mean': error_mean,
            'error_std': error_std,
            'variance': variance,
            'cv': cv,
            'is_integer': is_integer
        })

        int_marker = "INT" if is_integer else "   "
        print(f"  J/I₀ = {J_mult:.1f} {int_marker}: error = {error_mean:.4f} ± {error_std:.4f}, CV = {cv:.4f}")

    # Analysis
    print("\n[2] Comparing integer vs non-integer stability...")

    integer_results = [r for r in results if r['is_integer']]
    nonint_results = [r for r in results if not r['is_integer']]

    if integer_results and nonint_results:
        int_cv_mean = np.mean([r['cv'] for r in integer_results])
        nonint_cv_mean = np.mean([r['cv'] for r in nonint_results])

        int_error_mean = np.mean([r['error_mean'] for r in integer_results])
        nonint_error_mean = np.mean([r['error_mean'] for r in nonint_results])

        print(f"\n  Integer J/I₀:")
        print(f"    Mean CV: {int_cv_mean:.4f}")
        print(f"    Mean error: {int_error_mean:.4f}")

        print(f"\n  Non-integer J/I₀:")
        print(f"    Mean CV: {nonint_cv_mean:.4f}")
        print(f"    Mean error: {nonint_error_mean:.4f}")

        print(f"\n  Ratio (nonint/int):")
        print(f"    CV ratio: {nonint_cv_mean / int_cv_mean:.2f}")
        print(f"    Error ratio: {nonint_error_mean / int_error_mean:.2f}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. CV vs J/I₀
    ax = axes[0, 0]
    J_mults = [r['J_mult'] for r in results]
    cvs = [r['cv'] for r in results]
    colors = ['blue' if r['is_integer'] else 'red' for r in results]
    ax.bar(range(len(J_mults)), cvs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(J_mults)))
    ax.set_xticklabels([f'{m:.1f}' for m in J_mults])
    ax.set_xlabel('J / I₀')
    ax.set_ylabel('CV of P estimates')
    ax.set_title('Estimation Stability (blue=integer, red=non-int)')
    ax.grid(True, alpha=0.3)

    # 2. Error vs J/I₀
    ax = axes[0, 1]
    errors = [r['error_mean'] for r in results]
    ax.bar(range(len(J_mults)), errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(J_mults)))
    ax.set_xticklabels([f'{m:.1f}' for m in J_mults])
    ax.set_xlabel('J / I₀')
    ax.set_ylabel('Mean |P_est - P_true|')
    ax.set_title('Estimation Error')
    ax.grid(True, alpha=0.3)

    # 3. Histogram of P_est for integer example
    ax = axes[0, 2]
    if integer_results:
        r = integer_results[1] if len(integer_results) > 1 else integer_results[0]
        ax.hist(r['P_estimates'], bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(r['P_true'], color='red', linestyle='--', label=f'True P')
        ax.set_xlabel('P estimate')
        ax.set_ylabel('Density')
        ax.set_title(f'Integer: J/I₀ = {r["J_mult"]:.1f}')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Histogram of P_est for non-integer example
    ax = axes[1, 0]
    if nonint_results:
        r = nonint_results[1] if len(nonint_results) > 1 else nonint_results[0]
        ax.hist(r['P_estimates'], bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(r['P_true'], color='red', linestyle='--', label=f'True P')
        ax.set_xlabel('P estimate')
        ax.set_ylabel('Density')
        ax.set_title(f'Non-integer: J/I₀ = {r["J_mult"]:.1f}')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. P_est distribution across all targets
    ax = axes[1, 1]
    all_P_est = []
    all_J_mult = []
    for r in results:
        all_P_est.extend(r['P_estimates'])
        all_J_mult.extend([r['J_mult']] * len(r['P_estimates']))

    ax.scatter(all_J_mult, all_P_est, alpha=0.3, s=10)
    # Add true P values
    for r in results:
        ax.scatter(r['J_mult'], r['P_true'], color='red', s=50, marker='x')
    ax.set_xlabel('J / I₀')
    ax.set_ylabel('P estimate')
    ax.set_title('All P estimates (red X = true P)')
    ax.grid(True, alpha=0.3)

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Verdict
    if integer_results and nonint_results:
        cv_ratio = nonint_cv_mean / int_cv_mean if int_cv_mean > 0 else 1
        washing_detected = cv_ratio > 1.5
    else:
        washing_detected = False
        cv_ratio = 0

    summary = f"""
    NON-INTEGER P WASHING OUT TEST
    ==============================

    Observation: Delta_tau/T = {delta_tau_ratio}
    I₀ = {I0:.4f}

    Hypothesis: Non-integer J/I₀ should be
    less stable than integer values.

    Results:
    - Integer CV mean: {int_cv_mean:.4f}
    - Non-int CV mean: {nonint_cv_mean:.4f}
    - CV ratio: {cv_ratio:.2f}

    Verdict: {'WASHING DETECTED' if washing_detected else 'NO CLEAR WASHING'}

    Interpretation:
    {'Non-integer actions show higher variance,'
    if washing_detected else
    'Integer and non-integer show similar stability,'}
    {'suggesting instability of non-eigenstates.'
    if washing_detected else
    'suggesting P is continuous (no quantization).'}

    Note: This tests observation stability, not
    interaction-induced discretization. See docs
    for future work on measurement perturbation.
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('washing_out_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to washing_out_test.png")
    plt.show()

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if washing_detected:
        print(f"""
    WASHING EFFECT DETECTED!

    Non-integer J/I₀ values show {cv_ratio:.1f}x higher variance
    than integer values under coarse observation.

    Interpretation:
    - Integer actions = stable "standing waves"
    - Non-integer actions = unstable, wash out

    This supports Bohr-Sommerfeld quantization as physical,
    not just a counting convention.
        """)
    else:
        print(f"""
    NO CLEAR WASHING EFFECT

    CV ratio (nonint/int): {cv_ratio:.2f}

    Both integer and non-integer J/I₀ show similar stability.
    This suggests action is truly continuous, and quantization
    emerges only from periodicity requirements on probability
    (not from intrinsic instability of non-integer values).

    Note: The physical mechanism may require explicit
    measurement interaction (action exchange in units of I₀).
    This is noted for future investigation.
        """)

    return results


if __name__ == "__main__":
    results = run_washing_out_test()
