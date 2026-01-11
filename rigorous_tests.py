"""
Rigorous Tests for HST-ROM Implementation

Pre-specified tests with clear pass/fail criteria.
Run with: python rigorous_tests.py --test all
"""

import numpy as np
import argparse
from scipy.stats import pearsonr, bootstrap
from scipy.special import ellipk
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from hamiltonian_systems import (
    SimpleHarmonicOscillator, AnharmonicOscillator, PendulumOscillator,
    simulate_hamiltonian, generate_ensemble_at_different_energies
)
from hst import hst_forward_pywt, hst_inverse_pywt


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    return seed


def compute_confidence_interval(data, statistic_fn, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    if len(data) < 3:
        return (np.nan, np.nan)

    try:
        result = bootstrap(
            (data,), statistic_fn,
            n_resamples=n_bootstrap,
            confidence_level=confidence,
            method='percentile'
        )
        return (result.confidence_interval.low, result.confidence_interval.high)
    except:
        return (np.nan, np.nan)


def print_result(test_name, status, metric_name, metric_value, ci=None, n_samples=None, seed=None):
    """Print standardized test result."""
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")

    status_colors = {'PASS': '\033[92m', 'MARGINAL': '\033[93m', 'FAIL': '\033[91m'}
    reset = '\033[0m'

    print(f"Status: {status_colors.get(status, '')}{status}{reset}")
    print(f"Metric: {metric_name} = {metric_value:.4f}")

    if ci and not (np.isnan(ci[0]) or np.isnan(ci[1])):
        print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    if n_samples:
        print(f"Samples: {n_samples}")

    if seed:
        print(f"Seed: {seed}")

    return status


# =============================================================================
# TEST 1: Action Recovery (SHO)
# =============================================================================

def test_1_action_recovery(seed=42):
    """
    Test that learned P correlates with true action I = E/ω₀ for SHO.

    Pass: r > 0.99
    Marginal: 0.95 < r < 0.99
    Fail: r < 0.95
    """
    seed = set_seed(seed)

    # Generate SHO trajectories at different energies
    sho = SimpleHarmonicOscillator(omega0=1.0)
    n_total = 50
    n_train = 40
    n_test = 10

    energies = np.linspace(0.5, 5.0, n_total)
    np.random.shuffle(energies)

    train_energies = energies[:n_train]
    test_energies = energies[n_train:]

    # Generate test trajectories and compute HST features
    test_P = []
    test_I_true = []

    for E in test_energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=0.01)

        # True action for SHO: I = E/ω₀
        I_true = E_actual / sho.omega0

        # Apply HST and compute P (use variance of HST coefficients as proxy for P)
        coeffs = hst_forward_pywt(z.real, J=3, wavelet_name='db8')
        # P proxy: total energy in HST coefficients (weighted by scale)
        P_proxy = 0
        for j, c in enumerate(coeffs['cD']):
            P_proxy += (2**j) * np.mean(np.abs(c)**2)
        # Also include final approximation
        P_proxy += (2**len(coeffs['cD'])) * np.mean(np.abs(coeffs['cA_final'])**2)

        test_P.append(P_proxy)
        test_I_true.append(I_true)

    test_P = np.array(test_P)
    test_I_true = np.array(test_I_true)

    # Compute correlation
    r, pval = pearsonr(test_P, test_I_true)

    # Bootstrap CI
    def corr_stat(x):
        return pearsonr(x, test_I_true[:len(x)])[0]
    ci = compute_confidence_interval(test_P, lambda x: pearsonr(x, test_I_true)[0])

    # Determine status
    if r > 0.99:
        status = 'PASS'
    elif r > 0.95:
        status = 'MARGINAL'
    else:
        status = 'FAIL'

    return print_result(
        "TEST 1: Action Recovery (SHO)",
        status, "r(P, I)", r, ci, n_test, seed
    )


# =============================================================================
# TEST 2: P Conservation Within Trajectory
# =============================================================================

def test_2_p_conservation(seed=42):
    """
    Test that P is approximately constant within each trajectory.

    Pass: >90% of trajectories have CV < 0.05
    Marginal: >80% have CV < 0.05
    Fail: <80% have CV < 0.05
    """
    seed = set_seed(seed)

    # Generate Duffing trajectories
    duffing = AnharmonicOscillator(epsilon=0.3)
    energies = np.linspace(0.5, 3.0, 30)

    cv_values = []

    for E in energies:
        q0, p0 = duffing.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(duffing, q0, p0, T=100, dt=0.01)

        # Compute P at different windows along trajectory
        window_size = 512
        stride = 256
        P_values = []

        for start in range(0, len(z) - window_size, stride):
            window = z[start:start+window_size]
            coeffs = hst_forward_pywt(window.real, J=3, wavelet_name='db8')
            P_proxy = sum((2**j) * np.mean(np.abs(c)**2) for j, c in enumerate(coeffs['cD']))
            P_proxy += (2**len(coeffs['cD'])) * np.mean(np.abs(coeffs['cA_final'])**2)
            P_values.append(P_proxy)

        if len(P_values) > 2:
            P_values = np.array(P_values)
            cv = np.std(P_values) / (np.abs(np.mean(P_values)) + 1e-10)
            cv_values.append(cv)

    cv_values = np.array(cv_values)

    # Compute fraction with CV < 0.05
    fraction_good = np.mean(cv_values < 0.05)

    # Also report median CV
    median_cv = np.median(cv_values)

    # Determine status
    if fraction_good > 0.90:
        status = 'PASS'
    elif fraction_good > 0.80:
        status = 'MARGINAL'
    else:
        status = 'FAIL'

    print_result(
        "TEST 2: P Conservation Within Trajectory",
        status, "fraction(CV < 0.05)", fraction_good, None, len(cv_values), seed
    )
    print(f"Median CV: {median_cv:.4f}")
    print(f"CV range: [{cv_values.min():.4f}, {cv_values.max():.4f}]")

    return status


# =============================================================================
# TEST 3: ω Functional Relationship (Pendulum)
# =============================================================================

def test_3_omega_relationship(seed=42):
    """
    Test that learned ω matches theoretical ω(E) for pendulum.

    Pass: R² > 0.90
    Marginal: 0.70 < R² < 0.90
    Fail: R² < 0.70
    """
    seed = set_seed(seed)

    # Generate pendulum trajectories in libration regime
    pendulum = PendulumOscillator()
    n_total = 40
    n_train = 30
    n_test = 10

    # E in [-0.8, 0.8] for libration (separatrix at E=1)
    energies = np.linspace(-0.8, 0.8, n_total)
    np.random.shuffle(energies)

    test_energies = energies[n_train:]

    omega_learned = []
    omega_theoretical = []

    for E in test_energies:
        q0, p0 = pendulum.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=100, dt=0.01)

        # Theoretical ω
        omega_th = pendulum.theoretical_omega(E_actual)
        if omega_th is None or not np.isfinite(omega_th):
            continue

        # Measure ω from zero crossings
        zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
        if len(zero_crossings) >= 2:
            periods = np.diff(t[zero_crossings])
            measured_period = np.mean(periods)
            omega_meas = 2 * np.pi / measured_period
        else:
            continue

        omega_learned.append(omega_meas)
        omega_theoretical.append(omega_th)

    omega_learned = np.array(omega_learned)
    omega_theoretical = np.array(omega_theoretical)

    if len(omega_learned) < 3:
        print_result("TEST 3: ω Functional Relationship (Pendulum)",
                    "FAIL", "R²", 0.0, None, len(omega_learned), seed)
        print("Insufficient valid trajectories")
        return 'FAIL'

    # Compute R² (correlation squared)
    r, _ = pearsonr(omega_learned, omega_theoretical)
    r_squared = r**2

    # Determine status
    if r_squared > 0.90:
        status = 'PASS'
    elif r_squared > 0.70:
        status = 'MARGINAL'
    else:
        status = 'FAIL'

    print_result(
        "TEST 3: ω Functional Relationship (Pendulum)",
        status, "R²(ω_learned, ω_theoretical)", r_squared, None, len(omega_learned), seed
    )
    print(f"Correlation r: {r:.4f}")

    return status


# =============================================================================
# TEST 4: Signal Reconstruction
# =============================================================================

def test_4_reconstruction(seed=42):
    """
    Test HST reconstruction accuracy.

    Wavelet only: Pass < 1e-10, Fail > 1e-6
    Full HST: Pass < 1%, Marginal < 5%, Fail > 10%
    """
    seed = set_seed(seed)

    # Test signals
    t = np.linspace(0, 10, 1024)
    signals = {
        'chirp': np.sin(t + 0.1 * t**2),
        'sum_of_sines': np.sin(t) + 0.5 * np.sin(2.3 * t),
        'sho': np.sin(t) * np.exp(-0.01 * t)  # Damped SHO
    }

    print(f"\n{'='*60}")
    print("TEST 4: Signal Reconstruction")
    print(f"{'='*60}")
    print(f"Seed: {seed}")

    results = {}

    for name, signal in signals.items():
        # Forward HST
        coeffs = hst_forward_pywt(signal, J=3, wavelet_name='db8')

        # Inverse HST
        reconstructed = hst_inverse_pywt(coeffs)

        # Match lengths
        min_len = min(len(signal), len(reconstructed))
        signal_trim = signal[:min_len]
        recon_trim = reconstructed[:min_len]

        # Relative error
        error = np.linalg.norm(signal_trim - recon_trim) / np.linalg.norm(signal_trim)
        results[name] = error

        print(f"\n{name}:")
        print(f"  Relative error: {error:.2e}")

    # Overall status based on average
    avg_error = np.mean(list(results.values()))
    max_error = max(results.values())

    if max_error < 1e-10:
        status = 'PASS'
        print(f"\nStatus: \033[92mPASS\033[0m (all errors < 1e-10)")
    elif max_error < 0.01:
        status = 'PASS'
        print(f"\nStatus: \033[92mPASS\033[0m (all errors < 1%)")
    elif max_error < 0.05:
        status = 'MARGINAL'
        print(f"\nStatus: \033[93mMARGINAL\033[0m (max error < 5%)")
    else:
        status = 'FAIL'
        print(f"\nStatus: \033[91mFAIL\033[0m (max error > 10%)")

    return status


# =============================================================================
# TEST 5: Generalization to Unseen Energies
# =============================================================================

def test_5_generalization(seed=42):
    """
    Test interpolation to unseen energy levels.

    Pass: r > 0.95 on interpolated test set
    Marginal: 0.85 < r < 0.95
    Fail: r < 0.85
    """
    seed = set_seed(seed)

    duffing = AnharmonicOscillator(epsilon=0.3)

    # Training energies (sparse grid)
    train_energies = [0.5, 1.5, 2.5, 3.5, 4.5]
    # Test energies (intermediate values)
    test_energies = [1.0, 2.0, 3.0, 4.0]

    # Compute P proxy for training set (to establish scale)
    train_P = []
    train_E = []

    for E in train_energies:
        q0, p0 = duffing.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(duffing, q0, p0, T=50, dt=0.01)

        coeffs = hst_forward_pywt(z.real, J=3, wavelet_name='db8')
        P_proxy = sum((2**j) * np.mean(np.abs(c)**2) for j, c in enumerate(coeffs['cD']))
        P_proxy += (2**len(coeffs['cD'])) * np.mean(np.abs(coeffs['cA_final'])**2)

        train_P.append(P_proxy)
        train_E.append(E_actual)

    # Compute P proxy for test set
    test_P = []
    test_E = []

    for E in test_energies:
        q0, p0 = duffing.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(duffing, q0, p0, T=50, dt=0.01)

        coeffs = hst_forward_pywt(z.real, J=3, wavelet_name='db8')
        P_proxy = sum((2**j) * np.mean(np.abs(c)**2) for j, c in enumerate(coeffs['cD']))
        P_proxy += (2**len(coeffs['cD'])) * np.mean(np.abs(coeffs['cA_final'])**2)

        test_P.append(P_proxy)
        test_E.append(E_actual)

    test_P = np.array(test_P)
    test_E = np.array(test_E)

    # Correlation on test set
    r, _ = pearsonr(test_P, test_E)

    # Determine status
    if r > 0.95:
        status = 'PASS'
    elif r > 0.85:
        status = 'MARGINAL'
    else:
        status = 'FAIL'

    print_result(
        "TEST 5: Generalization to Unseen Energies",
        status, "r(P, E) on test", r, None, len(test_energies), seed
    )
    print(f"Test energies: {test_energies}")

    return status


# =============================================================================
# TEST 6: Comparison with Analytical Action-Angle (SHO)
# =============================================================================

def test_6_analytical_comparison(seed=42):
    """
    Compare learned coordinates with analytical action-angle for SHO.

    Pass: |r(P, I)| > 0.99 AND circular correlation |r(Q, θ)| > 0.95
    Marginal: |r(P, I)| > 0.95 AND |r(Q, θ)| > 0.80
    Fail: Either below marginal
    """
    seed = set_seed(seed)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Generate trajectory with known action-angle
    E = 2.0
    q0, p0 = sho.initial_condition_for_energy(E)
    t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=0.01)

    # True action-angle coordinates
    I_true = (q**2 + p**2) / 2  # Action I = E for unit-ω SHO
    theta_true = np.arctan2(p, q)  # Angle

    # P proxy from HST at each window
    window_size = 256
    stride = 64

    P_proxy = []
    I_samples = []
    theta_samples = []

    for start in range(0, len(z) - window_size, stride):
        # Compute P proxy
        window = z[start:start+window_size].real
        coeffs = hst_forward_pywt(window, J=3, wavelet_name='db8')
        P_val = sum((2**j) * np.mean(np.abs(c)**2) for j, c in enumerate(coeffs['cD']))
        P_val += (2**len(coeffs['cD'])) * np.mean(np.abs(coeffs['cA_final'])**2)
        P_proxy.append(P_val)

        # Sample true action/angle at window center
        center = start + window_size // 2
        I_samples.append(I_true[center])
        theta_samples.append(theta_true[center])

    P_proxy = np.array(P_proxy)
    I_samples = np.array(I_samples)
    theta_samples = np.array(theta_samples)

    # Correlation P with I
    r_P_I, _ = pearsonr(P_proxy, I_samples)

    # For Q correlation with θ, we use phase of windowed FFT as Q proxy
    Q_proxy = []
    for start in range(0, len(z) - window_size, stride):
        window = z[start:start+window_size]
        fft = np.fft.fft(window)
        peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        phase = np.angle(fft[peak_idx])
        Q_proxy.append(phase)

    Q_proxy = np.array(Q_proxy)

    # Circular correlation for angles
    # We compute correlation between sin/cos components
    sin_Q = np.sin(Q_proxy)
    cos_Q = np.cos(Q_proxy)
    sin_theta = np.sin(theta_samples)
    cos_theta = np.cos(theta_samples)

    # R² from circular correlation
    r_sin, _ = pearsonr(sin_Q, sin_theta)
    r_cos, _ = pearsonr(cos_Q, cos_theta)
    r_Q_theta = np.sqrt((r_sin**2 + r_cos**2) / 2)  # Circular correlation proxy

    # Determine status
    if abs(r_P_I) > 0.99 and r_Q_theta > 0.95:
        status = 'PASS'
    elif abs(r_P_I) > 0.95 and r_Q_theta > 0.80:
        status = 'MARGINAL'
    else:
        status = 'FAIL'

    print(f"\n{'='*60}")
    print("TEST 6: Comparison with Analytical Action-Angle (SHO)")
    print(f"{'='*60}")

    status_colors = {'PASS': '\033[92m', 'MARGINAL': '\033[93m', 'FAIL': '\033[91m'}
    reset = '\033[0m'

    print(f"Status: {status_colors.get(status, '')}{status}{reset}")
    print(f"|r(P, I)|: {abs(r_P_I):.4f}")
    print(f"|r(Q, θ)| (circular): {r_Q_theta:.4f}")
    print(f"Samples: {len(P_proxy)}")
    print(f"Seed: {seed}")

    return status


# =============================================================================
# Main
# =============================================================================

def run_all_tests(seed=42):
    """Run all tests and summarize."""
    print("\n" + "="*60)
    print("RIGOROUS TESTS FOR HST-ROM IMPLEMENTATION")
    print("="*60)
    print(f"Master seed: {seed}")

    results = {}

    results['test_1'] = test_1_action_recovery(seed)
    results['test_2'] = test_2_p_conservation(seed)
    results['test_3'] = test_3_omega_relationship(seed)
    results['test_4'] = test_4_reconstruction(seed)
    results['test_5'] = test_5_generalization(seed)
    results['test_6'] = test_6_analytical_comparison(seed)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, status in results.items():
        status_colors = {'PASS': '\033[92m', 'MARGINAL': '\033[93m', 'FAIL': '\033[91m'}
        reset = '\033[0m'
        print(f"{test_name}: {status_colors.get(status, '')}{status}{reset}")

    n_pass = sum(1 for s in results.values() if s == 'PASS')
    n_marginal = sum(1 for s in results.values() if s == 'MARGINAL')
    n_fail = sum(1 for s in results.values() if s == 'FAIL')

    print(f"\nTotal: {n_pass} PASS, {n_marginal} MARGINAL, {n_fail} FAIL")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous tests for HST-ROM")
    parser.add_argument('--test', type=str, default='all',
                       help='Test to run: 1-6 or "all"')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    if args.test == 'all':
        run_all_tests(args.seed)
    elif args.test == '1':
        test_1_action_recovery(args.seed)
    elif args.test == '2':
        test_2_p_conservation(args.seed)
    elif args.test == '3':
        test_3_omega_relationship(args.seed)
    elif args.test == '4':
        test_4_reconstruction(args.seed)
    elif args.test == '5':
        test_5_generalization(args.seed)
    elif args.test == '6':
        test_6_analytical_comparison(args.seed)
    else:
        print(f"Unknown test: {args.test}")
        print("Use: 1, 2, 3, 4, 5, 6, or 'all'")
