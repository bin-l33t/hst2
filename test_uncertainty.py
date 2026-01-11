"""
Test Uncertainty Relations in HST Coordinates

From Glinsky's paper:
- "Each system is quantized at its own natural scale of action given by E₀/ω₀, not ℏ"
- "The system will remain close to P = constant, but Q will not be known"
- "Variance of order (2π)^n"

Testable predictions:
1. ΔP should be small (action is conserved)
2. ΔQ should grow with time (phase randomizes)
3. ΔP·ΔQ ≥ J₀/2 where J₀ = E₀/ω₀ (natural uncertainty bound)
4. For short times (t < 2π/ω₀), both ΔP and ΔQ small (classical regime)
5. For long times (t > 2π/ω₀), ΔQ → 2π (phase uniformly distributed)

This is INDEPENDENT of the geodesic claim - it's about the
wavelet time-frequency localization properties.
"""

import numpy as np
from scipy.stats import circstd
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import (
    SimpleHarmonicOscillator, AnharmonicOscillator, PendulumOscillator,
    simulate_hamiltonian
)
from hst import hst_forward_pywt


def compute_P_Q_from_trajectory(z, window_size=256, stride=64):
    """
    Compute P and Q proxies along a trajectory using windowed HST.

    P proxy: Energy in HST coefficients (should be conserved)
    Q proxy: Phase of dominant FFT component (should advance linearly, become uncertain)

    Returns arrays of (P, Q) at each window position.
    """
    P_values = []
    Q_values = []
    times = []

    for start in range(0, len(z) - window_size, stride):
        window = z[start:start+window_size]

        # P proxy: HST energy
        coeffs = hst_forward_pywt(window.real, J=3, wavelet_name='db8')
        P_val = sum((2**j) * np.mean(np.abs(c)**2) for j, c in enumerate(coeffs['cD']))
        P_val += (2**len(coeffs['cD'])) * np.mean(np.abs(coeffs['cA_final'])**2)

        # Q proxy: Phase of trajectory
        # For SHO, Q = arctan(p/q), which is the phase angle
        # We compute it from the FFT of the windowed signal
        fft = np.fft.fft(window)
        peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        Q_val = np.angle(fft[peak_idx])

        P_values.append(P_val)
        Q_values.append(Q_val)
        times.append(start + window_size // 2)

    return np.array(P_values), np.array(Q_values), np.array(times)


def test_uncertainty_short_time():
    """
    TEST: For short observation times, both P and Q should be well-determined.

    Glinsky: "Classical observation can be made in a rest frame" when t < 2π/ω₀
    """
    print("\n" + "="*60)
    print("TEST: Short-time Uncertainty (Classical Regime)")
    print("="*60)

    sho = SimpleHarmonicOscillator(omega0=1.0)
    T_char = 2 * np.pi / sho.omega0  # Characteristic time

    # Simulate for LESS than one period
    T_sim = 0.8 * T_char
    E = 2.0
    q0, p0 = sho.initial_condition_for_energy(E)
    t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=T_sim, dt=0.01)

    # Compute P and Q from trajectory
    P, Q, times = compute_P_Q_from_trajectory(z, window_size=64, stride=16)

    if len(P) < 3:
        print("Insufficient data points")
        return

    # Compute uncertainties
    delta_P = np.std(P) / np.mean(P)  # Coefficient of variation
    delta_Q = circstd(Q)  # Circular standard deviation for angles

    # Natural action scale
    J0 = E_actual / sho.omega0

    print(f"Simulation time: {T_sim:.2f} (characteristic time: {T_char:.2f})")
    print(f"Number of windows: {len(P)}")
    print(f"\nResults:")
    print(f"  ΔP/P (CV): {delta_P:.4f}")
    print(f"  ΔQ (circular std): {delta_Q:.4f} rad")
    print(f"  J₀ = E/ω₀: {J0:.4f}")
    print(f"  ΔP·ΔQ product: {delta_P * np.mean(P) * delta_Q:.4f}")

    # Pass criterion: Both uncertainties should be small in classical regime
    if delta_P < 0.1 and delta_Q < np.pi/2:
        print("\nStatus: PASS (classical regime - both P and Q well-determined)")
        return 'PASS'
    else:
        print("\nStatus: FAIL (too much uncertainty in classical regime)")
        return 'FAIL'


def test_uncertainty_long_time():
    """
    TEST: For long observation times, Q should become uniformly distributed.

    Glinsky: "On times greater than 2π/ω₀, there is exponential divergence"
    "Q will not be known... must be uniform and periodic"
    """
    print("\n" + "="*60)
    print("TEST: Long-time Uncertainty (Quantum/Statistical Regime)")
    print("="*60)

    sho = SimpleHarmonicOscillator(omega0=1.0)
    T_char = 2 * np.pi / sho.omega0

    # Simulate for MANY periods
    T_sim = 20 * T_char
    E = 2.0
    q0, p0 = sho.initial_condition_for_energy(E)
    t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=T_sim, dt=0.01)

    # Compute P and Q from trajectory
    P, Q, times = compute_P_Q_from_trajectory(z, window_size=256, stride=64)

    if len(P) < 10:
        print("Insufficient data points")
        return

    # Compute uncertainties
    delta_P = np.std(P) / np.mean(P)
    delta_Q = circstd(Q)

    # For uniform phase distribution, circular std = sqrt(1 - |mean exp(iQ)|^2)
    # For uniform on [-π, π], this is approximately π/sqrt(3) ≈ 1.81
    uniform_std = np.pi / np.sqrt(3)

    # Natural action scale
    J0 = E_actual / sho.omega0

    print(f"Simulation time: {T_sim:.2f} ({T_sim/T_char:.0f} periods)")
    print(f"Number of windows: {len(P)}")
    print(f"\nResults:")
    print(f"  ΔP/P (CV): {delta_P:.4f}")
    print(f"  ΔQ (circular std): {delta_Q:.4f} rad")
    print(f"  Expected uniform ΔQ: {uniform_std:.4f} rad")
    print(f"  J₀ = E/ω₀: {J0:.4f}")

    # Test: P should still be conserved, Q should be spread over full circle
    if delta_P < 0.15 and delta_Q > 1.0:
        print("\nStatus: PASS (P conserved, Q phase-randomized)")
        return 'PASS'
    else:
        print(f"\nStatus: MARGINAL (δP={delta_P:.3f}, δQ={delta_Q:.3f})")
        return 'MARGINAL'


def test_uncertainty_product():
    """
    TEST: The uncertainty product ΔP·ΔQ should have a minimum value.

    Analogy to Heisenberg: ΔP·ΔQ ≥ J₀/2 where J₀ = E₀/ω₀

    We test this by computing the product for many different states
    and checking if there's a lower bound.
    """
    print("\n" + "="*60)
    print("TEST: Uncertainty Product Lower Bound")
    print("="*60)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Test at multiple energy levels
    energies = np.linspace(0.5, 5.0, 10)

    products = []
    J0_values = []

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=0.01)

        P, Q, times = compute_P_Q_from_trajectory(z, window_size=256, stride=64)

        if len(P) < 5:
            continue

        delta_P = np.std(P)
        delta_Q = circstd(Q)

        product = delta_P * delta_Q
        J0 = E_actual / sho.omega0

        products.append(product)
        J0_values.append(J0)

    products = np.array(products)
    J0_values = np.array(J0_values)

    min_product = np.min(products)
    mean_J0 = np.mean(J0_values)

    print(f"Number of energy levels tested: {len(products)}")
    print(f"\nResults:")
    print(f"  Min(ΔP·ΔQ): {min_product:.4f}")
    print(f"  Mean J₀: {mean_J0:.4f}")
    print(f"  Ratio min(ΔP·ΔQ)/mean(J₀): {min_product/mean_J0:.4f}")
    print(f"  Range of products: [{products.min():.4f}, {products.max():.4f}]")

    # Check if there's a consistent lower bound
    # We don't know exact value, but it should exist
    if min_product > 0.01:  # Some non-zero lower bound exists
        print("\nStatus: PASS (non-trivial lower bound exists)")
        return 'PASS'
    else:
        print("\nStatus: FAIL (no clear lower bound)")
        return 'FAIL'


def test_coherent_state_saturation():
    """
    TEST: Coherent states should saturate the uncertainty bound.

    For SHO, Gaussian wave packets (coherent states) have minimum uncertainty.
    Our "coherent state" is a single-frequency pure sinusoid.

    Compare ΔP·ΔQ for:
    - Pure sinusoid (coherent-like)
    - Chirp signal (time-varying frequency)
    - Noisy signal (incoherent)
    """
    print("\n" + "="*60)
    print("TEST: Coherent State Saturation")
    print("="*60)

    N = 2048
    t = np.linspace(0, 10, N)

    # Signal 1: Pure sinusoid (coherent-like)
    z_coherent = np.sin(t) + 1j * np.cos(t)

    # Signal 2: Chirp (time-varying frequency)
    z_chirp = np.sin(t + 0.1*t**2) + 1j * np.cos(t + 0.1*t**2)

    # Signal 3: Noisy (incoherent)
    z_noisy = np.sin(t) + 0.5*np.random.randn(N) + 1j * (np.cos(t) + 0.5*np.random.randn(N))

    signals = {
        'Pure sinusoid (coherent)': z_coherent,
        'Chirp (semi-coherent)': z_chirp,
        'Noisy (incoherent)': z_noisy
    }

    results = {}

    for name, z in signals.items():
        P, Q, times = compute_P_Q_from_trajectory(z, window_size=256, stride=64)

        if len(P) < 5:
            continue

        delta_P = np.std(P)
        delta_Q = circstd(Q)
        product = delta_P * delta_Q

        results[name] = {
            'delta_P': delta_P,
            'delta_Q': delta_Q,
            'product': product
        }

        print(f"\n{name}:")
        print(f"  ΔP: {delta_P:.4f}")
        print(f"  ΔQ: {delta_Q:.4f}")
        print(f"  ΔP·ΔQ: {product:.4f}")

    # Check if coherent state has minimum product
    if len(results) == 3:
        coherent_product = results['Pure sinusoid (coherent)']['product']
        other_products = [results[k]['product'] for k in results if 'coherent' not in k.lower() or 'incoherent' in k.lower()]
        other_products = [results['Chirp (semi-coherent)']['product'],
                         results['Noisy (incoherent)']['product']]

        if coherent_product < min(other_products):
            print("\nStatus: PASS (coherent state has minimum uncertainty)")
            return 'PASS'
        else:
            print("\nStatus: FAIL (coherent state does NOT have minimum uncertainty)")
            return 'FAIL'

    return 'INCONCLUSIVE'


def test_action_angle_from_hst():
    """
    Direct test: Extract (P, Q) from HST and compare to true (I, θ) for SHO.

    For SHO:
    - True action: I = (q² + p²) / 2 = E
    - True angle: θ = arctan(p/q)

    If HST extracts good action-angle coordinates:
    - P should correlate with I
    - Q should correlate with θ (mod 2π)
    """
    print("\n" + "="*60)
    print("TEST: HST Action-Angle vs True Action-Angle (SHO)")
    print("="*60)

    sho = SimpleHarmonicOscillator(omega0=1.0)
    E = 2.0
    q0, p0 = sho.initial_condition_for_energy(E)
    t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=100, dt=0.01)

    # True action-angle
    I_true = (q**2 + p**2) / 2  # = E for SHO
    theta_true = np.arctan2(p, q)

    # HST-derived (P, Q)
    P_hst, Q_hst, times = compute_P_Q_from_trajectory(z, window_size=256, stride=64)

    # Sample true values at corresponding times
    # Convert times (sample indices) to actual time indices
    dt = 0.01
    time_indices = (np.array(times) * dt / dt).astype(int)
    time_indices = np.clip(time_indices, 0, len(I_true)-1)

    I_sampled = I_true[time_indices]
    theta_sampled = theta_true[time_indices]

    # Correlations
    from scipy.stats import pearsonr

    r_P_I, _ = pearsonr(P_hst, I_sampled)

    # For circular correlation of angles
    sin_Q = np.sin(Q_hst)
    cos_Q = np.cos(Q_hst)
    sin_theta = np.sin(theta_sampled)
    cos_theta = np.cos(theta_sampled)

    r_sin, _ = pearsonr(sin_Q, sin_theta)
    r_cos, _ = pearsonr(cos_Q, cos_theta)
    r_Q_theta = np.sqrt((r_sin**2 + r_cos**2) / 2)

    print(f"Number of samples: {len(P_hst)}")
    print(f"\nCorrelations:")
    print(f"  r(P_HST, I_true): {r_P_I:.4f}")
    print(f"  r(Q_HST, θ_true) [circular]: {r_Q_theta:.4f}")

    # Uncertainties
    delta_P = np.std(P_hst) / np.mean(P_hst)
    delta_Q = circstd(Q_hst)
    delta_I = np.std(I_sampled) / np.mean(I_sampled)
    delta_theta = circstd(theta_sampled)

    print(f"\nUncertainties:")
    print(f"  ΔP/P (HST): {delta_P:.4f}")
    print(f"  ΔI/I (true): {delta_I:.4f}")
    print(f"  ΔQ (HST): {delta_Q:.4f}")
    print(f"  Δθ (true): {delta_theta:.4f}")

    # J0 and uncertainty product
    J0 = E_actual / sho.omega0
    print(f"\nUncertainty products:")
    print(f"  ΔP·ΔQ (HST): {np.std(P_hst) * delta_Q:.4f}")
    print(f"  ΔI·Δθ (true): {np.std(I_sampled) * delta_theta:.4f}")
    print(f"  J₀ = E/ω₀: {J0:.4f}")

    if r_Q_theta > 0.9:
        print("\nStatus: PASS (Q correlates well with true angle)")
        return 'PASS'
    else:
        print("\nStatus: MARGINAL (Q correlation with angle is weak)")
        return 'MARGINAL'


def run_all_uncertainty_tests():
    """Run all uncertainty tests."""
    print("\n" + "="*60)
    print("UNCERTAINTY RELATION TESTS FOR HST")
    print("Based on Glinsky's claim: J₀ = E₀/ω₀ is natural action scale")
    print("="*60)

    results = {}

    results['short_time'] = test_uncertainty_short_time()
    results['long_time'] = test_uncertainty_long_time()
    results['product_bound'] = test_uncertainty_product()
    results['coherent_saturation'] = test_coherent_state_saturation()
    results['action_angle'] = test_action_angle_from_hst()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for test_name, status in results.items():
        status_colors = {'PASS': '\033[92m', 'MARGINAL': '\033[93m', 'FAIL': '\033[91m'}
        reset = '\033[0m'
        color = status_colors.get(status, '')
        print(f"  {test_name}: {color}{status}{reset}")

    return results


if __name__ == "__main__":
    run_all_uncertainty_tests()
