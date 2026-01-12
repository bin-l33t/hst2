"""
Test: Quantization from Coarse-Graining (Glinsky's Actual Claim)

Glinsky's claim is NOT about discrete eigenvalues, but about
observation timescale:
  - Fine scale (Δτ < T): Both P and Q observable
  - Coarse scale (Δτ >> T): Only P observable, Q uniformizes

This tests whether HST wavelet scale acts as effective Δτ,
and whether "quantization" emerges from coarse-graining.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, entropy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from hst import hst_forward_pywt, extract_features
from hamiltonian_systems import SimpleHarmonicOscillator, simulate_hamiltonian


def extract_features_at_scale(z, J_max, wavelet_name='db8'):
    """
    Extract features using only wavelet levels up to J_max.

    Higher J = coarser scale = longer effective Δτ
    """
    coeffs_real = hst_forward_pywt(z.real, J=J_max, wavelet_name=wavelet_name)
    coeffs_imag = hst_forward_pywt(z.imag, J=J_max, wavelet_name=wavelet_name)

    features = []

    # Only use the coarsest level (level J_max)
    # This corresponds to effective Δτ ~ 2^J_max / fs
    cD_real = coeffs_real['cD'][-1]  # Coarsest detail
    cD_imag = coeffs_imag['cD'][-1]
    cD_complex = cD_real + 1j * cD_imag

    # Magnitude features (P information)
    features.extend([
        np.mean(np.abs(cD_complex)),
        np.std(np.abs(cD_complex)),
    ])

    # Phase features (Q information)
    phases = np.angle(cD_complex)
    features.extend([
        np.mean(np.cos(phases)),
        np.mean(np.sin(phases)),
        np.std(phases),
    ])

    # Approximation coefficients (most coarse)
    cA_real = coeffs_real['cA_final']
    cA_imag = coeffs_imag['cA_final']
    cA_complex = cA_real + 1j * cA_imag

    features.extend([
        np.mean(np.abs(cA_complex)),
        np.std(np.abs(cA_complex)),
        np.mean(np.cos(np.angle(cA_complex))),
        np.mean(np.sin(np.angle(cA_complex))),
    ])

    return np.array(features)


def mutual_information_discrete(x, y, n_bins=20):
    """Estimate mutual information I(X; Y) using binning."""
    # Joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = hist_xy / hist_xy.sum()

    # Marginals
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # MI = H(X) + H(Y) - H(X,Y)
    H_x = entropy(p_x + 1e-10)
    H_y = entropy(p_y + 1e-10)
    H_xy = entropy(p_xy.flatten() + 1e-10)

    return H_x + H_y - H_xy


def test_scale_dependent_information():
    """
    Test: Does Q information decrease at coarser wavelet scales?

    At fine scales (small J): Should recover both P and Q
    At coarse scales (large J): Should only recover P
    """
    print("=" * 70)
    print("TEST: Scale-Dependent Information (Glinsky Coarse-Graining)")
    print("=" * 70)

    np.random.seed(42)

    omega0 = 1.0
    sho = SimpleHarmonicOscillator(omega0=omega0)
    T_period = 2 * np.pi / omega0
    dt = 0.01
    fs = 1.0 / dt

    print(f"\nSystem: SHO with ω₀ = {omega0}")
    print(f"Period T = {T_period:.2f} s")
    print(f"Sampling: dt = {dt} s, fs = {fs} Hz")

    # Generate ensemble of trajectories
    n_trajectories = 100
    window_size = 1024  # Longer window to allow more scales

    # Spread in initial conditions
    energies = np.random.uniform(0.5, 3.0, n_trajectories)
    initial_phases = np.random.uniform(0, 2*np.pi, n_trajectories)

    print(f"\nGenerating {n_trajectories} trajectories...")
    print(f"Window size: {window_size} samples = {window_size * dt:.1f} s")
    print(f"Window contains {window_size * dt / T_period:.1f} periods")

    # Test different wavelet scales
    J_values = [1, 2, 3, 4, 5]  # Different coarseness levels

    results = {J: {'features': [], 'P': [], 'Q': []} for J in J_values}

    for i, (E, phi0) in enumerate(zip(energies, initial_phases)):
        # Initial condition with specified phase
        amplitude = np.sqrt(2 * E / omega0**2)
        q0 = amplitude * np.cos(phi0)
        p0 = -amplitude * omega0 * np.sin(phi0)

        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=dt)

        # True action and angle at window center
        P_true = E_actual / omega0

        # Extract features at different scales
        for start in range(0, len(z) - window_size, window_size // 2):
            window = z[start:start + window_size]

            # True Q at center of window
            center = start + window_size // 2
            Q_true = np.arctan2(p[center], q[center])

            for J in J_values:
                try:
                    feat = extract_features_at_scale(window, J_max=J)
                    results[J]['features'].append(feat)
                    results[J]['P'].append(P_true)
                    results[J]['Q'].append(Q_true)
                except:
                    pass

    # Analyze information content at each scale
    print("\n" + "=" * 70)
    print("INFORMATION CONTENT AT DIFFERENT SCALES")
    print("=" * 70)

    # Effective Δτ at each scale
    print(f"\n{'J':<5} {'Δτ_eff (s)':<12} {'Δτ/T':<10} {'r(P)':<10} {'r(sin Q)':<12} {'r(cos Q)':<12} {'MI(feat;Q)':<12}")
    print("-" * 85)

    scale_results = []

    for J in J_values:
        X = np.array(results[J]['features'])
        P = np.array(results[J]['P'])
        Q = np.array(results[J]['Q'])

        if len(X) < 50:
            continue

        # Effective timescale at level J
        delta_tau = 2**J / fs
        delta_tau_over_T = delta_tau / T_period

        # Normalize features
        X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
        P_norm = (P - P.mean()) / P.std()

        # Train simple linear models
        # For P
        from sklearn.linear_model import Ridge
        reg_P = Ridge(alpha=0.1)
        reg_P.fit(X_norm, P_norm)
        P_pred = reg_P.predict(X_norm)
        r_P, _ = pearsonr(P_pred, P_norm)

        # For sin(Q) and cos(Q)
        sin_Q = np.sin(Q)
        cos_Q = np.cos(Q)

        reg_sin = Ridge(alpha=0.1)
        reg_sin.fit(X_norm, sin_Q)
        sin_pred = reg_sin.predict(X_norm)
        r_sin, _ = pearsonr(sin_pred, sin_Q)

        reg_cos = Ridge(alpha=0.1)
        reg_cos.fit(X_norm, cos_Q)
        cos_pred = reg_cos.predict(X_norm)
        r_cos, _ = pearsonr(cos_pred, cos_Q)

        # Mutual information between features and Q
        # Use first 2 PCA components of features
        from numpy.linalg import svd
        U, S, Vt = svd(X_norm, full_matrices=False)
        feat_pca = U[:, :2] * S[:2]
        mi_Q = mutual_information_discrete(feat_pca[:, 0], Q, n_bins=15)

        print(f"{J:<5} {delta_tau:<12.4f} {delta_tau_over_T:<10.2f} {r_P:<10.4f} {r_sin:<12.4f} {r_cos:<12.4f} {mi_Q:<12.4f}")

        scale_results.append({
            'J': J,
            'delta_tau': delta_tau,
            'delta_tau_over_T': delta_tau_over_T,
            'r_P': r_P,
            'r_sin_Q': r_sin,
            'r_cos_Q': r_cos,
            'MI_Q': mi_Q
        })

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    J_arr = [r['J'] for r in scale_results]
    tau_arr = [r['delta_tau_over_T'] for r in scale_results]
    r_P_arr = [r['r_P'] for r in scale_results]
    r_sin_arr = [r['r_sin_Q'] for r in scale_results]
    r_cos_arr = [r['r_cos_Q'] for r in scale_results]
    mi_arr = [r['MI_Q'] for r in scale_results]

    # Plot 1: P and Q correlation vs scale
    ax1 = axes[0, 0]
    ax1.plot(tau_arr, r_P_arr, 'bo-', markersize=10, linewidth=2, label='r(P)')
    ax1.plot(tau_arr, r_sin_arr, 'r^-', markersize=10, linewidth=2, label='r(sin Q)')
    ax1.plot(tau_arr, r_cos_arr, 'gs-', markersize=10, linewidth=2, label='r(cos Q)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=1.0, color='gray', linestyle=':', label='Δτ = T')
    ax1.set_xlabel('Δτ / T (wavelet scale / period)')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Information Content vs Observation Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Mutual information vs scale
    ax2 = axes[0, 1]
    ax2.plot(tau_arr, mi_arr, 'mo-', markersize=10, linewidth=2)
    ax2.axvline(x=1.0, color='gray', linestyle=':', label='Δτ = T')
    ax2.set_xlabel('Δτ / T (wavelet scale / period)')
    ax2.set_ylabel('Mutual Information I(features; Q)')
    ax2.set_title('Q Information vs Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Plot 3: P vs Q correlation ratio
    ax3 = axes[1, 0]
    r_Q_avg = [(s + c) / 2 for s, c in zip(r_sin_arr, r_cos_arr)]
    ratio = [p / (q + 0.01) for p, q in zip(r_P_arr, r_Q_avg)]
    ax3.plot(tau_arr, ratio, 'ko-', markersize=10, linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='P = Q information')
    ax3.axvline(x=1.0, color='gray', linestyle=':', label='Δτ = T')
    ax3.set_xlabel('Δτ / T')
    ax3.set_ylabel('r(P) / r(Q)')
    ax3.set_title('Action/Angle Information Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Determine if coarse-graining effect is present
    if len(scale_results) >= 2:
        fine = scale_results[0]
        coarse = scale_results[-1]

        P_change = coarse['r_P'] - fine['r_P']
        Q_change = (coarse['r_sin_Q'] + coarse['r_cos_Q']) / 2 - (fine['r_sin_Q'] + fine['r_cos_Q']) / 2

        summary = f"""
COARSE-GRAINING ANALYSIS

Fine scale (J={fine['J']}, Δτ/T = {fine['delta_tau_over_T']:.2f}):
  r(P) = {fine['r_P']:.4f}
  r(Q) = {(fine['r_sin_Q'] + fine['r_cos_Q'])/2:.4f}

Coarse scale (J={coarse['J']}, Δτ/T = {coarse['delta_tau_over_T']:.2f}):
  r(P) = {coarse['r_P']:.4f}
  r(Q) = {(coarse['r_sin_Q'] + coarse['r_cos_Q'])/2:.4f}

Changes:
  Δr(P) = {P_change:+.4f}
  Δr(Q) = {Q_change:+.4f}

Glinsky prediction:
  At Δτ >> T, Q information lost, P preserved.

Result: {'CONFIRMED' if Q_change < -0.1 and P_change > -0.1 else 'NOT CLEAR'}
        """
    else:
        summary = "Insufficient data for analysis"

    ax4.text(0.1, 0.5, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('/home/ubuntu/rectifier/coarse_graining_test.png', dpi=150)
    print("\nSaved: coarse_graining_test.png")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if len(scale_results) >= 2:
        fine = scale_results[0]
        coarse = scale_results[-1]

        P_preserved = coarse['r_P'] > 0.8
        Q_lost = (coarse['r_sin_Q'] + coarse['r_cos_Q']) / 2 < 0.3

        if P_preserved and Q_lost:
            print("\n\033[92mGLINSKY COARSE-GRAINING CONFIRMED!\033[0m")
            print("  At coarse scales (Δτ >> T):")
            print(f"    - P information preserved (r = {coarse['r_P']:.3f})")
            print(f"    - Q information lost (r = {(coarse['r_sin_Q'] + coarse['r_cos_Q'])/2:.3f})")
            print("  This is the 'quantization' effect: only action P is observable")
            print("  at coarse timescales, phase Q uniformizes.")
        elif Q_lost and not P_preserved:
            print("\n\033[93mBOTH P AND Q LOST at coarse scale\033[0m")
            print("  Information loss is not selective.")
        else:
            print("\n\033[91mNO CLEAR COARSE-GRAINING EFFECT\033[0m")
            print(f"  P preserved: {P_preserved}")
            print(f"  Q lost: {Q_lost}")

    return scale_results


if __name__ == "__main__":
    # Need sklearn for Ridge
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        print("Installing sklearn...")
        import subprocess
        subprocess.run(['pip', 'install', 'scikit-learn', '-q'])
        from sklearn.linear_model import Ridge

    results = test_scale_dependent_information()
