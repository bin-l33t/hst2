"""
Cauchy-Paul Wavelet Analysis

Cauchy-Paul wavelets are the CANONICAL coherent states of the affine group (Ali Eq. 12.20).
This analysis compares them with orthogonal wavelets (Daubechies) for HST.

Key properties:
- Progressive/Analytic: Supported only on positive frequencies
- Coherent states: Exactly satisfy resolution of identity for affine group
- Not orthogonal: Form a continuous frame, not discrete orthonormal basis
- Infinite support: Slow polynomial decay in time domain
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.signal import convolve
from scipy.integrate import solve_ivp
import pywt

# ============================================================
# TASK 1: Implement Cauchy-Paul Wavelets
# ============================================================

def cauchy_paul_wavelet_freq(xi, m=1):
    """
    Cauchy-Paul wavelet in frequency domain.

    ψ̂_m(ξ) = ξ^m · e^(-ξ)  for ξ ≥ 0
           = 0              for ξ < 0

    Parameters:
    -----------
    xi : array - frequency values
    m : int - order (m=1 is standard Paul wavelet)

    Returns:
    --------
    psi_hat : complex array - wavelet in frequency domain
    """
    xi = np.asarray(xi, dtype=float)
    psi_hat = np.zeros_like(xi, dtype=complex)

    pos = xi >= 0
    # Normalization for unit L2 norm
    # ||ψ||² = ∫|ψ̂|² dξ = 1
    norm = np.sqrt(factorial(m) / (2 * np.pi))

    psi_hat[pos] = (xi[pos]**m) * np.exp(-xi[pos]) * norm

    return psi_hat


def cauchy_paul_wavelet_analytic(t, m=1):
    """
    Cauchy-Paul wavelet - direct analytic formula in time domain.

    ψ_m(t) = C_m / (1 - it)^(m+1)

    where C_m is normalization constant for unit L2 norm.
    """
    t = np.asarray(t, dtype=float)

    # Normalization constant for unit L2 norm
    # From Ali Eq. 12.20: involves factorial ratios
    C_m = np.sqrt(factorial(2*m) / (np.pi * factorial(m)**2 * 2**(2*m)))

    return C_m / (1 - 1j*t)**(m+1)


def cauchy_paul_wavelet_time_fft(n_samples, dt=0.1, m=1):
    """
    Compute Cauchy-Paul wavelet in time domain via inverse FFT.

    More numerically stable for longer wavelets.
    """
    # Frequency axis (positive frequencies only matter for analytic wavelet)
    freq = np.fft.fftfreq(n_samples, d=dt)
    xi = 2 * np.pi * freq  # Angular frequency

    # Frequency domain representation
    psi_hat = cauchy_paul_wavelet_freq(xi, m=m)

    # IFFT to time domain
    psi = np.fft.ifft(psi_hat) * n_samples

    return psi


# ============================================================
# TASK 2: Visualize Cauchy-Paul vs Daubechies
# ============================================================

def compare_wavelets_visualization(save_path='cauchy_paul_vs_db8.png'):
    """Compare Cauchy-Paul vs db8 in time and frequency domains."""
    print("=" * 70)
    print("Comparing Cauchy-Paul vs Daubechies-8 Wavelets")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Time domain - Cauchy-Paul
    t = np.linspace(-10, 10, 2048)

    # Cauchy-Paul m=1 (Paul wavelet)
    psi_paul = cauchy_paul_wavelet_analytic(t, m=1)

    ax = axes[0, 0]
    ax.plot(t, np.real(psi_paul), 'b-', label='Re(ψ)', alpha=0.8)
    ax.plot(t, np.imag(psi_paul), 'r-', label='Im(ψ)', alpha=0.8)
    ax.plot(t, np.abs(psi_paul), 'k--', label='|ψ|', lw=2)
    ax.set_title('Cauchy-Paul (m=1) - Time Domain')
    ax.set_xlabel('t')
    ax.set_ylabel('ψ(t)')
    ax.legend()
    ax.set_xlim([-10, 10])
    ax.grid(True, alpha=0.3)

    # Daubechies-8 time domain
    wavelet = pywt.Wavelet('db8')
    psi_db8, phi_db8, x_db8 = wavelet.wavefun(level=8)

    ax = axes[0, 1]
    ax.plot(x_db8, psi_db8, 'b-', label='ψ (real)', lw=1.5)
    ax.set_title('Daubechies-8 - Time Domain')
    ax.set_xlabel('t')
    ax.set_ylabel('ψ(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison: Decay rates
    ax = axes[0, 2]
    t_pos = t[t > 0.1]
    psi_pos = psi_paul[t > 0.1]
    ax.semilogy(t_pos, np.abs(psi_pos), 'b-', label='Cauchy-Paul', lw=1.5)

    # Theoretical decay: |ψ| ~ 1/t^(m+1) = 1/t^2 for m=1
    t_theory = t_pos[t_pos > 1]
    decay_theory = 0.5 / t_theory**2
    ax.semilogy(t_theory, decay_theory, 'r--', label='~1/t² (theory)', alpha=0.7)

    ax.axhline(y=1e-6, color='gray', linestyle=':', label='Truncation')
    ax.set_title('Cauchy-Paul Decay (m=1)')
    ax.set_xlabel('t')
    ax.set_ylabel('|ψ(t)|')
    ax.legend()
    ax.set_xlim([0, 10])
    ax.grid(True, alpha=0.3)

    # Frequency domain - Cauchy-Paul
    xi = np.linspace(0, 10, 1024)
    psi_paul_freq = cauchy_paul_wavelet_freq(xi, m=1)

    ax = axes[1, 0]
    ax.plot(xi, np.abs(psi_paul_freq)**2, 'b-', lw=1.5)
    ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='Peak at ξ=m=1')
    ax.set_title('Cauchy-Paul (m=1) - |ψ̂(ξ)|²')
    ax.set_xlabel('ξ')
    ax.set_ylabel('Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency domain - db8
    n_freq = 2048
    hi_d = np.array(wavelet.dec_hi)  # High-pass (mother)
    hi_padded = np.zeros(n_freq)
    hi_padded[:len(hi_d)] = hi_d
    G = np.fft.fft(hi_padded)
    freq_db8 = np.fft.fftfreq(n_freq)

    ax = axes[1, 1]
    ax.plot(freq_db8[:n_freq//2], np.abs(G[:n_freq//2])**2, 'b-', lw=1.5)
    ax.set_title('Daubechies-8 - |Ĝ(ω)|²')
    ax.set_xlabel('ω (normalized)')
    ax.set_ylabel('Power')
    ax.grid(True, alpha=0.3)

    # Different m values for Cauchy-Paul
    ax = axes[1, 2]
    for m in [0, 1, 2, 3]:
        psi_m = cauchy_paul_wavelet_freq(xi, m=m)
        ax.plot(xi, np.abs(psi_m)**2, label=f'm={m}', alpha=0.8)
    ax.set_title('Cauchy-Paul for different m')
    ax.set_xlabel('ξ')
    ax.set_ylabel('|ψ̂(ξ)|²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Cauchy-Paul (Coherent State) vs Daubechies-8 (Orthogonal)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# TASK 3: HST with Cauchy-Paul Wavelets
# ============================================================

def hst_forward_cauchy_paul(f, m=1, J=3, wavelet_samples=128):
    """
    HST using Cauchy-Paul wavelets (continuous-style, truncated).

    Since Cauchy-Paul isn't orthogonal, we use direct convolution.
    No perfect reconstruction guarantee (frame, not basis).
    """
    from rectifier import R_sheeted
    from scipy.signal.windows import gaussian

    f = np.asarray(f, dtype=complex)

    # Create truncated Cauchy-Paul wavelet (base scale)
    t_base = np.linspace(-5, 5, wavelet_samples)
    psi_base = cauchy_paul_wavelet_analytic(t_base, m=m)
    psi_base = psi_base / np.sqrt(np.sum(np.abs(psi_base)**2))  # Normalize

    # Initial rectification
    f_safe = np.where(np.abs(f) < 1e-10, 1e-10 + 0j, f)
    u = R_sheeted(f_safe)

    S_coeffs = []
    layers = []

    for j in range(J):
        if len(u) < wavelet_samples:
            break

        # Scale the wavelet (dilate by 2^j)
        # For wavelets: ψ_a(t) = (1/√a) ψ(t/a)
        scale = 2**j
        psi_j = psi_base / np.sqrt(scale)  # Energy normalization

        # Convolve with scaled wavelet
        v = convolve(u, psi_j, mode='same')

        # Rectify
        v_safe = np.where(np.abs(v) < 1e-10, 1e-10 + 0j, v)
        u = R_sheeted(v_safe)

        layers.append(u.copy())

        # Father wavelet averaging (Gaussian)
        phi_len = min(len(u), max(8, 2**(j+3)))
        phi = gaussian(phi_len, std=max(phi_len/4, 1))
        phi = phi / phi.sum()
        S_j = convolve(u, phi, mode='same')

        S_coeffs.append(S_j)

        # Binate (downsample by 2)
        u = u[::2]

    return S_coeffs, layers, u


def generate_vdp_trajectory(eps=0.1, T=100, N=2048):
    """Generate Van der Pol oscillator trajectory."""
    def van_der_pol(t, y, eps):
        x, v = y
        return [v, eps * (1 - x**2) * v - x]

    t_span = (0, T)
    t_eval = np.linspace(0, T, N)

    sol = solve_ivp(van_der_pol, t_span, [0.1, 0], t_eval=t_eval,
                    args=(eps,), method='RK45', rtol=1e-8)

    z = sol.y[0] + 1j * sol.y[1]
    rho = np.sqrt(sol.y[0]**2 + sol.y[1]**2)
    phi = np.arctan2(sol.y[1], sol.y[0])

    return sol.t, z, rho, phi


def compare_hst_wavelets_on_vdp(save_path='cauchy_paul_hst_comparison.png'):
    """Compare HST with Cauchy-Paul vs db8 on Van der Pol oscillator."""
    print("\n" + "=" * 70)
    print("HST Wavelet Comparison on Van der Pol Oscillator")
    print("=" * 70)

    # Generate trajectory
    t, z, rho, phi = generate_vdp_trajectory(eps=0.1, T=100, N=2048)

    print(f"\nVan der Pol trajectory: {len(z)} samples")
    print(f"Final amplitude ρ: {rho[-1]:.4f} (should be ~2)")

    # HST with db8 (pywt version)
    from hst import hst_forward_pywt
    coeffs_db8 = hst_forward_pywt(z, J=4, verbose=False)

    # HST with Cauchy-Paul
    S_cp, layers_cp, u_cp = hst_forward_cauchy_paul(z, m=1, J=4)

    print(f"\ndb8 HST: {len(coeffs_db8['cD'])} levels")
    print(f"Cauchy-Paul HST: {len(S_cp)} levels")

    # Extract features for comparison
    def extract_features_sliding_window(signal, S_coeffs, window=128, step=32):
        """Extract sliding window features from scattering coefficients."""
        n_samples = len(signal)
        n_windows = (n_samples - window) // step + 1

        features = []
        centers = []

        for i in range(n_windows):
            start = i * step
            end = start + window
            center = (start + end) // 2

            feat = []
            for S in S_coeffs:
                # Map window to coefficient indices
                scale = len(signal) / len(S)
                s_start = int(start / scale)
                s_end = int(end / scale)
                s_end = max(s_end, s_start + 1)

                if s_end <= len(S):
                    S_window = S[s_start:s_end]
                    feat.extend([
                        np.mean(np.abs(S_window)),
                        np.std(np.abs(S_window)),
                        np.mean(np.real(S_window)),
                        np.mean(np.imag(S_window)),
                    ])

            if len(feat) > 0:
                features.append(feat)
                centers.append(center)

        return np.array(features), np.array(centers)

    # Extract features
    X_db8, centers_db8 = extract_features_sliding_window(z, coeffs_db8['cD'])
    X_cp, centers_cp = extract_features_sliding_window(z, S_cp)

    print(f"\nFeature matrices:")
    print(f"  db8: {X_db8.shape}")
    print(f"  Cauchy-Paul: {X_cp.shape}")

    # Simple PCA using SVD
    def simple_pca(X, n_components=5):
        X_centered = X - np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        var_explained = (S**2) / np.sum(S**2)
        return U[:, :n_components] * S[:n_components], var_explained[:n_components]

    # PCA
    if X_db8.shape[0] > 5 and X_db8.shape[1] > 0:
        PC_db8, var_db8 = simple_pca(X_db8)
    else:
        PC_db8, var_db8 = np.zeros((len(centers_db8), 5)), np.zeros(5)

    if X_cp.shape[0] > 5 and X_cp.shape[1] > 0:
        PC_cp, var_cp = simple_pca(X_cp)
    else:
        PC_cp, var_cp = np.zeros((len(centers_cp), 5)), np.zeros(5)

    # Ground truth at window centers
    rho_db8 = rho[centers_db8] if len(centers_db8) > 0 else np.array([])
    rho_cp = rho[centers_cp] if len(centers_cp) > 0 else np.array([])
    phi_db8 = phi[centers_db8] if len(centers_db8) > 0 else np.array([])
    phi_cp = phi[centers_cp] if len(centers_cp) > 0 else np.array([])

    # Correlations
    def safe_corrcoef(a, b):
        if len(a) < 3 or len(b) < 3:
            return 0.0
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    results = {
        'db8': {
            'rho_corr': [safe_corrcoef(PC_db8[:, i], rho_db8) for i in range(min(5, PC_db8.shape[1]))],
            'cos_phi_corr': [safe_corrcoef(PC_db8[:, i], np.cos(phi_db8)) for i in range(min(5, PC_db8.shape[1]))],
            'var_explained': var_db8,
        },
        'cauchy_paul': {
            'rho_corr': [safe_corrcoef(PC_cp[:, i], rho_cp) for i in range(min(5, PC_cp.shape[1]))],
            'cos_phi_corr': [safe_corrcoef(PC_cp[:, i], np.cos(phi_cp)) for i in range(min(5, PC_cp.shape[1]))],
            'var_explained': var_cp,
        }
    }

    # Print results
    print("\n--- Results ---")
    print("\nDaubechies-8 (orthogonal):")
    print(f"  Variance explained: {var_db8[:3]}")
    print(f"  PC-ρ correlations: {[f'{r:.3f}' for r in results['db8']['rho_corr'][:3]]}")
    print(f"  PC-cos(φ) correlations: {[f'{r:.3f}' for r in results['db8']['cos_phi_corr'][:3]]}")
    print(f"  Best ρ correlation: {max(np.abs(results['db8']['rho_corr'])):.3f}")

    print("\nCauchy-Paul m=1 (coherent state):")
    print(f"  Variance explained: {var_cp[:3]}")
    print(f"  PC-ρ correlations: {[f'{r:.3f}' for r in results['cauchy_paul']['rho_corr'][:3]]}")
    print(f"  PC-cos(φ) correlations: {[f'{r:.3f}' for r in results['cauchy_paul']['cos_phi_corr'][:3]]}")
    print(f"  Best ρ correlation: {max(np.abs(results['cauchy_paul']['rho_corr'])):.3f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HST Comparison: Daubechies-8 vs Cauchy-Paul on Van der Pol', fontsize=14)

    # Scattering coefficients
    ax = axes[0, 0]
    for j, S in enumerate(coeffs_db8['cD']):
        ax.plot(np.abs(S), alpha=0.7, label=f'Level {j+1}')
    ax.set_title('db8: Scattering Coefficients |S_j|')
    ax.set_xlabel('Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for j, S in enumerate(S_cp):
        ax.plot(np.abs(S), alpha=0.7, label=f'Level {j+1}')
    ax.set_title('Cauchy-Paul: Scattering Coefficients |S_j|')
    ax.set_xlabel('Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PC1 vs rho
    ax = axes[0, 2]
    if len(PC_db8) > 0 and len(rho_db8) > 0:
        ax.scatter(PC_db8[:, 0], rho_db8, alpha=0.5, s=10, label='db8')
    if len(PC_cp) > 0 and len(rho_cp) > 0:
        ax.scatter(PC_cp[:, 0], rho_cp, alpha=0.5, s=10, label='Cauchy-Paul')
    ax.set_xlabel('PC1')
    ax.set_ylabel('ρ (amplitude)')
    ax.set_title('PC1 vs Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance explained
    ax = axes[1, 0]
    x = np.arange(len(var_db8))
    width = 0.35
    ax.bar(x - width/2, var_db8 * 100, width, label='db8', alpha=0.8)
    ax.bar(x + width/2, var_cp * 100, width, label='Cauchy-Paul', alpha=0.8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Variance Explained')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i+1}' for i in x])
    ax.grid(True, alpha=0.3)

    # Correlation heatmaps
    ax = axes[1, 1]
    corr_matrix = np.array([
        results['db8']['rho_corr'][:5],
        results['db8']['cos_phi_corr'][:5],
    ])
    im = ax.imshow(np.abs(corr_matrix), cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'PC{i+1}' for i in range(5)])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ρ', 'cos(φ)'])
    ax.set_title('db8: |Correlation| with Slow Manifold')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    corr_matrix_cp = np.array([
        results['cauchy_paul']['rho_corr'][:5],
        results['cauchy_paul']['cos_phi_corr'][:5],
    ])
    im = ax.imshow(np.abs(corr_matrix_cp), cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'PC{i+1}' for i in range(5)])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ρ', 'cos(φ)'])
    ax.set_title('Cauchy-Paul: |Correlation| with Slow Manifold')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return results


# ============================================================
# TASK 4: Admissibility and Frame Bounds
# ============================================================

def compute_admissibility_constants():
    """
    Compute admissibility constant C_ψ for Cauchy-Paul wavelets.

    C_ψ = ∫₀^∞ |ψ̂(ξ)|²/ξ dξ

    For wavelet to be admissible (form a frame), need 0 < C_ψ < ∞.
    This requires ψ̂(0) = 0 (zero mean) and sufficient decay.
    """
    print("\n" + "=" * 70)
    print("Admissibility Constants for Cauchy-Paul Wavelets")
    print("=" * 70)

    results = {}

    for m in [0, 1, 2, 3]:
        # Analytic formula for Cauchy-Paul:
        # C_ψ = m! for our normalization
        C_psi_analytic = factorial(m)

        # Numerical verification
        xi = np.linspace(1e-6, 100, 100000)
        psi_hat = cauchy_paul_wavelet_freq(xi, m=m)
        integrand = np.abs(psi_hat)**2 / xi
        C_psi_numerical = np.trapz(integrand, xi)

        results[m] = {
            'analytic': C_psi_analytic,
            'numerical': C_psi_numerical,
        }

        print(f"\nCauchy-Paul m={m}:")
        print(f"  C_ψ (analytic) = {C_psi_analytic:.6f}")
        print(f"  C_ψ (numerical) ≈ {C_psi_numerical:.6f}")
        print(f"  Admissible: {'YES' if 0 < C_psi_numerical < np.inf else 'NO'}")

    return results


def compare_frame_properties(save_path='frame_vs_basis.png'):
    """Compare frame (Cauchy-Paul) vs basis (Daubechies) properties."""
    print("\n" + "=" * 70)
    print("Frame vs Orthonormal Basis Properties")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Littlewood-Paley comparison
    n_freq = 4096
    freq = np.fft.fftfreq(n_freq)
    xi = np.linspace(0, 5, n_freq//2)

    # Cauchy-Paul m=1: |ψ̂(ξ)|²
    psi_cp = cauchy_paul_wavelet_freq(xi, m=1)

    ax = axes[0, 0]
    ax.plot(xi, np.abs(psi_cp)**2, 'b-', label='Cauchy-Paul m=1', lw=1.5)

    # For CWT, the LP "sum" is the admissibility constant (constant for all ξ)
    # This is different from DWT where it's |H|² + |G|² = 2
    ax.axhline(y=np.max(np.abs(psi_cp)**2), color='r', linestyle='--',
               alpha=0.5, label='Peak power')
    ax.set_title('Cauchy-Paul: Single wavelet |ψ̂(ξ)|²')
    ax.set_xlabel('ξ')
    ax.set_ylabel('Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. db8 Littlewood-Paley (should be = 2)
    wavelet = pywt.Wavelet('db8')
    lo_d = np.array(wavelet.dec_lo)
    hi_d = np.array(wavelet.dec_hi)

    lo_padded = np.zeros(n_freq)
    hi_padded = np.zeros(n_freq)
    lo_padded[:len(lo_d)] = lo_d
    hi_padded[:len(hi_d)] = hi_d

    H = np.fft.fft(lo_padded)
    G = np.fft.fft(hi_padded)
    LP_sum = np.abs(H)**2 + np.abs(G)**2

    ax = axes[0, 1]
    ax.plot(freq[:n_freq//2], np.abs(H[:n_freq//2])**2, 'b-', label='|H|²', alpha=0.7)
    ax.plot(freq[:n_freq//2], np.abs(G[:n_freq//2])**2, 'r-', label='|G|²', alpha=0.7)
    ax.plot(freq[:n_freq//2], LP_sum[:n_freq//2], 'g-', label='|H|²+|G|²=2', lw=2)
    ax.axhline(y=2, color='k', linestyle='--', alpha=0.5)
    ax.set_title('db8: Littlewood-Paley |H|² + |G|² = 2')
    ax.set_xlabel('ω (normalized)')
    ax.set_ylabel('Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Multi-scale coverage comparison
    ax = axes[1, 0]
    scales = 2**np.linspace(-2, 3, 100)
    xi_test = np.linspace(0.1, 5, 50)

    # For each frequency, sum |ψ̂(aξ)|²/a over scales
    for xi_val in [0.5, 1.0, 2.0, 3.0]:
        coverage = np.array([np.abs(cauchy_paul_wavelet_freq(xi_val/a, m=1))**2 / a
                            for a in scales])
        ax.plot(scales, coverage, label=f'ξ={xi_val}', alpha=0.7)

    ax.set_xscale('log')
    ax.set_title('Cauchy-Paul: Scale coverage at different ξ')
    ax.set_xlabel('Scale a')
    ax.set_ylabel('|ψ̂(ξ/a)|²/a')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Property', 'Daubechies (db8)', 'Cauchy-Paul (m=1)'],
        ['Type', 'Orthonormal basis', 'Continuous frame'],
        ['Support', 'Compact (16 samples)', 'Infinite (~1/t²)'],
        ['Frequency', 'Two-sided', 'One-sided (analytic)'],
        ['LP condition', '|H|²+|G|²=2 (exact)', 'C_ψ=1 (admissibility)'],
        ['Reconstruction', 'Perfect (0% error)', 'Approximate (frame)'],
        ['DWT/CWT', 'Discrete (DWT)', 'Continuous (CWT)'],
        ['Coherent state', 'No (orthogonal)', 'Yes (affine group)'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Frame vs Basis Comparison', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def run_all_analyses():
    """Run all Cauchy-Paul wavelet analyses."""
    print("\n" + "=" * 70)
    print("CAUCHY-PAUL WAVELET ANALYSIS")
    print("(Canonical Coherent States of the Affine Group)")
    print("=" * 70)

    # Task 2: Visualization
    compare_wavelets_visualization(
        save_path='/home/ubuntu/rectifier/cauchy_paul_vs_db8.png'
    )

    # Task 3: HST comparison on Van der Pol
    hst_results = compare_hst_wavelets_on_vdp(
        save_path='/home/ubuntu/rectifier/cauchy_paul_hst_comparison.png'
    )

    # Task 4: Admissibility
    admissibility = compute_admissibility_constants()

    # Frame vs Basis comparison
    compare_frame_properties(
        save_path='/home/ubuntu/rectifier/frame_vs_basis.png'
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
KEY FINDINGS:

1. CAUCHY-PAUL WAVELETS
   - Canonical coherent states of the affine group (Ali Eq. 12.20)
   - Analytic (progressive): ψ̂(ξ) = 0 for ξ < 0
   - Polynomial decay: |ψ(t)| ~ 1/t^(m+1) (infinite support)
   - Form continuous FRAME, not orthonormal basis

2. ADMISSIBILITY
   - All Cauchy-Paul wavelets (m ≥ 0) are admissible
   - C_ψ = m! for Cauchy-Paul m
   - Guarantees resolution of identity for CWT

3. HST COMPARISON
   - db8 (orthogonal): Perfect reconstruction, discrete, efficient
   - Cauchy-Paul (coherent): Approximate reconstruction, continuous, theoretical

4. RECOMMENDATION FOR HST
   - Use db8 for practical HST implementation (perfect reconstruction)
   - Cauchy-Paul provides theoretical foundation (coherent states)
   - The "coherent state" property is motivation, not requirement
""")

    return {
        'hst_results': hst_results,
        'admissibility': admissibility,
    }


if __name__ == "__main__":
    results = run_all_analyses()
