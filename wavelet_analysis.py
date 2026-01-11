"""
Wavelet Analysis for HST

Deep dive into wavelet structure to understand (not just use) pywt:
1. Extract and verify filter coefficients
2. Verify Littlewood-Paley frequency tiling
3. Implement custom DWT/IDWT from first principles
4. Verify father wavelet partition of unity
5. Compare custom implementation to pywt
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

from rectifier import R_sheeted, R_inv

# ============================================================
# TASK 1: Extract and Verify pywt Filters
# ============================================================

def inspect_wavelet(name='db8'):
    """Extract and analyze wavelet filters from pywt"""
    print("=" * 70)
    print(f"TASK 1: Inspecting Wavelet '{name}'")
    print("=" * 70)

    wavelet = pywt.Wavelet(name)

    print(f"\nWavelet: {name}")
    print(f"Filter length: {wavelet.dec_len}")
    print(f"Orthogonal: {wavelet.orthogonal}")
    print(f"Biorthogonal: {wavelet.biorthogonal}")
    print(f"Symmetry: {wavelet.symmetry}")

    # Decomposition filters
    lo_d = np.array(wavelet.dec_lo)  # Father (low-pass) for decomposition
    hi_d = np.array(wavelet.dec_hi)  # Mother (high-pass) for decomposition

    # Reconstruction filters
    lo_r = np.array(wavelet.rec_lo)  # Father for reconstruction
    hi_r = np.array(wavelet.rec_hi)  # Mother for reconstruction

    print(f"\n--- Filter Coefficients ---")
    print(f"Decomposition low-pass (h, father):")
    for i, c in enumerate(lo_d):
        print(f"  h[{i}] = {c:+.10f}")

    print(f"\nDecomposition high-pass (g, mother):")
    for i, c in enumerate(hi_d):
        print(f"  g[{i}] = {c:+.10f}")

    # Verify QMF relations for orthogonal wavelets:
    # g[n] = (-1)^n * h[N-1-n]  (alternating flip)
    N = len(lo_d)
    hi_d_from_qmf = np.array([(-1)**n * lo_d[N-1-n] for n in range(N)])
    qmf_error = np.max(np.abs(hi_d - hi_d_from_qmf))
    qmf_match = qmf_error < 1e-10

    print(f"\n--- QMF Verification ---")
    print(f"QMF relation: g[n] = (-1)^n * h[N-1-n]")
    print(f"Max deviation: {qmf_error:.2e}")
    print(f"QMF satisfied: {qmf_match}")

    # Verify orthonormality: sum of squares = 1 (for energy preservation)
    # Actually for DWT: sum of squares = 2 (due to downsampling by 2)
    lo_sq_sum = np.sum(lo_d**2)
    hi_sq_sum = np.sum(hi_d**2)
    print(f"\n--- Normalization ---")
    print(f"Sum(h²) = {lo_sq_sum:.10f}")
    print(f"Sum(g²) = {hi_sq_sum:.10f}")
    print(f"(For orthogonal wavelets, these should equal 1)")

    # Verify orthogonality: <lo, hi> = 0
    inner = np.sum(lo_d * hi_d)
    print(f"\n--- Orthogonality ---")
    print(f"<h, g> = {inner:.2e} (should be 0)")

    # Verify double-shift orthogonality: sum_n h[n]*h[n-2k] = delta[k]
    print(f"\n--- Double-shift Orthogonality ---")
    for k in range(4):
        shifted = np.zeros(N + 2*k)
        shifted[:N] = lo_d
        original = np.zeros(N + 2*k)
        original[2*k:2*k+N] = lo_d
        inner_k = np.sum(shifted * original)
        expected = 1.0 if k == 0 else 0.0
        print(f"  Sum_n h[n]*h[n-{2*k}] = {inner_k:.6f} (expected {expected})")

    # Verify reconstruction filters are time-reversed
    print(f"\n--- Reconstruction Filters ---")
    lo_r_expected = lo_d[::-1]
    hi_r_expected = hi_d[::-1]
    print(f"rec_lo = dec_lo[::-1]: {np.allclose(lo_r, lo_r_expected)}")
    print(f"rec_hi = dec_hi[::-1]: {np.allclose(hi_r, hi_r_expected)}")

    return {
        'lo_d': lo_d, 'hi_d': hi_d,
        'lo_r': lo_r, 'hi_r': hi_r,
        'qmf_satisfied': qmf_match,
        'name': name
    }


# ============================================================
# TASK 2: Verify Littlewood-Paley Condition
# ============================================================

def verify_littlewood_paley(lo_d, hi_d, n_freq=1024, save_path='littlewood_paley.png'):
    """
    Verify Littlewood-Paley: |H(ω)|² + |G(ω)|² = 2 for all ω

    Where:
    - H(ω) = Fourier transform of low-pass filter h
    - G(ω) = Fourier transform of high-pass filter g

    For orthogonal wavelets with ||h||² = ||g||² = 1,
    the sum should be 2 (not 1) due to downsampling.
    """
    print("\n" + "=" * 70)
    print("TASK 2: Littlewood-Paley Condition")
    print("=" * 70)

    # Zero-pad for frequency resolution
    n_pad = n_freq
    lo_padded = np.zeros(n_pad)
    hi_padded = np.zeros(n_pad)
    lo_padded[:len(lo_d)] = lo_d
    hi_padded[:len(hi_d)] = hi_d

    # Compute frequency responses
    H = np.fft.fft(lo_padded)
    G = np.fft.fft(hi_padded)

    # Littlewood-Paley sum: |H(ω)|² + |G(ω)|²
    LP_sum = np.abs(H)**2 + np.abs(G)**2

    print(f"\nLittlewood-Paley sum |H(ω)|² + |G(ω)|²:")
    print(f"  Min: {LP_sum.min():.10f}")
    print(f"  Max: {LP_sum.max():.10f}")
    print(f"  Mean: {LP_sum.mean():.10f}")
    print(f"  Expected: 2.0 for orthogonal wavelets")
    print(f"  Deviation from 2: {np.abs(LP_sum - 2).max():.2e}")

    # Additional check: |H(ω)|² + |H(ω+π)|² = 2 (power complementary)
    H_shifted = np.roll(H, n_pad//2)
    power_comp = np.abs(H)**2 + np.abs(H_shifted)**2
    print(f"\nPower complementary |H(ω)|² + |H(ω+π)|²:")
    print(f"  Min: {power_comp.min():.10f}")
    print(f"  Max: {power_comp.max():.10f}")

    # Plot
    freq = np.fft.fftfreq(n_pad)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Littlewood-Paley Analysis', fontsize=14)

    # Frequency responses
    ax = axes[0, 0]
    ax.plot(freq[:n_pad//2], np.abs(H[:n_pad//2])**2, 'b-', label='|H(ω)|² (low-pass)', lw=1.5)
    ax.plot(freq[:n_pad//2], np.abs(G[:n_pad//2])**2, 'r-', label='|G(ω)|² (high-pass)', lw=1.5)
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('Power')
    ax.legend()
    ax.set_title('Filter Frequency Responses')
    ax.grid(True, alpha=0.3)

    # Littlewood-Paley sum
    ax = axes[0, 1]
    ax.plot(freq[:n_pad//2], LP_sum[:n_pad//2], 'g-', lw=1.5)
    ax.axhline(y=2, color='r', linestyle='--', label='Expected = 2', alpha=0.7)
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('Sum')
    ax.legend()
    ax.set_title('Littlewood-Paley Sum: |H|² + |G|²')
    ax.set_ylim([1.9, 2.1])
    ax.grid(True, alpha=0.3)

    # Filter impulse responses
    ax = axes[1, 0]
    n = np.arange(len(lo_d))
    ax.stem(n, lo_d, 'b', markerfmt='bo', label='h (low-pass)')
    ax.stem(n + 0.2, hi_d, 'r', markerfmt='ro', label='g (high-pass)')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Coefficient')
    ax.legend()
    ax.set_title('Filter Coefficients')
    ax.grid(True, alpha=0.3)

    # Phase responses
    ax = axes[1, 1]
    ax.plot(freq[:n_pad//2], np.angle(H[:n_pad//2]), 'b-', label='∠H(ω)', lw=1.5)
    ax.plot(freq[:n_pad//2], np.angle(G[:n_pad//2]), 'r-', label='∠G(ω)', lw=1.5)
    ax.set_xlabel('Frequency (normalized)')
    ax.set_ylabel('Phase (radians)')
    ax.legend()
    ax.set_title('Filter Phase Responses')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return LP_sum


# ============================================================
# TASK 3: Custom DWT/IDWT Implementation
# ============================================================

def custom_dwt(x, lo_d, hi_d, mode='periodization'):
    """
    Custom DWT implementation from first principles.

    Algorithm:
    1. Convolve with low-pass filter → approximation
    2. Convolve with high-pass filter → detail
    3. Downsample both by 2

    For periodization mode: periodic extension before convolution.
    """
    N = len(x)
    L = len(lo_d)

    if mode == 'periodization':
        # Periodic extension
        x_ext = np.tile(x, 3)  # Repeat signal
        offset = N

        # Convolve
        cA_full = np.convolve(x_ext, lo_d, mode='full')
        cD_full = np.convolve(x_ext, hi_d, mode='full')

        # Extract central portion and downsample
        start = offset + L//2 - 1
        cA = cA_full[start:start + N:2]
        cD = cD_full[start:start + N:2]

    else:  # 'full' mode
        cA_full = np.convolve(x, lo_d, mode='full')
        cD_full = np.convolve(x, hi_d, mode='full')
        cA = cA_full[::2]
        cD = cD_full[::2]

    return cA, cD


def custom_idwt(cA, cD, lo_r, hi_r, mode='periodization'):
    """
    Custom inverse DWT.

    Algorithm:
    1. Upsample both by 2 (insert zeros)
    2. Convolve with reconstruction filters
    3. Add results
    """
    L = len(lo_r)

    # Upsample by inserting zeros
    N = len(cA)
    cA_up = np.zeros(2 * N)
    cA_up[::2] = cA
    cD_up = np.zeros(2 * N)
    cD_up[::2] = cD

    # Convolve with reconstruction filters
    x_lo = np.convolve(cA_up, lo_r, mode='full')
    x_hi = np.convolve(cD_up, hi_r, mode='full')

    # Sum
    x_rec = x_lo + x_hi

    if mode == 'periodization':
        # Extract correct portion
        start = L - 1
        x_rec = x_rec[start:start + 2*N]

    return x_rec


def test_custom_dwt(filters, save_path='custom_dwt_test.png'):
    """Test custom DWT implementation vs pywt"""
    print("\n" + "=" * 70)
    print("TASK 3: Custom DWT/IDWT Implementation")
    print("=" * 70)

    lo_d = filters['lo_d']
    hi_d = filters['hi_d']
    lo_r = filters['lo_r']
    hi_r = filters['hi_r']
    wavelet_name = filters['name']

    # Test signal
    np.random.seed(42)
    N = 128
    x = np.sin(2*np.pi*5*np.linspace(0, 1, N)) + 0.3*np.random.randn(N)

    print(f"\nTest signal: N = {N}")

    # pywt with periodization mode
    cA_pywt, cD_pywt = pywt.dwt(x, wavelet_name, mode='periodization')
    print(f"\npywt (periodization mode):")
    print(f"  cA shape: {cA_pywt.shape}")
    print(f"  cD shape: {cD_pywt.shape}")

    # Custom implementation
    cA_custom, cD_custom = custom_dwt(x, lo_d, hi_d, mode='periodization')
    print(f"\nCustom implementation:")
    print(f"  cA shape: {cA_custom.shape}")
    print(f"  cD shape: {cD_custom.shape}")

    # Compare coefficients
    if len(cA_pywt) == len(cA_custom):
        cA_error = np.linalg.norm(cA_pywt - cA_custom) / np.linalg.norm(cA_pywt)
        cD_error = np.linalg.norm(cD_pywt - cD_custom) / np.linalg.norm(cD_pywt)
        print(f"\nCoefficient comparison:")
        print(f"  cA relative error: {cA_error:.2e}")
        print(f"  cD relative error: {cD_error:.2e}")
    else:
        print(f"\n  Length mismatch, skipping coefficient comparison")
        cA_error = None

    # Test reconstruction
    x_rec_pywt = pywt.idwt(cA_pywt, cD_pywt, wavelet_name, mode='periodization')
    x_rec_custom = custom_idwt(cA_custom, cD_custom, lo_r, hi_r, mode='periodization')

    print(f"\nReconstruction:")
    print(f"  pywt output length: {len(x_rec_pywt)}")
    print(f"  Custom output length: {len(x_rec_custom)}")

    # Reconstruction errors
    min_len = min(len(x), len(x_rec_pywt), len(x_rec_custom))
    error_pywt = np.linalg.norm(x[:min_len] - x_rec_pywt[:min_len]) / np.linalg.norm(x[:min_len])
    error_custom = np.linalg.norm(x[:min_len] - x_rec_custom[:min_len]) / np.linalg.norm(x[:min_len])

    print(f"\nReconstruction errors:")
    print(f"  pywt:   {error_pywt:.2e}")
    print(f"  Custom: {error_custom:.2e}")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Custom DWT vs pywt Comparison', fontsize=14)

    # Original signal
    ax = axes[0, 0]
    ax.plot(x, 'b-', lw=0.8)
    ax.set_title('Original Signal')
    ax.set_xlabel('Sample')
    ax.grid(True, alpha=0.3)

    # Approximation coefficients
    ax = axes[0, 1]
    ax.plot(cA_pywt, 'b-', label='pywt', alpha=0.7)
    ax.plot(cA_custom, 'r--', label='Custom', alpha=0.7)
    ax.set_title('Approximation Coefficients (cA)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Detail coefficients
    ax = axes[1, 0]
    ax.plot(cD_pywt, 'b-', label='pywt', alpha=0.7)
    ax.plot(cD_custom, 'r--', label='Custom', alpha=0.7)
    ax.set_title('Detail Coefficients (cD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction comparison
    ax = axes[1, 1]
    ax.plot(x[:min_len], 'b-', label='Original', alpha=0.7)
    ax.plot(x_rec_pywt[:min_len], 'g--', label='pywt', alpha=0.7)
    ax.plot(x_rec_custom[:min_len], 'r:', label='Custom', alpha=0.7)
    ax.set_title('Reconstruction Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction error
    ax = axes[2, 0]
    ax.plot(np.abs(x[:min_len] - x_rec_pywt[:min_len]), 'g-', label='pywt error')
    ax.plot(np.abs(x[:min_len] - x_rec_custom[:min_len]), 'r-', label='Custom error')
    ax.set_title('Point-wise Reconstruction Error')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Multi-level test
    ax = axes[2, 1]
    # Test 3-level decomposition
    x_test = x.copy()
    coeffs_custom = []
    for level in range(3):
        cA_l, cD_l = custom_dwt(x_test, lo_d, hi_d, mode='periodization')
        coeffs_custom.append(cD_l)
        x_test = cA_l
    coeffs_custom.append(cA_l)  # Final approximation

    # Reconstruct
    u = coeffs_custom[-1]
    for level in range(2, -1, -1):
        u = custom_idwt(u, coeffs_custom[level], lo_r, hi_r, mode='periodization')

    error_3level = np.linalg.norm(x[:len(u)] - u[:len(x)]) / np.linalg.norm(x)
    ax.plot(x, 'b-', label='Original', alpha=0.7)
    ax.plot(u[:len(x)], 'r--', label=f'3-level rec (err={error_3level:.2e})', alpha=0.7)
    ax.set_title('3-Level Custom DWT Reconstruction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return {
        'cA_error': cA_error,
        'reconstruction_error_pywt': error_pywt,
        'reconstruction_error_custom': error_custom,
        'multilevel_error': error_3level
    }


# ============================================================
# TASK 4: Father Wavelet and Partition of Unity
# ============================================================

def create_father_wavelet_gaussian(length, scale):
    """
    Gaussian father wavelet for spatial averaging.

    Glinsky: "Father wavelet is Gaussian-like windowing"
    """
    from scipy.signal.windows import gaussian

    phi = gaussian(length, std=scale/4)
    phi = phi / phi.sum()  # Normalize to sum to 1

    return phi


def verify_partition_of_unity(phi, signal_length, save_path='partition_of_unity.png'):
    """
    Verify that overlapping father wavelets sum to ~1.

    For stride s and wavelets φ centered at positions 0, s, 2s, ...
    we need: Σ_k φ(x - k*s) ≈ 1 for all x
    """
    print("\n" + "=" * 70)
    print("TASK 4: Father Wavelet and Partition of Unity")
    print("=" * 70)

    print(f"\nFather wavelet length: {len(phi)}")
    print(f"Signal length: {signal_length}")

    # Test different strides
    strides = [len(phi)//4, len(phi)//2, len(phi)*3//4, len(phi)]
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Partition of Unity Analysis', fontsize=14)

    for idx, stride in enumerate(strides):
        n_windows = (signal_length - len(phi)) // stride + 1

        partition_sum = np.zeros(signal_length)

        for k in range(n_windows):
            start = k * stride
            end = start + len(phi)
            if end <= signal_length:
                partition_sum[start:end] += phi

        # Check flatness (excluding edges)
        margin = len(phi)
        if signal_length > 2 * margin:
            central = partition_sum[margin:-margin]
            deviation = np.abs(central - 1).max()
            mean_val = central.mean()
        else:
            central = partition_sum
            deviation = np.abs(partition_sum - 1).max()
            mean_val = partition_sum.mean()

        results[stride] = {
            'deviation': deviation,
            'mean': mean_val,
            'partition_sum': partition_sum
        }

        print(f"\nStride = {stride} ({100*stride/len(phi):.0f}% of filter length):")
        print(f"  Number of windows: {n_windows}")
        print(f"  Central region mean: {mean_val:.4f}")
        print(f"  Max deviation from 1: {deviation:.4f}")

        ax = axes[idx // 2, idx % 2]
        ax.plot(partition_sum, 'b-', lw=1)
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Target = 1')
        ax.fill_between(range(margin), 0, 2, alpha=0.2, color='gray')
        ax.fill_between(range(signal_length-margin, signal_length), 0, 2, alpha=0.2, color='gray')
        ax.set_xlabel('Position')
        ax.set_ylabel('Sum')
        ax.set_title(f'Stride = {stride} ({100*stride/len(phi):.0f}%), dev={deviation:.3f}')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    # Recommend optimal stride
    best_stride = min(results.keys(), key=lambda s: results[s]['deviation'])
    print(f"\n*** Recommended stride: {best_stride} ({100*best_stride/len(phi):.0f}% overlap)")
    print(f"    Deviation from unity: {results[best_stride]['deviation']:.4f}")

    return results


# ============================================================
# TASK 5: Custom HST Implementation
# ============================================================

def hst_forward_custom(f, lo_d, hi_d, J=3, verbose=False):
    """HST forward using custom DWT (complex signals)"""
    print("\n" + "=" * 70)
    print("TASK 5: Custom HST Implementation")
    print("=" * 70)

    f = np.asarray(f, dtype=complex)

    # Initial rectification
    f_safe = np.where(np.abs(f) < 1e-10, 1e-10 + 0j, f)
    u = R_sheeted(f_safe)

    layers = []
    lengths = [len(u)]

    if verbose:
        print(f"\nForward HST (custom wavelets):")
        print(f"  Initial: len = {len(u)}")

    for j in range(J):
        # DWT on complex signal (separate real and imaginary)
        cA_r, cD_r = custom_dwt(u.real, lo_d, hi_d, mode='periodization')
        cA_i, cD_i = custom_dwt(u.imag, lo_d, hi_d, mode='periodization')

        cA = cA_r + 1j * cA_i
        cD = cD_r + 1j * cD_i

        layers.append({'cA': cA.copy(), 'cD': cD.copy()})
        lengths.append(len(cA))

        # Rectify approximation
        cA_safe = np.where(np.abs(cA) < 1e-10, 1e-10 + 0j, cA)
        u = R_sheeted(cA_safe)

        if verbose:
            print(f"  Layer {j}: cA = {len(cA)}, cD = {len(cD)}")

    return {
        'layers': layers,
        'u_final': u.copy(),
        'lengths': lengths,
        'J': J
    }


def hst_inverse_custom(coeffs, lo_r, hi_r, original_length=None, verbose=False):
    """HST inverse using custom IDWT"""
    layers = coeffs['layers']
    u = coeffs['u_final'].copy()
    lengths = coeffs['lengths']
    J = coeffs['J']

    if verbose:
        print(f"\nInverse HST (custom wavelets):")
        print(f"  Starting: len = {len(u)}")

    for j in range(J-1, -1, -1):
        # Inverse rectifier
        cA = R_inv(u)

        # Get detail coefficients
        cD = layers[j]['cD']

        # Ensure matching lengths
        min_len = min(len(cA), len(cD))
        cA = cA[:min_len]
        cD = cD[:min_len]

        # IDWT (separate real and imaginary)
        x_r = custom_idwt(cA.real, cD.real, lo_r, hi_r, mode='periodization')
        x_i = custom_idwt(cA.imag, cD.imag, lo_r, hi_r, mode='periodization')

        u = x_r + 1j * x_i

        # Trim to expected length
        if j < len(lengths):
            target_len = lengths[j]
            u = u[:target_len]

        if verbose:
            print(f"  Layer {j}: reconstructed len = {len(u)}")

    # Final inverse rectifier
    f_rec = R_inv(u)

    if original_length is not None:
        f_rec = f_rec[:original_length]

    return f_rec


def test_custom_hst(filters, save_path='custom_hst_test.png'):
    """Test custom HST vs pywt-based HST"""
    print("\n--- Testing Custom HST ---")

    lo_d = filters['lo_d']
    hi_d = filters['hi_d']
    lo_r = filters['lo_r']
    hi_r = filters['hi_r']

    # Test signals
    np.random.seed(42)
    t = np.linspace(0, 1, 256)
    signals = [
        ("Sinusoid", np.exp(2j * np.pi * 10 * t)),
        ("Chirp", np.exp(2j * np.pi * (5*t + 20*t**2))),
        ("Random", np.random.randn(256) + 1j * np.random.randn(256)),
    ]

    results = {}
    fig, axes = plt.subplots(len(signals), 2, figsize=(12, 4*len(signals)))
    fig.suptitle('Custom HST Reconstruction Test', fontsize=14)

    for idx, (name, f) in enumerate(signals):
        # Forward
        coeffs = hst_forward_custom(f, lo_d, hi_d, J=3, verbose=(idx == 0))

        # Inverse
        f_rec = hst_inverse_custom(coeffs, lo_r, hi_r,
                                   original_length=len(f), verbose=(idx == 0))

        # Error
        error = np.linalg.norm(f - f_rec) / np.linalg.norm(f)
        results[name] = error

        print(f"\n{name}: reconstruction error = {error:.2e}")

        # Plot
        ax = axes[idx, 0]
        ax.plot(f.real, 'b-', label='Original (Re)', alpha=0.7)
        ax.plot(f_rec.real, 'r--', label='Reconstructed (Re)', alpha=0.7)
        ax.set_title(f'{name}: Real Part')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[idx, 1]
        ax.plot(np.abs(f - f_rec), 'purple')
        ax.set_title(f'{name}: |Error| (total = {error:.2e})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return results


# ============================================================
# TASK 6: Compare Custom vs pywt HST
# ============================================================

def compare_custom_vs_pywt():
    """Direct comparison of custom and pywt-based HST"""
    print("\n" + "=" * 70)
    print("TASK 6: Custom vs pywt HST Comparison")
    print("=" * 70)

    from hst import hst_forward_pywt, hst_inverse_pywt

    # Test signal
    np.random.seed(42)
    t = np.linspace(0, 1, 256)
    f = np.exp(2j * np.pi * 10 * t) + 0.5 * np.exp(2j * np.pi * 25 * t)

    # pywt-based HST
    coeffs_pywt = hst_forward_pywt(f, J=3)
    f_rec_pywt = hst_inverse_pywt(coeffs_pywt, original_length=len(f))
    error_pywt = np.linalg.norm(f - f_rec_pywt) / np.linalg.norm(f)

    # Custom HST
    filters = inspect_wavelet('db8')
    coeffs_custom = hst_forward_custom(f, filters['lo_d'], filters['hi_d'], J=3)
    f_rec_custom = hst_inverse_custom(coeffs_custom, filters['lo_r'], filters['hi_r'],
                                       original_length=len(f))
    error_custom = np.linalg.norm(f - f_rec_custom) / np.linalg.norm(f)

    print(f"\nReconstruction errors:")
    print(f"  pywt-based:  {error_pywt:.2e}")
    print(f"  Custom impl: {error_custom:.2e}")

    # Compare intermediate coefficients
    print(f"\nFinal approximation comparison:")
    print(f"  pywt u_final shape: {coeffs_pywt['cA_final'].shape}")
    print(f"  Custom u_final shape: {coeffs_custom['u_final'].shape}")

    return {
        'error_pywt': error_pywt,
        'error_custom': error_custom
    }


# ============================================================
# MAIN
# ============================================================

def run_all_analyses():
    """Run all wavelet analyses"""
    results = {}

    # Task 1: Inspect wavelet
    filters = inspect_wavelet('db8')
    results['filters'] = filters

    # Task 2: Littlewood-Paley
    LP_sum = verify_littlewood_paley(filters['lo_d'], filters['hi_d'],
                                      save_path='/home/ubuntu/rectifier/littlewood_paley.png')
    results['LP_sum'] = LP_sum

    # Task 3: Custom DWT
    dwt_results = test_custom_dwt(filters,
                                   save_path='/home/ubuntu/rectifier/custom_dwt_test.png')
    results['dwt'] = dwt_results

    # Task 4: Partition of unity
    phi = create_father_wavelet_gaussian(64, scale=32)
    pou_results = verify_partition_of_unity(phi, 512,
                                             save_path='/home/ubuntu/rectifier/partition_of_unity.png')
    results['partition_of_unity'] = pou_results

    # Task 5: Custom HST
    hst_results = test_custom_hst(filters,
                                   save_path='/home/ubuntu/rectifier/custom_hst_test.png')
    results['hst_custom'] = hst_results

    # Task 6: Comparison
    comparison = compare_custom_vs_pywt()
    results['comparison'] = comparison

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n[1] QMF relation satisfied: {filters['qmf_satisfied']}")
    print(f"[2] Littlewood-Paley deviation from 2: {np.abs(LP_sum - 2).max():.2e}")
    print(f"[3] Custom DWT reconstruction error: {dwt_results['reconstruction_error_custom']:.2e}")
    print(f"[4] Best partition-of-unity stride: {min(pou_results.keys(), key=lambda s: pou_results[s]['deviation'])}")
    print(f"[5] Custom HST reconstruction errors: {hst_results}")
    print(f"[6] pywt vs Custom HST: pywt={comparison['error_pywt']:.2e}, custom={comparison['error_custom']:.2e}")

    return results


if __name__ == "__main__":
    results = run_all_analyses()
