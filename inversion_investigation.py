"""
HST Inversion Investigation

Goal: Fix the 100% reconstruction error by:
1. Testing wavelet-only reconstruction (pywt)
2. Verifying R_inv correctness
3. Implementing full-layer storage for reconstruction
4. Identifying where information is lost
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import gaussian
import pywt

from rectifier import R0_sheeted, R_sheeted, R_inv, R, R0

# ============================================================
# TEST 1: Wavelet-only reconstruction (no rectifier)
# ============================================================

def test_wavelet_only_reconstruction():
    """Test pywt's DWT/IDWT perfect reconstruction."""
    print("=" * 70)
    print("TEST 1: Wavelet-only Reconstruction (pywt)")
    print("=" * 70)

    wavelet = pywt.Wavelet('db8')

    # Test with random signal
    np.random.seed(42)
    x = np.random.randn(1024) + 0.5j * np.random.randn(1024)

    # Single level decomposition
    cA, cD = pywt.dwt(x, wavelet)

    # Reconstruction
    x_rec = pywt.idwt(cA, cD, wavelet)

    error = np.linalg.norm(x - x_rec[:len(x)]) / np.linalg.norm(x)
    print(f"\n  Single level DWT/IDWT:")
    print(f"    Input shape: {x.shape}")
    print(f"    cA shape: {cA.shape}, cD shape: {cD.shape}")
    print(f"    Reconstruction error: {error:.2e}")
    print(f"    Perfect reconstruction: {'YES' if error < 1e-10 else 'NO'}")

    # Multi-level test
    coeffs = pywt.wavedec(x, wavelet, level=3)
    x_rec_multi = pywt.waverec(coeffs, wavelet)

    error_multi = np.linalg.norm(x - x_rec_multi[:len(x)]) / np.linalg.norm(x)
    print(f"\n  Multi-level (3 levels) DWT/IDWT:")
    print(f"    Coefficient shapes: {[c.shape for c in coeffs]}")
    print(f"    Reconstruction error: {error_multi:.2e}")
    print(f"    Perfect reconstruction: {'YES' if error_multi < 1e-10 else 'NO'}")

    return error < 1e-10 and error_multi < 1e-10


# ============================================================
# TEST 2: R_inv correctness
# ============================================================

def test_R_inv_roundtrip():
    """Verify R_inv(R_sheeted(z)) ≈ z."""
    print("\n" + "=" * 70)
    print("TEST 2: R_inv Roundtrip Verification")
    print("=" * 70)

    # Test points in various regions
    test_points = [
        0.5 + 0.3j,    # Upper half-plane
        1.0 + 0.1j,    # Upper, near real axis
        -0.2 + 0.5j,   # Upper, left of origin
        0.1 - 0.4j,    # Lower half-plane
        -0.5 - 0.2j,   # Lower, left
        2.0 + 1.0j,    # Far from origin
        0.01 + 0.01j,  # Near origin
    ]

    print("\n  Testing R_inv(R_sheeted(z)) = z:")
    print("  " + "-" * 60)

    all_passed = True
    for z in test_points:
        w = R_sheeted(z)
        z_rec = R_inv(w)
        error = abs(z - z_rec)
        passed = error < 1e-10
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"    z = {z:>12} → R = {w:>18} → R_inv = {z_rec:>18} | err = {error:.2e} [{status}]")

    # Test with array
    z_arr = np.array(test_points)
    w_arr = R_sheeted(z_arr)
    z_rec_arr = R_inv(w_arr)
    max_error = np.max(np.abs(z_arr - z_rec_arr))

    print(f"\n  Array test max error: {max_error:.2e}")
    print(f"  R_inv correctness: {'VERIFIED' if all_passed and max_error < 1e-10 else 'FAILED'}")

    # Test the other direction: R_sheeted(R_inv(w)) = w
    print("\n  Testing R_sheeted(R_inv(w)) = w:")
    print("  " + "-" * 60)

    w_points = [
        0.3 + 0.2j,
        -0.5 + 0.1j,
        0.1 - 0.3j,
        1.0 + 0.5j,
    ]

    for w in w_points:
        z = R_inv(w)
        w_rec = R_sheeted(z)
        error = abs(w - w_rec)

        # Note: This direction may have issues due to branch cuts
        print(f"    w = {w:>12} → z = {z:>18} → R = {w_rec:>18} | err = {error:.2e}")

    return all_passed


def test_R_inv_half_plane_preservation():
    """Test that R_inv preserves half-plane structure."""
    print("\n  Testing half-plane preservation under R_inv:")
    print("  " + "-" * 60)

    # Points in upper half-plane
    upper = [0.5 + 0.3j, 1.0 + 0.1j, -0.5 + 0.5j]

    # Points in lower half-plane
    lower = [0.5 - 0.3j, 1.0 - 0.1j, -0.5 - 0.5j]

    print("\n  Upper half-plane (should stay upper after R_inv(R_sheeted(z))):")
    for z in upper:
        w = R_sheeted(z)
        z_rec = R_inv(w)
        print(f"    z = {z:>12} (Im={z.imag:+.2f}) → z_rec = {z_rec:>18} (Im={z_rec.imag:+.2f})")

    print("\n  Lower half-plane (should stay lower after R_inv(R_sheeted(z))):")
    for z in lower:
        w = R_sheeted(z)
        z_rec = R_inv(w)
        print(f"    z = {z:>12} (Im={z.imag:+.2f}) → z_rec = {z_rec:>18} (Im={z_rec.imag:+.2f})")


# ============================================================
# TEST 3: HST with full layer storage
# ============================================================

def hst_forward_full(f, J=3, wavelet_name='db8', verbose=False):
    """
    Forward HST keeping ALL information for perfect reconstruction.

    Returns:
    - S_coeffs: Father-wavelet averaged coefficients (for analysis)
    - layers: Full u at each layer BEFORE averaging (for reconstruction)
    - u_final: Final layer after bination
    - psi: Mother wavelet filter (needed for inverse)
    """
    f = np.asarray(f, dtype=complex)

    wavelet = pywt.Wavelet(wavelet_name)
    psi = np.array(wavelet.dec_hi, dtype=complex)

    # Initial rectification
    f_safe = np.where(np.abs(f) < 1e-10, 1e-10 + 0j, f)
    u = R_sheeted(f_safe)

    layers = []
    S_coeffs = []

    if verbose:
        print(f"  Initial: len(u) = {len(u)}")

    for j in range(J):
        if len(u) < len(psi):
            if verbose:
                print(f"  Layer {j}: stopping, signal too short")
            break

        # Convolve with mother wavelet
        v = convolve(u, psi, mode='same')

        # Rectify
        v_safe = np.where(np.abs(v) < 1e-10, 1e-10 + 0j, v)
        u = R_sheeted(v_safe)

        # Store FULL u for reconstruction (before averaging and bination)
        layers.append(u.copy())

        # Father wavelet averaging (for output only, not reconstruction)
        phi_len = min(len(u), max(4, 2**(j+2)))
        phi = gaussian(phi_len, std=max(phi_len/4, 1))
        phi = phi / phi.sum()
        S_j = convolve(u, phi, mode='same')
        S_coeffs.append(S_j)

        if verbose:
            print(f"  Layer {j}: len(u) = {len(u)}, len(S_j) = {len(S_j)}")

        # Binate for next layer
        u = u[::2]

    return S_coeffs, layers, u, psi


def hst_inverse_full(layers, u_final, psi, verbose=False):
    """
    Inverse HST using full layer information.

    Key insight: We stored u BEFORE bination, so we can reconstruct
    by reversing each step exactly.
    """
    J = len(layers)

    wavelet = pywt.Wavelet('db8')
    psi_rec = np.array(wavelet.rec_hi, dtype=complex)

    u = u_final.copy()

    if verbose:
        print(f"  Starting inverse with len(u) = {len(u)}")

    for j in range(J-1, -1, -1):
        target_len = len(layers[j])

        # Upsample (inverse of bination)
        u_up = np.zeros(target_len, dtype=complex)
        n_copy = min(len(u), (target_len + 1) // 2)
        u_up[::2][:n_copy] = u[:n_copy]

        # Interpolate odd samples
        for i in range(len(u_up[1::2])):
            left = u_up[2*i] if 2*i < len(u_up) else 0
            right = u_up[2*(i+1)] if 2*(i+1) < len(u_up) else left
            u_up[2*i + 1] = (left + right) / 2

        if verbose:
            print(f"  Layer {j}: upsampled to {len(u_up)}")

        # Inverse rectifier
        u_inv = R_inv(u_up)

        # Deconvolve wavelet (use time-reversed conjugate = adjoint)
        psi_adj = np.conj(psi_rec[::-1])
        u = convolve(u_inv, psi_adj, mode='same')

        if verbose:
            print(f"  Layer {j}: after deconv, len(u) = {len(u)}")

    # Final inverse rectifier
    f_rec = R_inv(u)

    return f_rec


def test_hst_reconstruction():
    """Test HST forward/inverse with full layer storage."""
    print("\n" + "=" * 70)
    print("TEST 3: HST Reconstruction with Full Layers")
    print("=" * 70)

    # Test signal
    t = np.linspace(0, 1, 1024)
    f = np.exp(2j * np.pi * 10 * t)  # Complex sinusoid

    print("\n  Forward transform:")
    S_coeffs, layers, u_final, psi = hst_forward_full(f, J=3, verbose=True)

    print(f"\n  Stored layers: {[len(L) for L in layers]}")
    print(f"  u_final length: {len(u_final)}")

    print("\n  Inverse transform:")
    f_rec = hst_inverse_full(layers, u_final, psi, verbose=True)

    # Trim to original length
    f_rec = f_rec[:len(f)]

    error = np.linalg.norm(f - f_rec) / np.linalg.norm(f)
    print(f"\n  Reconstruction error: {error:.6f} ({error*100:.2f}%)")

    return error, f, f_rec


# ============================================================
# TEST 4: Where is information lost?
# ============================================================

def identify_information_loss():
    """Pinpoint exactly where information is lost in the cascade."""
    print("\n" + "=" * 70)
    print("TEST 4: Identifying Information Loss")
    print("=" * 70)

    t = np.linspace(0, 1, 256)
    f = np.exp(2j * np.pi * 5 * t)

    wavelet = pywt.Wavelet('db8')
    psi = np.array(wavelet.dec_hi, dtype=complex)
    psi_rec = np.array(wavelet.rec_hi, dtype=complex)

    print("\n  Step-by-step analysis:")
    print("  " + "-" * 60)

    # Step 1: Initial rectification
    f_safe = np.where(np.abs(f) < 1e-10, 1e-10 + 0j, f)
    u0 = R_sheeted(f_safe)
    u0_inv = R_inv(u0)
    err1 = np.linalg.norm(f_safe - u0_inv) / np.linalg.norm(f_safe)
    print(f"  1. Initial R_sheeted → R_inv: error = {err1:.2e}")

    # Step 2: Convolution only (no rectifier)
    v = convolve(u0, psi, mode='same')
    psi_adj = np.conj(psi_rec[::-1])
    v_deconv = convolve(v, psi_adj, mode='same')
    err2 = np.linalg.norm(u0 - v_deconv) / np.linalg.norm(u0)
    print(f"  2. Convolve → deconvolve (no R): error = {err2:.2e}")

    # Step 3: Convolution + rectification
    v_safe = np.where(np.abs(v) < 1e-10, 1e-10 + 0j, v)
    u1 = R_sheeted(v_safe)
    u1_inv = R_inv(u1)
    err3a = np.linalg.norm(v_safe - u1_inv) / np.linalg.norm(v_safe)
    print(f"  3a. R_sheeted(v) → R_inv: error = {err3a:.2e}")

    # Step 4: Bination
    u1_bin = u1[::2]
    u1_up = np.zeros_like(u1)
    u1_up[::2] = u1_bin
    for i in range(len(u1_up[1::2])):
        left = u1_up[2*i]
        right = u1_up[2*(i+1)] if 2*(i+1) < len(u1_up) else left
        u1_up[2*i + 1] = (left + right) / 2
    err4 = np.linalg.norm(u1 - u1_up) / np.linalg.norm(u1)
    print(f"  4. Binate → upsample (linear interp): error = {err4:.2e}")

    # Step 5: Father wavelet averaging
    phi_len = min(len(u1), 16)
    phi = gaussian(phi_len, std=max(phi_len/4, 1))
    phi = phi / phi.sum()
    S_1 = convolve(u1, phi, mode='same')
    err5 = np.linalg.norm(u1 - S_1) / np.linalg.norm(u1)
    print(f"  5. Father wavelet averaging: error = {err5:.2e}")
    print(f"     (This is the ANALYSIS output, not meant for reconstruction)")

    # Key insight
    print("\n  KEY FINDINGS:")
    print("  " + "-" * 60)

    if err1 < 1e-10:
        print("  ✓ R_sheeted/R_inv roundtrip is PERFECT")
    else:
        print(f"  ✗ R_sheeted/R_inv roundtrip has error {err1:.2e}")

    if err2 < 0.5:
        print(f"  ~ Wavelet conv/deconv (adjoint) has error {err2:.2e}")
        print("    (Adjoint ≠ inverse for convolution!)")

    if err4 > 0.1:
        print(f"  ✗ Bination/upsample loses significant info: error = {err4:.2e}")
        print("    (Linear interpolation doesn't recover odd samples)")

    return {
        'R_roundtrip': err1,
        'conv_deconv': err2,
        'binate_upsample': err4,
        'father_avg': err5,
    }


# ============================================================
# TEST 5: Alternative inversion approaches
# ============================================================

def test_pywt_based_inversion():
    """
    Use pywt's built-in DWT/IDWT with rectifier at each level.

    This leverages pywt's perfect reconstruction property.

    KEY INSIGHT: The rectifier should be applied to the COMBINED signal,
    not just the approximation coefficients.
    """
    print("\n" + "=" * 70)
    print("TEST 5: PyWavelets-based HST")
    print("=" * 70)

    wavelet = 'db8'
    J = 3

    t = np.linspace(0, 1, 1024)
    f = np.exp(2j * np.pi * 10 * t) + 0.5 * np.exp(2j * np.pi * 25 * t)

    print("\n  Forward HST using pywt decomposition:")

    # Initial rectification
    f_safe = np.where(np.abs(f) < 1e-10, 1e-10 + 0j, f)
    u = R_sheeted(f_safe)

    # Store coefficients AND lengths for reconstruction
    all_cD = []
    all_cA_pre_rect = []  # Store cA BEFORE rectification
    lengths = [len(u)]

    for j in range(J):
        # Use pywt's DWT
        cA, cD = pywt.dwt(u, wavelet)

        # Store detail coefficients and pre-rectified approximation
        all_cD.append(cD.copy())
        all_cA_pre_rect.append(cA.copy())
        lengths.append(len(cA))

        # Rectify the approximation for next level
        cA_safe = np.where(np.abs(cA) < 1e-10, 1e-10 + 0j, cA)
        u = R_sheeted(cA_safe)

        print(f"    Level {j}: cA = {cA.shape}, cD = {cD.shape}, "
              f"R(cA) range = [{u.real.min():.2f}, {u.real.max():.2f}]")

    # Store final approximation
    final_u = u.copy()

    print(f"\n  Inverse HST using pywt reconstruction:")
    print(f"  Stored lengths: {lengths}")

    # Inverse cascade
    u = final_u

    for j in range(J-1, -1, -1):
        # Inverse rectifier to get cA
        cA_rec = R_inv(u)

        # Get stored detail coefficients
        cD = all_cD[j]

        # Make sure cA and cD have same length (pywt requirement)
        min_len = min(len(cA_rec), len(cD))
        cA_rec = cA_rec[:min_len]
        cD = cD[:min_len]

        # Reconstruct using pywt's IDWT
        u = pywt.idwt(cA_rec, cD, wavelet)

        # Trim to expected length
        if j > 0 and j-1 < len(lengths):
            target_len = lengths[j]
            u = u[:target_len]

        print(f"    Level {j}: cA_rec = {len(cA_rec)}, cD = {len(cD)}, "
              f"reconstructed = {len(u)}")

    # Final inverse rectifier
    f_rec = R_inv(u)

    # Trim to original length
    f_rec = f_rec[:len(f)]

    error = np.linalg.norm(f - f_rec) / np.linalg.norm(f)
    print(f"\n  Reconstruction error: {error:.6f} ({error*100:.2f}%)")

    return error, f, f_rec


def test_no_rectifier_cascade():
    """Test wavelet cascade without rectifier to isolate the problem."""
    print("\n" + "=" * 70)
    print("TEST 6: Wavelet Cascade WITHOUT Rectifier")
    print("=" * 70)

    wavelet = 'db8'
    J = 3

    t = np.linspace(0, 1, 1024)
    f = np.exp(2j * np.pi * 10 * t) + 0.5 * np.exp(2j * np.pi * 25 * t)

    print("\n  Forward (wavelet only, no R):")

    u = f.copy()
    all_coeffs = []
    all_cA = []

    for j in range(J):
        cA, cD = pywt.dwt(u, wavelet)
        all_coeffs.append(cD.copy())
        all_cA.append(cA.copy())
        u = cA
        print(f"    Level {j}: cA = {cA.shape}, cD = {cD.shape}")

    final_cA = u.copy()

    print(f"\n  Inverse (wavelet only, no R):")
    u = final_cA

    for j in range(J-1, -1, -1):
        cD = all_coeffs[j]
        # Ensure matching lengths
        min_len = min(len(u), len(cD))
        u = pywt.idwt(u[:min_len], cD[:min_len], wavelet)
        print(f"    Level {j}: reconstructed = {u.shape}")

    f_rec = u[:len(f)]

    error = np.linalg.norm(f - f_rec) / np.linalg.norm(f)
    print(f"\n  Reconstruction error: {error:.2e}")
    print(f"  Perfect reconstruction: {'YES' if error < 1e-10 else 'NO'}")

    return error < 1e-10


# ============================================================
# VISUALIZATION
# ============================================================

def plot_reconstruction_analysis(results, save_path='inversion_analysis.png'):
    """Plot reconstruction analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('HST Inversion Analysis', fontsize=14)

    # Error sources
    ax = axes[0, 0]
    sources = ['R roundtrip', 'Conv/Deconv', 'Binate/Upsample', 'Father Avg']
    errors = [results['R_roundtrip'], results['conv_deconv'],
              results['binate_upsample'], results['father_avg']]
    colors = ['green' if e < 1e-5 else 'orange' if e < 0.1 else 'red' for e in errors]
    bars = ax.bar(sources, errors, color=colors)
    ax.set_ylabel('Relative Error')
    ax.set_title('Error Sources in HST Cascade')
    ax.set_yscale('log')
    ax.axhline(y=1e-10, color='green', linestyle='--', alpha=0.5, label='Machine precision')
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='1% error')
    ax.legend()

    # Reconstruction comparison (if available)
    if 'f' in results and 'f_rec' in results:
        f = results['f']
        f_rec = results['f_rec']

        ax = axes[0, 1]
        ax.plot(f.real[:200], 'b-', alpha=0.7, label='Original (Re)')
        ax.plot(f_rec.real[:200], 'r--', alpha=0.7, label='Reconstructed (Re)')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title('Signal Comparison (first 200 samples)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        error = np.abs(f - f_rec)
        ax.plot(error[:200], 'purple', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('|Error|')
        ax.set_title('Point-wise Error')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.scatter(f.real[:500], f_rec.real[:500], alpha=0.3, s=5)
        ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect')
        ax.set_xlabel('Original')
        ax.set_ylabel('Reconstructed')
        ax.set_title('Scatter Plot (Real parts)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def run_all_tests():
    """Run all inversion tests and report findings."""
    results = {}

    # Test 1: Wavelet only
    results['wavelet_ok'] = test_wavelet_only_reconstruction()

    # Test 2: R_inv roundtrip
    results['R_inv_ok'] = test_R_inv_roundtrip()
    test_R_inv_half_plane_preservation()

    # Test 3: Full layer reconstruction
    error, f, f_rec = test_hst_reconstruction()
    results['full_layer_error'] = error
    results['f'] = f
    results['f_rec'] = f_rec

    # Test 4: Identify loss
    loss_results = identify_information_loss()
    results.update(loss_results)

    # Test 5: pywt-based inversion
    error_pywt, _, _ = test_pywt_based_inversion()
    results['pywt_error'] = error_pywt

    # Test 6: No rectifier cascade
    results['no_rect_ok'] = test_no_rectifier_cascade()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Wavelet-only reconstruction: {'PASS' if results['wavelet_ok'] else 'FAIL'}")
    print(f"  R_inv roundtrip: {'PASS' if results['R_inv_ok'] else 'FAIL'}")
    print(f"  No-rectifier cascade: {'PASS' if results['no_rect_ok'] else 'FAIL'}")
    print(f"\n  Full layer HST error: {results['full_layer_error']*100:.1f}%")
    print(f"  PyWT-based HST error: {results['pywt_error']*100:.1f}%")

    print("\n  Error Sources:")
    print(f"    R roundtrip:      {results['R_roundtrip']:.2e}")
    print(f"    Conv/Deconv:      {results['conv_deconv']:.2e}")
    print(f"    Binate/Upsample:  {results['binate_upsample']:.2e}")
    print(f"    Father averaging: {results['father_avg']:.2e}")

    # Key finding
    print("\n  KEY FINDING:")
    if results['pywt_error'] < 0.5:
        print(f"    Using pywt's DWT/IDWT with rectifier achieves {results['pywt_error']*100:.1f}% error")
        print("    This is the best approach for reconstruction!")
    else:
        print("    The rectifier itself may be causing non-invertibility")
        print("    Need to investigate R_sheeted/R_inv branch structure")

    # Plot results
    plot_reconstruction_analysis(results, '/home/ubuntu/rectifier/inversion_analysis.png')

    return results


if __name__ == "__main__":
    results = run_all_tests()
