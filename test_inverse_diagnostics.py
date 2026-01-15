"""
Diagnose where inverse path fails in the Glinsky pipeline.

Tests:
1. HST roundtrip (no PCA)
2. HST + PCA roundtrip
3. β ↔ (p,q) pseudo-inverse roundtrip
4. Simple sine wave forecast

Expected: Find which component causes the 5.0 reconstruction error.
"""

import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from hst import hst_forward_pywt, hst_inverse_pywt
from hst_rom import HST_ROM, SimplePCA


def generate_sho_signal(omega: float = 1.0, n_points: int = 128,
                        E: float = 2.0, theta0: float = 0.0) -> np.ndarray:
    """Generate complex SHO signal z(t) = q(t) + i*p(t)/omega."""
    dt = 0.01
    t = np.arange(n_points) * dt

    theta_t = theta0 + omega * t
    p_t = np.sqrt(2 * E) * np.sin(theta_t)
    q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)

    z = q_t + 1j * p_t / omega
    return z


def test_1_hst_roundtrip():
    """
    Pure HST roundtrip (no PCA).
    signal → hst_forward() → coeffs → hst_inverse() → signal_reconstructed

    Expected: error < 1e-6 (near-perfect reconstruction)
    """
    print("=" * 70)
    print("TEST 1: HST ROUNDTRIP (no PCA)")
    print("=" * 70)

    # Generate test signal
    z = generate_sho_signal(omega=1.0, n_points=128, E=2.0, theta0=0.5)

    print(f"\nInput signal:")
    print(f"  Length: {len(z)}")
    print(f"  |z| range: [{np.abs(z).min():.4f}, {np.abs(z).max():.4f}]")

    # Forward HST
    coeffs = hst_forward_pywt(z, J=3, wavelet_name='db8')

    print(f"\nHST coefficients:")
    print(f"  Levels: {len(coeffs['cD'])}")
    print(f"  cA_final length: {len(coeffs['cA_final'])}")

    # Inverse HST
    z_rec = hst_inverse_pywt(coeffs, original_length=len(z))

    # Error
    error = np.linalg.norm(z_rec - z) / np.linalg.norm(z)

    print(f"\nReconstruction:")
    print(f"  |z_rec| range: [{np.abs(z_rec).min():.4f}, {np.abs(z_rec).max():.4f}]")
    print(f"  Relative error: {error:.2e}")

    passed = error < 1e-6
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'} (threshold: 1e-6)")

    return {'error': error, 'passed': passed}


def test_2_pca_roundtrip():
    """
    HST + PCA roundtrip.
    signal → HST → flatten → PCA.transform() → β → PCA.inverse_transform() → unflatten → iHST

    Measure: How much error from PCA truncation?
    """
    print("\n" + "=" * 70)
    print("TEST 2: HST + PCA ROUNDTRIP")
    print("=" * 70)

    # Generate multiple signals for PCA fitting
    np.random.seed(42)
    n_signals = 100
    n_points = 128

    signals = []
    for i in range(n_signals):
        E = np.random.uniform(0.5, 4.5)
        theta0 = np.random.uniform(0, 2 * np.pi)
        z = generate_sho_signal(omega=1.0, n_points=n_points, E=E, theta0=theta0)
        signals.append(z)

    print(f"\nGenerated {n_signals} signals for PCA fitting")

    # Test different n_components
    for n_comp in [2, 4, 8, 16]:
        rom = HST_ROM(n_components=n_comp, wavelet='db8', J=3, window_size=n_points)
        betas = rom.fit(signals, extract_windows=False)

        # Variance explained
        var_explained = sum(rom.pca.explained_variance_ratio_)

        # Test reconstruction on held-out signal
        test_signal = generate_sho_signal(omega=1.0, n_points=n_points, E=2.0, theta0=0.5)
        beta = rom.transform(test_signal)
        z_rec = rom.inverse_transform(beta, original_length=n_points)

        error = np.linalg.norm(z_rec - test_signal) / np.linalg.norm(test_signal)

        status = "✓" if error < 0.5 else "✗"
        print(f"\n  n_components={n_comp:2d}: var_explained={var_explained:.3f}, error={error:.4f} {status}")

    # Detailed analysis with n_comp=4
    print("\n" + "-" * 40)
    print("Detailed analysis with n_components=4:")

    rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=n_points)
    betas = rom.fit(signals, extract_windows=False)

    errors = []
    for i in range(20):
        z = signals[i]
        beta = rom.transform(z)
        z_rec = rom.inverse_transform(beta, original_length=n_points)
        err = np.linalg.norm(z_rec - z) / np.linalg.norm(z)
        errors.append(err)

    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Std error: {np.std(errors):.4f}")
    print(f"  Max error: {np.max(errors):.4f}")

    passed = np.mean(errors) < 0.5
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'} (threshold: 0.5)")

    return {'mean_error': np.mean(errors), 'passed': passed}


def test_3_beta_pq_roundtrip():
    """
    The pseudo-inverse path.
    β → W @ β → (p,q) → W_inv @ (p,q) → β_reconstructed

    Measure: ||β - β_reconstructed|| / ||β||
    """
    print("\n" + "=" * 70)
    print("TEST 3: β ↔ (p,q) PSEUDO-INVERSE ROUNDTRIP")
    print("=" * 70)

    # Generate signals with known (p,q)
    np.random.seed(42)
    n_signals = 100
    n_points = 128
    omega = 1.0

    signals = []
    pq_true = []

    for i in range(n_signals):
        E = np.random.uniform(0.5, 4.5)
        theta0 = np.random.uniform(0, 2 * np.pi)

        p0 = np.sqrt(2 * E) * np.sin(theta0)
        q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

        z = generate_sho_signal(omega=omega, n_points=n_points, E=E, theta0=theta0)
        signals.append(z)

        # (p,q) at window center
        center_idx = n_points // 2
        theta_c = theta0 + omega * center_idx * 0.01
        p_c = np.sqrt(2 * E) * np.sin(theta_c)
        q_c = np.sqrt(2 * E) / omega * np.cos(theta_c)
        pq_true.append([p_c, q_c])

    pq_true = np.array(pq_true)

    # Fit ROM
    rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=n_points)
    betas = rom.fit(signals, extract_windows=False)

    print(f"\nFitted HST_ROM with n_components=4")
    print(f"β shape: {betas.shape}")

    # Learn β → (p,q) mapping
    W_beta_to_pq, _, _, _ = np.linalg.lstsq(betas, pq_true, rcond=None)
    W_pq_to_beta = np.linalg.pinv(W_beta_to_pq)

    print(f"\nW_beta_to_pq shape: {W_beta_to_pq.shape}")
    print(f"W_pq_to_beta shape: {W_pq_to_beta.shape}")

    # Test roundtrip: β → (p,q) → β
    pq_from_beta = betas @ W_beta_to_pq
    beta_reconstructed = pq_from_beta @ W_pq_to_beta

    # Error
    beta_errors = np.linalg.norm(beta_reconstructed - betas, axis=1) / (np.linalg.norm(betas, axis=1) + 1e-10)

    print(f"\nβ → (p,q) → β roundtrip:")
    print(f"  Mean relative error: {np.mean(beta_errors):.4f}")
    print(f"  Max relative error: {np.max(beta_errors):.4f}")

    # Also check (p,q) prediction quality
    r_p, _ = pearsonr(pq_from_beta[:, 0], pq_true[:, 0])
    r_q, _ = pearsonr(pq_from_beta[:, 1], pq_true[:, 1])
    print(f"\n(p,q) prediction quality:")
    print(f"  r(p): {r_p:.4f}")
    print(f"  r(q): {r_q:.4f}")

    # The key issue: (p,q) is 2D, β is 4D
    # Pseudo-inverse loses information!
    print(f"\n⚠ Note: Projecting 2D (p,q) into 4D β loses information!")
    print(f"  β has {betas.shape[1]} dimensions")
    print(f"  (p,q) has 2 dimensions")
    print(f"  Rank of W_beta_to_pq: {np.linalg.matrix_rank(W_beta_to_pq)}")

    passed = np.mean(beta_errors) < 0.5
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'} (threshold: 0.5)")

    return {'mean_error': np.mean(beta_errors), 'passed': passed}


def test_4_simple_signal_forecast():
    """
    Forecast a pure sine wave.
    signal = cos(ωt)

    This has P = const, Q = ωt - the simplest possible case.
    """
    print("\n" + "=" * 70)
    print("TEST 4: SIMPLE SINE WAVE FORECAST")
    print("=" * 70)

    omega = 1.0
    n_points = 128
    dt = 0.01

    # Pure sine wave (E=1, theta0=0)
    t = np.arange(n_points) * dt
    z_simple = np.cos(omega * t) + 1j * np.sin(omega * t)  # e^{iωt}

    print(f"\nSimple signal: z = e^{{iωt}} (pure rotation)")
    print(f"  |z| = {np.abs(z_simple[0]):.4f} (constant)")
    print(f"  arg(z[0]) = {np.angle(z_simple[0]):.4f}")
    print(f"  arg(z[-1]) = {np.angle(z_simple[-1]):.4f}")

    # HST roundtrip
    coeffs = hst_forward_pywt(z_simple, J=3, wavelet_name='db8')
    z_rec = hst_inverse_pywt(coeffs, original_length=n_points)

    hst_error = np.linalg.norm(z_rec - z_simple) / np.linalg.norm(z_simple)
    print(f"\nHST roundtrip error: {hst_error:.2e}")

    # Check phase preservation
    phase_orig = np.unwrap(np.angle(z_simple))
    phase_rec = np.unwrap(np.angle(z_rec))
    phase_error = np.mean(np.abs(phase_rec - phase_orig))
    print(f"Phase error: {phase_error:.4f} rad")

    # Forecast test: predict z at T = 2π (one period)
    T = 2 * np.pi
    z_T_true = np.cos(omega * (t + T)) + 1j * np.sin(omega * (t + T))

    # For this simple signal, forecasting should just phase-shift
    # z(t+T) = e^{iω(t+T)} = e^{iωT} * e^{iωt} = e^{iωT} * z(t)
    z_T_pred = z_simple * np.exp(1j * omega * T)

    forecast_error = np.linalg.norm(z_T_pred - z_T_true) / np.linalg.norm(z_T_true)
    print(f"\nForecast T={T/(2*np.pi):.1f} periods:")
    print(f"  Error: {forecast_error:.2e}")

    passed = hst_error < 1e-6 and forecast_error < 1e-10
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}")

    return {'hst_error': hst_error, 'forecast_error': forecast_error, 'passed': passed}


def test_5_identify_bottleneck():
    """
    Identify exact bottleneck in the full pipeline.
    """
    print("\n" + "=" * 70)
    print("TEST 5: BOTTLENECK IDENTIFICATION")
    print("=" * 70)

    np.random.seed(12345)
    omega = 1.0
    n_points = 128

    # Generate test signal
    E = 2.0
    theta0 = 0.5
    z_orig = generate_sho_signal(omega=omega, n_points=n_points, E=E, theta0=theta0)

    print(f"\nOriginal signal: E={E}, θ₀={theta0:.2f}")

    # Stage 1: HST forward
    coeffs = hst_forward_pywt(z_orig, J=3, wavelet_name='db8')
    z_after_hst = hst_inverse_pywt(coeffs, original_length=n_points)
    err_hst = np.linalg.norm(z_after_hst - z_orig) / np.linalg.norm(z_orig)
    print(f"\n[Stage 1] HST → iHST: error = {err_hst:.2e}")

    # Stage 2: Add PCA
    # Need to fit on multiple signals first
    signals = [generate_sho_signal(omega=omega, n_points=n_points,
                                   E=np.random.uniform(0.5, 4.5),
                                   theta0=np.random.uniform(0, 2*np.pi))
               for _ in range(100)]

    rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=n_points)
    rom.fit(signals, extract_windows=False)

    beta = rom.transform(z_orig)
    z_after_pca = rom.inverse_transform(beta, original_length=n_points)
    err_pca = np.linalg.norm(z_after_pca - z_orig) / np.linalg.norm(z_orig)
    print(f"[Stage 2] HST → PCA → iPCA → iHST: error = {err_pca:.4f}")

    # Stage 3: Add β → (p,q) mapping
    pq_true = []
    for i, z in enumerate(signals):
        # Get (p,q) at center
        E_i = np.random.uniform(0.5, 4.5)  # This is wrong - we need actual p,q
        # Actually compute from signal
        center = n_points // 2
        q_c = np.real(z[center])
        p_c = np.imag(z[center]) * omega
        pq_true.append([p_c, q_c])

    pq_true = np.array(pq_true)
    betas = np.array([rom.transform(z) for z in signals])

    W_beta_to_pq, _, _, _ = np.linalg.lstsq(betas, pq_true, rcond=None)
    W_pq_to_beta = np.linalg.pinv(W_beta_to_pq)

    # Test on original signal
    beta_orig = rom.transform(z_orig)
    pq_from_beta = beta_orig @ W_beta_to_pq
    beta_reconstructed = pq_from_beta @ W_pq_to_beta

    # This reconstructed beta → signal
    # Need to use rom's internal structure
    features_rec = rom.pca.inverse_transform(beta_reconstructed.reshape(1, -1))[0]
    coeffs_rec = rom._unflatten_hst(features_rec, rom.reference_coeffs_)
    z_after_pq = hst_inverse_pywt(coeffs_rec, original_length=n_points)

    err_pq = np.linalg.norm(z_after_pq - z_orig) / np.linalg.norm(z_orig)
    print(f"[Stage 3] Full path with β↔(p,q): error = {err_pq:.4f}")

    # Summary
    print("\n" + "-" * 40)
    print("ERROR ACCUMULATION:")
    print("-" * 40)
    print(f"  HST alone:        {err_hst:.2e}")
    print(f"  + PCA:            {err_pca:.4f}")
    print(f"  + β↔(p,q) pseudo: {err_pq:.4f}")

    if err_pca > 0.3:
        print("\n⚠ BOTTLENECK: PCA truncation (need more components)")
    elif err_pq > err_pca * 2:
        print("\n⚠ BOTTLENECK: β↔(p,q) pseudo-inverse")
    else:
        print("\n✓ No single dominant bottleneck")

    return {
        'err_hst': err_hst,
        'err_pca': err_pca,
        'err_pq': err_pq
    }


def run_all_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "=" * 70)
    print("INVERSE PATH DIAGNOSTICS")
    print("=" * 70)

    results = {}

    results['hst'] = test_1_hst_roundtrip()
    results['pca'] = test_2_pca_roundtrip()
    results['beta_pq'] = test_3_beta_pq_roundtrip()
    results['simple'] = test_4_simple_signal_forecast()
    results['bottleneck'] = test_5_identify_bottleneck()

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print("\n| Component | Error | Status |")
    print("|-----------|-------|--------|")
    print(f"| HST roundtrip | {results['hst']['error']:.2e} | {'✓' if results['hst']['passed'] else '✗'} |")
    print(f"| PCA roundtrip | {results['pca']['mean_error']:.4f} | {'✓' if results['pca']['passed'] else '✗'} |")
    print(f"| β↔(p,q) pseudo | {results['beta_pq']['mean_error']:.4f} | {'✓' if results['beta_pq']['passed'] else '✗'} |")
    print(f"| Simple forecast | {results['simple']['hst_error']:.2e} | {'✓' if results['simple']['passed'] else '✗'} |")

    print("\n" + "-" * 70)
    print("BOTTLENECK ANALYSIS:")
    print("-" * 70)
    bn = results['bottleneck']
    print(f"  HST:        {bn['err_hst']:.2e} (should be ~0)")
    print(f"  + PCA:      {bn['err_pca']:.4f} (truncation loss)")
    print(f"  + β↔(p,q):  {bn['err_pq']:.4f} (pseudo-inverse loss)")

    # Identify main culprit
    if bn['err_pca'] > 0.3:
        print("\n→ MAIN ISSUE: PCA truncation loses too much information")
        print("  SOLUTION: Use more PCA components or train decoder")
    elif bn['err_pq'] > bn['err_pca'] * 1.5:
        print("\n→ MAIN ISSUE: β↔(p,q) pseudo-inverse is rank-deficient")
        print("  SOLUTION: Train MLP decoder β → signal directly")
    else:
        print("\n→ Error accumulates across stages")
        print("  SOLUTION: Train end-to-end decoder")

    return results


if __name__ == "__main__":
    results = run_all_diagnostics()
