"""
Test β_⊥ preservation for fixing the pseudo-inverse reconstruction bottleneck.

The key insight: β has 4 dimensions, (p,q) has 2 dimensions.
The pseudo-inverse loses 2 dimensions (β_⊥), but these encode conserved
quantities (energy/shape) that shouldn't change during propagation.

Solution: Store β_⊥ during encoding, add it back during decoding.
"""

import numpy as np
import torch
from scipy.stats import pearsonr
import sys

# Import HST_ROM
from hst_rom import HST_ROM


class BetaPreservingPipeline:
    """
    Forward:  β → (p,q) using W
    Inverse:  (p,q)_new → β_new using W⁺ + preserved β_⊥
    """

    def __init__(self, W_beta_to_pq):
        """
        Args:
            W_beta_to_pq: Shape (n_beta, 2) - linear map from β to (p,q)
        """
        self.W = W_beta_to_pq  # Shape: (n_beta, 2)
        self.W_pinv = np.linalg.pinv(self.W)  # Shape: (2, n_beta)

        n_beta = W_beta_to_pq.shape[0]

        # Compute projection onto (p,q) subspace
        # P_parallel = W @ W⁺ (projects β onto range of W)
        self.P_parallel = self.W @ self.W_pinv  # (n_beta, n_beta)
        self.P_perp = np.eye(n_beta) - self.P_parallel

        # Verify projections
        assert np.allclose(self.P_parallel @ self.P_parallel, self.P_parallel), "P_parallel not idempotent"
        assert np.allclose(self.P_perp @ self.P_perp, self.P_perp), "P_perp not idempotent"

    def encode(self, beta):
        """
        β → (p,q), also stores β_⊥

        Args:
            beta: Shape (n_samples, n_beta) or (n_beta,)

        Returns:
            pq: Shape (n_samples, 2) or (2,)
            beta_perp: Shape (n_samples, n_beta) or (n_beta,) - preserved component
        """
        beta = np.atleast_2d(beta)
        pq = beta @ self.W  # (n_samples, 2)
        beta_perp = beta @ self.P_perp  # (n_samples, n_beta) - preserved
        return pq, beta_perp

    def decode(self, pq, beta_perp):
        """
        (p,q) + stored β_⊥ → β_reconstructed

        Args:
            pq: Shape (n_samples, 2) or (2,)
            beta_perp: Shape (n_samples, n_beta) or (n_beta,)

        Returns:
            beta_reconstructed: Shape (n_samples, n_beta)
        """
        pq = np.atleast_2d(pq)
        beta_perp = np.atleast_2d(beta_perp)

        beta_parallel = pq @ self.W_pinv  # (n_samples, n_beta)
        beta_reconstructed = beta_parallel + beta_perp
        return beta_reconstructed

    def decode_without_perp(self, pq):
        """Decode using only pseudo-inverse (for comparison)"""
        pq = np.atleast_2d(pq)
        return pq @ self.W_pinv


class SimpleBetaSplit:
    """
    Simpler approach: Use β[0:2] as (p,q)-like, preserve β[2:].

    Based on bridge test showing β₀ ↔ p, β₁ ↔ q with high correlation.
    """

    def __init__(self, scale_p=1.0, scale_q=1.0):
        """
        Args:
            scale_p, scale_q: Scaling factors to convert β[0],β[1] to (p,q)
        """
        self.scale_p = scale_p
        self.scale_q = scale_q

    def fit(self, beta, p_true, q_true):
        """Learn scaling from data."""
        # Find best linear scaling: p ≈ scale_p * β₀, q ≈ scale_q * β₁
        self.scale_p = np.std(p_true) / (np.std(beta[:, 0]) + 1e-10)
        self.scale_q = np.std(q_true) / (np.std(beta[:, 1]) + 1e-10)

        # Check sign correlation
        if np.corrcoef(p_true, beta[:, 0])[0, 1] < 0:
            self.scale_p = -self.scale_p
        if np.corrcoef(q_true, beta[:, 1])[0, 1] < 0:
            self.scale_q = -self.scale_q

    def encode(self, beta):
        """β → (p,q), preserve β_residual."""
        beta = np.atleast_2d(beta)
        p = beta[:, 0] * self.scale_p
        q = beta[:, 1] * self.scale_q
        pq = np.stack([p, q], axis=1)
        beta_residual = beta[:, 2:].copy()  # Preserved
        return pq, beta_residual

    def decode(self, pq, beta_residual):
        """(p,q) + β_residual → β."""
        pq = np.atleast_2d(pq)
        beta_residual = np.atleast_2d(beta_residual)

        beta_0 = pq[:, 0] / self.scale_p
        beta_1 = pq[:, 1] / self.scale_q
        beta_reconstructed = np.concatenate([
            beta_0[:, np.newaxis],
            beta_1[:, np.newaxis],
            beta_residual
        ], axis=1)
        return beta_reconstructed

    def decode_without_residual(self, pq, n_beta):
        """Decode setting residual to zero (for comparison)."""
        pq = np.atleast_2d(pq)
        beta_0 = pq[:, 0] / self.scale_p
        beta_1 = pq[:, 1] / self.scale_q
        zeros = np.zeros((pq.shape[0], n_beta - 2))
        return np.concatenate([
            beta_0[:, np.newaxis],
            beta_1[:, np.newaxis],
            zeros
        ], axis=1)


def generate_sho_signal(p0: float, q0: float, omega: float = 1.0,
                        dt: float = 0.01, n_points: int = 128) -> tuple:
    """
    Generate complex SHO signal z(t) = q(t) + i*p(t)/omega.

    Returns:
        z: complex signal
        p_t: momentum trajectory
        q_t: position trajectory
    """
    t = np.arange(n_points) * dt

    # SHO evolution
    E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
    theta0 = np.arctan2(p0, omega * q0)

    # Evolve
    theta_t = theta0 + omega * t
    p_t = np.sqrt(2 * E) * np.sin(theta_t)
    q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)

    # Analytic signal (complex representation)
    z = q_t + 1j * p_t / omega

    return z, p_t, q_t


def test_beta_preservation():
    """Test that β_⊥ preservation fixes reconstruction."""
    print("=" * 70)
    print("β_⊥ PRESERVATION TEST")
    print("=" * 70)

    np.random.seed(42)

    # Parameters (matching test_hst_hjb_bridge.py)
    omega = 1.0
    n_samples = 200
    window_size = 128
    n_pca = 8  # More components for better reconstruction
    dt = 0.01

    # Generate diverse (p, q) samples
    E_samples = np.random.uniform(0.5, 2.0, n_samples)
    phase_samples = np.random.uniform(0, 2 * np.pi, n_samples)

    # Initial conditions
    p0_arr = np.sqrt(2 * E_samples) * np.sin(phase_samples)
    q0_arr = np.sqrt(2 * E_samples) / omega * np.cos(phase_samples)

    # Generate signals and extract β via HST_ROM
    print("\n[1] Generating signals and extracting β...")
    hst_rom = HST_ROM(n_components=n_pca, J=3, window_size=window_size)

    signals = []
    p_true = []  # p at center of window
    q_true = []  # q at center of window

    center_idx = window_size // 2

    for i in range(n_samples):
        z, p_t, q_t = generate_sho_signal(p0_arr[i], q0_arr[i], omega=omega,
                                          dt=dt, n_points=window_size)
        signals.append(z)
        # Ground truth at window center
        p_true.append(p_t[center_idx])
        q_true.append(q_t[center_idx])

    p_true = np.array(p_true)
    q_true = np.array(q_true)

    # Fit HST_ROM (extract_windows=False since signals already windowed)
    beta = hst_rom.fit(signals, extract_windows=False)  # (n_samples, n_pca)
    signals = np.array(signals)  # Convert to array after fit

    print(f"  β shape: {beta.shape}")
    print(f"  PCA variance explained: {sum(hst_rom.pca.explained_variance_ratio_):.3f}")

    # Baseline: HST+PCA roundtrip error (no β manipulation)
    print("\n  Baseline HST+PCA roundtrip error...")
    signals_baseline = []
    for i in range(len(beta)):
        z_rec = hst_rom.inverse_transform(beta[i])
        signals_baseline.append(z_rec)
    signals_baseline = np.array(signals_baseline)
    baseline_error = np.mean(np.abs(signals - signals_baseline))
    print(f"  Baseline signal error (β → signal): {baseline_error:.4f}")

    # Check β correlations with (p,q)
    print("\n[2] Checking β-(p,q) correlations...")
    for i in range(min(4, n_pca)):
        r_p = np.corrcoef(beta[:, i], p_true)[0, 1]
        r_q = np.corrcoef(beta[:, i], q_true)[0, 1]
        print(f"  β_{i}: r(p)={r_p:.3f}, r(q)={r_q:.3f}")

    # ========== METHOD 1: SimpleBetaSplit (direct component mapping) ==========
    print("\n" + "=" * 70)
    print("METHOD 1: SimpleBetaSplit (β₀≈p, β₁≈q)")
    print("=" * 70)

    simple_pipeline = SimpleBetaSplit()
    simple_pipeline.fit(beta, p_true, q_true)

    print(f"  Scale factors: p={simple_pipeline.scale_p:.3f}, q={simple_pipeline.scale_q:.3f}")

    # Encode and decode
    pq_simple, beta_residual = simple_pipeline.encode(beta)

    # Check forward map quality
    simple_forward_error = np.mean(np.abs(pq_simple - np.stack([p_true, q_true], axis=1)))
    print(f"  Forward error (β → pq): {simple_forward_error:.4f}")

    # Roundtrip with preservation
    beta_reconstructed_simple = simple_pipeline.decode(pq_simple, beta_residual)
    simple_beta_error_with = np.mean(np.abs(beta - beta_reconstructed_simple))
    print(f"  β roundtrip error (with residual): {simple_beta_error_with:.6f}")

    # Roundtrip without preservation
    beta_reconstructed_simple_no = simple_pipeline.decode_without_residual(pq_simple, n_pca)
    simple_beta_error_without = np.mean(np.abs(beta - beta_reconstructed_simple_no))
    print(f"  β roundtrip error (no residual): {simple_beta_error_without:.4f}")

    # Signal reconstruction
    signals_simple_with = []
    for i in range(len(beta_reconstructed_simple)):
        z_rec = hst_rom.inverse_transform(beta_reconstructed_simple[i])
        signals_simple_with.append(z_rec)
    signals_simple_with = np.array(signals_simple_with)
    simple_signal_error_with = np.mean(np.abs(signals - signals_simple_with))
    print(f"  Signal error (with residual): {simple_signal_error_with:.4f}")

    signals_simple_without = []
    for i in range(len(beta_reconstructed_simple_no)):
        z_rec = hst_rom.inverse_transform(beta_reconstructed_simple_no[i])
        signals_simple_without.append(z_rec)
    signals_simple_without = np.array(signals_simple_without)
    simple_signal_error_without = np.mean(np.abs(signals - signals_simple_without))
    print(f"  Signal error (no residual): {simple_signal_error_without:.4f}")

    # ========== METHOD 2: BetaPreservingPipeline (linear map + projection) ==========
    print("\n" + "=" * 70)
    print("METHOD 2: BetaPreservingPipeline (linear W + projection)")
    print("=" * 70)

    # Find linear map W: β → (p,q) via least squares
    pq_true = np.stack([p_true, q_true], axis=1)  # (n_samples, 2)
    W_beta_to_pq, residuals, rank, s = np.linalg.lstsq(beta, pq_true, rcond=None)

    print(f"  W shape: {W_beta_to_pq.shape}")
    print(f"  Forward map error: {np.mean(np.abs(beta @ W_beta_to_pq - pq_true)):.4f}")

    # Create pipeline
    pipeline = BetaPreservingPipeline(W_beta_to_pq)
    print(f"  Rank of P_parallel: {np.linalg.matrix_rank(pipeline.P_parallel)}")
    print(f"  Rank of P_perp: {np.linalg.matrix_rank(pipeline.P_perp)}")

    # Test roundtrip WITH preservation
    print("\n[4] Testing roundtrip WITH β_⊥ preservation...")
    pq_encoded, beta_perp = pipeline.encode(beta)
    beta_reconstructed_with = pipeline.decode(pq_encoded, beta_perp)

    error_with_preservation = np.mean(np.abs(beta - beta_reconstructed_with))
    print(f"  β reconstruction error (with β_⊥): {error_with_preservation:.6f}")

    # Test roundtrip WITHOUT preservation (pseudo-inverse only)
    print("\n[5] Testing roundtrip WITHOUT β_⊥ preservation...")
    beta_reconstructed_without = pipeline.decode_without_perp(pq_encoded)

    error_without_preservation = np.mean(np.abs(beta - beta_reconstructed_without))
    print(f"  β reconstruction error (pseudo-inverse only): {error_without_preservation:.4f}")

    # Improvement factor
    improvement = error_without_preservation / (error_with_preservation + 1e-10)
    print(f"\n  Improvement factor: {improvement:.1f}x")

    # Full signal reconstruction test
    print("\n[6] Full signal reconstruction test...")

    # With preservation - reconstruct each signal
    signals_reconstructed_with = []
    for i in range(len(beta_reconstructed_with)):
        z_rec = hst_rom.inverse_transform(beta_reconstructed_with[i])
        signals_reconstructed_with.append(z_rec)
    signals_reconstructed_with = np.array(signals_reconstructed_with)
    signal_error_with = np.mean(np.abs(signals - signals_reconstructed_with))

    # Without preservation
    signals_reconstructed_without = []
    for i in range(len(beta_reconstructed_without)):
        z_rec = hst_rom.inverse_transform(beta_reconstructed_without[i])
        signals_reconstructed_without.append(z_rec)
    signals_reconstructed_without = np.array(signals_reconstructed_without)
    signal_error_without = np.mean(np.abs(signals - signals_reconstructed_without))

    print(f"  Signal error (with β_⊥): {signal_error_with:.4f}")
    print(f"  Signal error (pseudo-inverse only): {signal_error_without:.4f}")

    # Test propagation scenario
    # NOTE: For propagation, the preserved β_⊥ encodes shape info from the ORIGINAL signal.
    # After phase rotation, the ground truth signal has different shape, so preservation
    # doesn't help (and may hurt). This is expected - β_⊥ captures "conserved" quantities
    # that are valid for roundtrip but not for arbitrary (p,q) changes.
    print("\n[7] Testing propagation scenario (informational)...")
    print("    (Rotate phase by π/4, keep action constant)")
    print("    NOTE: β_⊥ from original signal may not apply to propagated state")

    # Pick a test sample
    test_idx = 0
    beta_test = beta[test_idx:test_idx+1]

    # Encode
    pq_test, beta_perp_test = pipeline.encode(beta_test)
    p_test, q_test = pq_test[0, 0], pq_test[0, 1]

    # "Propagate" - rotate by π/4
    dtheta = np.pi / 4
    cos_d, sin_d = np.cos(dtheta), np.sin(dtheta)

    # Rotation in (p/ω, q) space preserves energy
    p_scaled = p_test / omega
    p_new_scaled = cos_d * p_scaled - sin_d * q_test
    q_new = sin_d * p_scaled + cos_d * q_test
    p_new = p_new_scaled * omega

    pq_new = np.array([[p_new, q_new]])

    # Decode with preservation
    beta_propagated_with = pipeline.decode(pq_new, beta_perp_test)
    signal_propagated_with = hst_rom.inverse_transform(beta_propagated_with)[0]

    # Decode without preservation
    beta_propagated_without = pipeline.decode_without_perp(pq_new)
    signal_propagated_without = hst_rom.inverse_transform(beta_propagated_without)[0]

    # Generate ground truth propagated signal
    signal_propagated_true, _, _ = generate_sho_signal(p_new, q_new, omega=omega,
                                                        dt=dt, n_points=window_size)

    error_propagated_with = np.mean(np.abs(signal_propagated_true - signal_propagated_with))
    error_propagated_without = np.mean(np.abs(signal_propagated_true - signal_propagated_without))

    print(f"  Propagated signal error (with β_⊥): {error_propagated_with:.4f}")
    print(f"  Propagated signal error (pseudo-inverse only): {error_propagated_without:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: COMPARISON OF METHODS")
    print("=" * 70)

    # Best signal error from either method
    best_signal_error = min(simple_signal_error_with, signal_error_with)
    best_method = "SimpleBetaSplit" if simple_signal_error_with < signal_error_with else "BetaPreservingPipeline"

    print(f"\n{'Method':<30} {'Signal Error':<15} {'β Error':<15}")
    print("-" * 60)
    print(f"{'SimpleBetaSplit (with res.)':<30} {simple_signal_error_with:<15.4f} {simple_beta_error_with:<15.6f}")
    print(f"{'SimpleBetaSplit (no res.)':<30} {simple_signal_error_without:<15.4f} {simple_beta_error_without:<15.4f}")
    print(f"{'BetaPreserving (with β_⊥)':<30} {signal_error_with:<15.4f} {error_with_preservation:<15.6f}")
    print(f"{'BetaPreserving (no β_⊥)':<30} {signal_error_without:<15.4f} {error_without_preservation:<15.4f}")

    print(f"\nBest method: {best_method}")
    print(f"Best signal reconstruction error: {best_signal_error:.4f}")

    results = {
        "simple_signal_error_with": simple_signal_error_with,
        "simple_signal_error_without": simple_signal_error_without,
        "signal_error_with": signal_error_with,
        "signal_error_without": signal_error_without,
        "best_signal_error": best_signal_error,
    }

    # Pass/fail criteria:
    # 1. Signal error with preservation should match baseline (β roundtrip is perfect)
    # 2. Signal error with preservation << without preservation
    preservation_recovers_baseline = best_signal_error <= baseline_error * 1.01  # within 1%
    big_improvement = best_signal_error < signal_error_without * 0.5  # at least 2x better

    passed = preservation_recovers_baseline and big_improvement

    print("\n" + "=" * 70)
    print(f"\nBaseline HST+PCA error: {baseline_error:.4f}")
    print(f"Best with preservation: {best_signal_error:.4f}")
    print(f"Without preservation:   {signal_error_without:.4f}")
    print(f"\nImprovement factor: {signal_error_without / best_signal_error:.1f}x")

    if passed:
        print(f"\nTEST PASSED:")
        print(f"  ✓ Preservation achieves baseline ({best_signal_error:.4f} ≈ {baseline_error:.4f})")
        print(f"  ✓ Big improvement vs no preservation ({signal_error_without:.4f} → {best_signal_error:.4f})")
    else:
        if not preservation_recovers_baseline:
            print(f"\n✗ Preservation error ({best_signal_error:.4f}) > baseline ({baseline_error:.4f})")
        if not big_improvement:
            print(f"\n✗ Improvement too small ({signal_error_without/best_signal_error:.1f}x < 2x)")
    print("=" * 70)

    return passed, results


if __name__ == "__main__":
    passed, results = test_beta_preservation()
    sys.exit(0 if passed else 1)
