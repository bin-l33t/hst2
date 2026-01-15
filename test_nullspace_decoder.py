"""
Test the HybridNullspaceDecoder for β reconstruction and forecasting.

Success criteria:
1. Forecast error ratio < 2.0 on TEST set (out-of-sample)
2. Nullspace constraint violation < 1e-3
3. Roundtrip error reasonable (within 3x baseline)
"""

import numpy as np
import torch
import sys

from hst_rom import HST_ROM
from fixed_hjb_loss import ImprovedHJB_MLP, FixedHJBLoss, train_on_sho_fixed
from nullspace_decoder import HybridNullspaceDecoder, train_nullspace_decoder


def generate_sho_signal(p0: float, q0: float, omega: float = 1.0,
                        dt: float = 0.01, n_points: int = 128) -> tuple:
    """Generate complex SHO signal z(t) = q(t) + i*p(t)/omega."""
    t = np.arange(n_points) * dt

    E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
    theta0 = np.arctan2(p0, omega * q0)

    theta_t = theta0 + omega * t
    p_t = np.sqrt(2 * E) * np.sin(theta_t)
    q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)

    z = q_t + 1j * p_t / omega
    return z, p_t, q_t


def train_hjb_encoder(omega, epochs=500):
    """Train an ImprovedHJB_MLP encoder using the standard training."""
    encoder = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)

    # Use the standard training function from fixed_hjb_loss.py
    # This trains with conservation, evolution, and symplectic losses
    losses = train_on_sho_fixed(
        encoder,
        n_epochs=epochs,
        n_trajectories=100,
        omega=omega,
        dt=0.5,
        lr=1e-3,
        device='cpu'
    )

    return encoder


def verify_nullspace_constraint(decoder, p, q, W, threshold=1e-3):
    """
    Verify that the correction term lives in the nullspace of W.

    Check: ||(β_correction) @ W|| ≈ 0

    Args:
        decoder: HybridNullspaceDecoder instance
        p, q: Test points (torch tensors)
        W: Linear map β → (p,q), shape (n_beta, 2)
        threshold: Maximum allowed violation

    Returns:
        (passed, violation): Tuple of (bool, float)
    """
    W_tensor = torch.tensor(W, dtype=torch.float32)

    with torch.no_grad():
        # Get full β prediction
        beta_pred = decoder(p, q)

        # Get linear part only
        pq = torch.stack([p, q], dim=-1)
        beta_linear = pq @ decoder.W_pinv

        # Correction is the difference
        beta_correction = beta_pred - beta_linear

        # Check: correction projected back to (p,q) should be ~0
        pq_from_correction = beta_correction @ W_tensor  # Should be ~0

        # Relative violation
        constraint_violation = torch.norm(pq_from_correction) / (torch.norm(pq) + 1e-8)

        return constraint_violation.item() < threshold, constraint_violation.item()


def test_nullspace_decoder():
    """Test HybridNullspaceDecoder for roundtrip and forecasting."""
    print("=" * 70)
    print("NULLSPACE DECODER TEST (with train/test split)")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    omega = 1.0
    n_samples = 300
    window_size = 128
    n_pca = 8
    dt = 0.01

    # Train/test split (80/20)
    n_train = int(0.8 * n_samples)
    all_idx = np.arange(n_samples)
    np.random.shuffle(all_idx)
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train:]

    print(f"\n[0] Data split: {n_train} train, {len(test_idx)} test")

    # Generate ALL data first
    print("\n[1] Generating data...")
    E_samples = np.random.uniform(0.5, 2.0, n_samples)
    phase_samples = np.random.uniform(0, 2 * np.pi, n_samples)

    p0_arr = np.sqrt(2 * E_samples) * np.sin(phase_samples)
    q0_arr = np.sqrt(2 * E_samples) / omega * np.cos(phase_samples)

    signals = []
    p_true = []
    q_true = []
    center_idx = window_size // 2

    for i in range(n_samples):
        z, p_t, q_t = generate_sho_signal(p0_arr[i], q0_arr[i], omega=omega,
                                          dt=dt, n_points=window_size)
        signals.append(z)
        p_true.append(p_t[center_idx])
        q_true.append(q_t[center_idx])

    signals = np.array(signals)
    p_true = np.array(p_true)
    q_true = np.array(q_true)

    # Fit HST_ROM on TRAIN only
    print("\n[2] Fitting HST_ROM on TRAIN set...")
    hst_rom = HST_ROM(n_components=n_pca, J=3, window_size=window_size)
    train_signals = [signals[i] for i in train_idx]
    beta_train = hst_rom.fit(train_signals, extract_windows=False)

    # Transform test signals using fitted HST_ROM
    beta_test = np.array([hst_rom.transform(signals[i]) for i in test_idx])

    # Also get full beta for all samples (for baseline computation)
    beta_all = np.zeros((n_samples, n_pca))
    beta_all[train_idx] = beta_train
    beta_all[test_idx] = beta_test

    print(f"  β_train shape: {beta_train.shape}")
    print(f"  β_test shape: {beta_test.shape}")
    print(f"  PCA variance explained: {sum(hst_rom.pca.explained_variance_ratio_):.3f}")

    # Compute baseline error on TEST set (HST+PCA roundtrip)
    signals_baseline_test = []
    for i, idx in enumerate(test_idx):
        z_rec = hst_rom.inverse_transform(beta_test[i])
        signals_baseline_test.append(z_rec)
    signals_baseline_test = np.array(signals_baseline_test)
    baseline_error = np.mean(np.abs(signals[test_idx] - signals_baseline_test))
    print(f"  Baseline signal error (TEST): {baseline_error:.4f}")

    # Learn linear map W: β → (p,q) on TRAIN only
    print("\n[3] Learning β → (p,q) linear map on TRAIN set...")
    pq_train = np.stack([p_true[train_idx], q_true[train_idx]], axis=1)
    W_beta_to_pq, _, _, _ = np.linalg.lstsq(beta_train, pq_train, rcond=None)
    print(f"  W shape: {W_beta_to_pq.shape}")

    # Check forward map quality on TRAIN and TEST
    pq_pred_train = beta_train @ W_beta_to_pq
    forward_error_train = np.mean(np.abs(pq_pred_train - pq_train))

    pq_test = np.stack([p_true[test_idx], q_true[test_idx]], axis=1)
    pq_pred_test = beta_test @ W_beta_to_pq
    forward_error_test = np.mean(np.abs(pq_pred_test - pq_test))

    print(f"  Forward map error (TRAIN): {forward_error_train:.4f}")
    print(f"  Forward map error (TEST): {forward_error_test:.4f}")

    # Train HJB encoder (uses its own generated data, not our split)
    print("\n[4] Training HJB encoder...")
    hjb_encoder = train_hjb_encoder(omega, epochs=500)

    # Verify HJB encoder works on test data
    with torch.no_grad():
        p_test_t = torch.tensor(p_true[test_idx], dtype=torch.float32)
        q_test_t = torch.tensor(q_true[test_idx], dtype=torch.float32)
        P_test, Q_test = hjb_encoder.encode(p_test_t, q_test_t)

        # Check action conservation
        E_test_true = 0.5 * p_true[test_idx]**2 + 0.5 * omega**2 * q_true[test_idx]**2
        P_expected = E_test_true / omega
        P_corr = np.corrcoef(P_test.numpy(), P_expected)[0, 1]
        print(f"  HJB encoder P correlation (TEST): {P_corr:.4f}")

    # Create and train nullspace decoder on TRAIN only
    print("\n[5] Training nullspace decoder on TRAIN set...")
    decoder = HybridNullspaceDecoder(W_beta_to_pq, hjb_encoder, hidden_dim=128)

    p_train_t = torch.tensor(p_true[train_idx], dtype=torch.float32)
    q_train_t = torch.tensor(q_true[train_idx], dtype=torch.float32)

    losses = train_nullspace_decoder(
        decoder, p_true[train_idx], q_true[train_idx], beta_train,
        epochs=1500, lr=1e-3, verbose=True
    )

    # Test roundtrip on TEST set
    print("\n[6] Testing roundtrip on TEST set...")
    decoder.eval()
    with torch.no_grad():
        beta_pred_test = decoder(p_test_t, q_test_t).numpy()

    # β reconstruction error on TEST
    beta_error_test = np.mean(np.abs(beta_test - beta_pred_test))
    print(f"  β reconstruction error (TEST): {beta_error_test:.4f}")

    # Signal reconstruction on TEST
    signals_decoded_test = []
    for i in range(len(beta_pred_test)):
        z_rec = hst_rom.inverse_transform(beta_pred_test[i])
        signals_decoded_test.append(z_rec)
    signals_decoded_test = np.array(signals_decoded_test)
    roundtrip_error = np.mean(np.abs(signals[test_idx] - signals_decoded_test))
    print(f"  Signal roundtrip error (TEST): {roundtrip_error:.4f}")
    print(f"  Ratio to baseline: {roundtrip_error / baseline_error:.2f}x")

    # Correction magnitude diagnostic on TEST
    correction_mag = decoder.get_correction_magnitude(p_test_t, q_test_t)
    print(f"  Correction magnitude (TEST): {correction_mag:.3f}")

    # Verify nullspace constraint on TEST
    print("\n[7] Verifying nullspace constraint on TEST set...")
    nullspace_ok, nullspace_violation = verify_nullspace_constraint(
        decoder, p_test_t, q_test_t, W_beta_to_pq, threshold=1e-3
    )
    print(f"  Nullspace constraint violation: {nullspace_violation:.6f}")
    print(f"  Status: {'PASS' if nullspace_ok else 'FAIL'} (threshold: < 1e-3)")

    # Test forecasting on TEST set
    print("\n[8] Testing forecasting on TEST set...")
    test_periods = [0.1, 1.0, 10.0, 100.0]
    forecast_errors = {}

    for T_periods in test_periods:
        T = T_periods * 2 * np.pi / omega  # Convert periods to time

        errors = []
        for idx in test_idx:  # Use all test samples for each period
            # Original state
            p0, q0 = p_true[idx], q_true[idx]

            # Get (P, Q) from HJB encoder
            with torch.no_grad():
                p0_t = torch.tensor([p0], dtype=torch.float32)
                q0_t = torch.tensor([q0], dtype=torch.float32)
                P0, Q0 = hjb_encoder.encode(p0_t, q0_t)

            # Propagate: Q_T = Q_0 + ω*T (action P is conserved)
            Q_T = Q0 + omega * T

            # Decode propagated state
            with torch.no_grad():
                # Get (p_T, q_T) from propagated (P, Q)
                p_T, q_T = hjb_encoder.decode(P0, Q_T)

                # Get β from decoder
                beta_T = decoder(p_T, q_T).numpy()[0]

            # Reconstruct signal
            signal_pred = hst_rom.inverse_transform(beta_T)

            # Ground truth signal at time T
            E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
            theta0 = np.arctan2(p0, omega * q0)
            theta_T = theta0 + omega * T

            p_T_true = np.sqrt(2 * E) * np.sin(theta_T)
            q_T_true = np.sqrt(2 * E) / omega * np.cos(theta_T)

            # Generate ground truth signal at propagated state
            signal_true, _, _ = generate_sho_signal(
                p_T_true, q_T_true, omega=omega, dt=dt, n_points=window_size
            )

            error = np.mean(np.abs(signal_true - signal_pred))
            errors.append(error)

        forecast_errors[T_periods] = np.mean(errors)

    print(f"\n  Forecast errors by period (TEST set, n={len(test_idx)}):")
    for T_periods, error in forecast_errors.items():
        print(f"    T = {T_periods:5.1f} periods: error = {error:.4f}")

    # Compute error ratio (long/short)
    short_error = forecast_errors[0.1]
    long_error = forecast_errors[100.0]
    error_ratio = long_error / short_error
    print(f"\n  Error ratio (T=100 / T=0.1): {error_ratio:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (all metrics on TEST set)")
    print("=" * 70)

    # Primary criterion: forecasting stability (the actual use case)
    forecast_pass = error_ratio < 2.0

    # Secondary: roundtrip should be reasonable (within 3x baseline)
    roundtrip_reasonable = roundtrip_error <= baseline_error * 3.0

    # Tertiary: nullspace constraint
    nullspace_pass = nullspace_ok

    # Quaternary: correction magnitude
    correction_reasonable = correction_mag < 0.8

    print(f"\n1. Roundtrip error (TEST): {roundtrip_error:.4f}")
    print(f"   Baseline: {baseline_error:.4f}")
    print(f"   Ratio: {roundtrip_error / baseline_error:.2f}x")
    print(f"   Status: {'OK' if roundtrip_reasonable else 'WARN'} (threshold: ≤3.0x)")

    print(f"\n2. Forecast stability (PRIMARY, TEST):")
    print(f"   Error ratio: {error_ratio:.2f}x")
    print(f"   Status: {'PASS' if forecast_pass else 'FAIL'} (threshold: <2.0x)")

    print(f"\n3. Nullspace constraint:")
    print(f"   Violation: {nullspace_violation:.6f}")
    print(f"   Status: {'PASS' if nullspace_pass else 'FAIL'} (threshold: <1e-3)")

    print(f"\n4. Correction magnitude: {correction_mag:.3f}")
    print(f"   Status: {'OK' if correction_reasonable else 'WARN'}")

    # Primary criteria are forecast stability and nullspace constraint
    passed = forecast_pass and nullspace_pass and roundtrip_reasonable

    print("\n" + "=" * 70)
    if passed:
        print("TEST PASSED: Nullspace decoder enables forecasting!")
    else:
        reasons = []
        if not roundtrip_reasonable:
            reasons.append(f"roundtrip {roundtrip_error/baseline_error:.2f}x > 3.0x")
        if not forecast_pass:
            reasons.append(f"forecast ratio {error_ratio:.2f}x >= 2.0x")
        if not nullspace_pass:
            reasons.append(f"nullspace violation {nullspace_violation:.6f} >= 1e-3")
        print(f"TEST FAILED: {', '.join(reasons)}")
    print("=" * 70)

    return passed, {
        "n_train": n_train,
        "n_test": len(test_idx),
        "baseline_error": baseline_error,
        "roundtrip_error": roundtrip_error,
        "forecast_errors": forecast_errors,
        "error_ratio": error_ratio,
        "nullspace_violation": nullspace_violation,
        "correction_magnitude": correction_mag,
    }


if __name__ == "__main__":
    passed, results = test_nullspace_decoder()
    sys.exit(0 if passed else 1)
