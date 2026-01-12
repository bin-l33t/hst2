"""
The problem: Our features use MAGNITUDES only!
  |c|, std(|c|), mean(|c|²)

This throws away the PHASE of the complex coefficients!

The earlier r = 0.9999 came from using the complex signal directly.

Test: Can we recover Q by using complex phase information?
"""

import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from hst import hst_forward_pywt
from hamiltonian_systems import SimpleHarmonicOscillator, simulate_hamiltonian


def extract_magnitude_features(z, J=3):
    """Original: magnitude-only features (loses phase!)."""
    coeffs = hst_forward_pywt(z.real, J=J, wavelet_name='db8')
    features = []
    for c in coeffs['cD']:
        features.extend([np.mean(np.abs(c)), np.std(np.abs(c)), np.mean(np.abs(c)**2)])
    ca = coeffs['cA_final']
    features.extend([np.mean(np.abs(ca)), np.std(np.abs(ca)), np.mean(np.abs(ca)**2)])
    return np.array(features)


def extract_phase_features(z, J=3):
    """NEW: Include phase information from complex coefficients."""
    # HST on COMPLEX signal, not just real part
    z_complex = z.astype(np.complex128)

    # Apply to real and imaginary parts separately
    coeffs_real = hst_forward_pywt(z.real, J=J, wavelet_name='db8')
    coeffs_imag = hst_forward_pywt(z.imag, J=J, wavelet_name='db8')

    features = []

    for j in range(len(coeffs_real['cD'])):
        cD_real = coeffs_real['cD'][j]
        cD_imag = coeffs_imag['cD'][j]

        # Construct complex coefficient
        cD_complex = cD_real + 1j * cD_imag

        # Magnitude features (as before)
        features.extend([
            np.mean(np.abs(cD_complex)),
            np.std(np.abs(cD_complex)),
        ])

        # PHASE features (NEW!)
        phases = np.angle(cD_complex)
        features.extend([
            np.mean(np.cos(phases)),  # Circular mean (real part)
            np.mean(np.sin(phases)),  # Circular mean (imag part)
            np.std(phases),           # Phase dispersion
        ])

    # Final approximation
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


def extract_direct_phase(z):
    """Extract phase directly from the complex trajectory."""
    # The complex trajectory z = q + i*p/ω encodes phase as arg(z)
    # Use mean angle of the window
    phases = np.angle(z)
    mean_phase = np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
    return mean_phase


def test_phase_extraction():
    """Compare magnitude-only vs phase-aware features."""
    print("=" * 60)
    print("TEST: Magnitude-only vs Phase-aware Features")
    print("=" * 60)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Generate data
    energies = np.linspace(0.5, 3.0, 30)
    window_size = 512

    data_mag = []
    data_phase = []
    data_direct = []
    P_arr = []
    Q_arr = []

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=100, dt=0.01)

        P_true = E_actual / 1.0

        for _ in range(10):
            start = np.random.randint(0, len(z) - window_size)
            window = z[start:start+window_size]

            center = start + window_size // 2
            Q_true = np.arctan2(p[center], q[center])

            # Extract features
            feat_mag = extract_magnitude_features(window)
            feat_phase = extract_phase_features(window)
            direct_phase = extract_direct_phase(window)

            data_mag.append(feat_mag)
            data_phase.append(feat_phase)
            data_direct.append([direct_phase])
            P_arr.append(P_true)
            Q_arr.append(Q_true)

    X_mag = np.array(data_mag)
    X_phase = np.array(data_phase)
    X_direct = np.array(data_direct)
    P_arr = np.array(P_arr)
    Q_arr = np.array(Q_arr)

    Q_sin = np.sin(Q_arr)
    Q_cos = np.cos(Q_arr)

    print(f"\nFeature dimensions:")
    print(f"  Magnitude-only: {X_mag.shape[1]}")
    print(f"  Phase-aware:    {X_phase.shape[1]}")
    print(f"  Direct phase:   {X_direct.shape[1]}")

    # Test 1: Direct phase correlation
    print("\n--- Direct Phase from Complex Signal ---")
    direct_sin = np.sin(X_direct.flatten())
    direct_cos = np.cos(X_direct.flatten())

    r_sin_direct, _ = pearsonr(direct_sin, Q_sin)
    r_cos_direct, _ = pearsonr(direct_cos, Q_cos)

    print(f"r(sin(phase_direct), sin(Q_true)) = {r_sin_direct:.4f}")
    print(f"r(cos(phase_direct), cos(Q_true)) = {r_cos_direct:.4f}")

    # Test 2: Train MLPs on different feature sets
    n = len(P_arr)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    def normalize(X):
        return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    P_mean, P_std = P_arr.mean(), P_arr.std()
    P_norm = (P_arr - P_mean) / P_std

    def train_and_eval(X, name):
        X_norm = normalize(X)

        model = nn.Sequential(
            nn.Linear(X.shape[1], 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)  # P, sin(Q), cos(Q)
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_t = torch.tensor(X_norm[train_idx], dtype=torch.float32)
        y_t = torch.tensor(np.column_stack([
            P_norm[train_idx],
            Q_sin[train_idx],
            Q_cos[train_idx]
        ]), dtype=torch.float32)

        for _ in range(3000):
            pred = model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test = torch.tensor(X_norm[test_idx], dtype=torch.float32)
            pred = model(X_test).numpy()

        r_P, _ = pearsonr(pred[:, 0], P_norm[test_idx])
        r_sin, _ = pearsonr(pred[:, 1], Q_sin[test_idx])
        r_cos, _ = pearsonr(pred[:, 2], Q_cos[test_idx])

        return r_P, r_sin, r_cos

    print("\n--- MLP Results ---")
    print(f"{'Features':>20} {'r(P)':>10} {'r(sin(Q))':>12} {'r(cos(Q))':>12}")
    print("-" * 58)

    np.random.seed(42)
    r_P_mag, r_sin_mag, r_cos_mag = train_and_eval(X_mag, "Magnitude-only")
    print(f"{'Magnitude-only':>20} {r_P_mag:>10.3f} {r_sin_mag:>12.3f} {r_cos_mag:>12.3f}")

    np.random.seed(42)
    r_P_phase, r_sin_phase, r_cos_phase = train_and_eval(X_phase, "Phase-aware")
    print(f"{'Phase-aware':>20} {r_P_phase:>10.3f} {r_sin_phase:>12.3f} {r_cos_phase:>12.3f}")

    # Test 3: Combine direct phase with magnitude features
    X_combined = np.column_stack([X_mag, np.sin(X_direct), np.cos(X_direct)])
    np.random.seed(42)
    r_P_comb, r_sin_comb, r_cos_comb = train_and_eval(X_combined, "Mag + Direct Phase")
    print(f"{'Mag + Direct Phase':>20} {r_P_comb:>10.3f} {r_sin_comb:>12.3f} {r_cos_comb:>12.3f}")

    print("\n--- Summary ---")
    if r_sin_comb > 0.8 and r_cos_comb > 0.8:
        print("\033[92mCONFIRMED: Adding direct phase recovers Q information!\033[0m")
        print("The HST magnitude features encode P, the complex phase encodes Q.")
    elif r_sin_phase > r_sin_mag + 0.2:
        print("\033[93mPARTIAL: Phase-aware features help but not perfect\033[0m")
    else:
        print("\033[91mPhase features don't improve Q prediction\033[0m")

    return {
        'direct': (r_sin_direct, r_cos_direct),
        'mag': (r_P_mag, r_sin_mag, r_cos_mag),
        'phase': (r_P_phase, r_sin_phase, r_cos_phase),
        'combined': (r_P_comb, r_sin_comb, r_cos_comb)
    }


def test_what_earlier_test_used():
    """Reproduce the earlier r = 0.9999 result."""
    print("\n" + "=" * 60)
    print("REPRODUCING EARLIER r = 0.9999 RESULT")
    print("=" * 60)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Generate trajectory
    E = 1.0
    q0, p0 = sho.initial_condition_for_energy(E)
    t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=100, dt=0.01)

    # The complex z = q + i*p/ω directly encodes phase as arg(z)
    # This is NOT the same as HST coefficients!

    phase_from_z = np.angle(z)
    phase_true = np.arctan2(p, q)

    r_direct, _ = pearsonr(phase_from_z, phase_true)
    print(f"\nr(arg(z), arctan2(p,q)) = {r_direct:.6f}")

    # The earlier test used HST coefficients, not raw z
    # Let's check what it actually computed
    print("\nThe earlier r = 0.9999 was for:")
    print("  r(Q_HST, θ_true)")
    print("where Q_HST came from some complex coefficient's phase")

    # Apply HST and check coefficient phases
    window_size = 512
    start = len(z) // 2 - window_size // 2
    window = z[start:start+window_size]

    coeffs = hst_forward_pywt(window.real, J=3, wavelet_name='db8')

    # Mean phase of each coefficient
    print("\nMean phase of HST coefficients vs true Q:")
    Q_true = np.arctan2(p[start + window_size//2], q[start + window_size//2])

    for j, cD in enumerate(coeffs['cD']):
        mean_phase = np.arctan2(np.mean(np.sin(np.angle(cD.astype(complex)))),
                                np.mean(np.cos(np.angle(cD.astype(complex)))))
        print(f"  cD[{j}]: mean_phase = {mean_phase:.4f}, Q_true = {Q_true:.4f}")


if __name__ == "__main__":
    np.random.seed(42)

    # Test phase extraction
    results = test_phase_extraction()

    # Reproduce earlier result
    test_what_earlier_test_used()
