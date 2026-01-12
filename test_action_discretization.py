"""
Test: Does HST+MLP Produce Discretized Action?

Glinsky's "Quantization from Determinism" Claim:
  - Periodic motion + complex structure → discrete action spectrum
  - "things can be quantized... from the fact that you are dealing with
     orbits or that you have a complex space and that you have a periodic variable"
  - HST as Wigner-Weyl transformation (phase-space QM formulation)

Test Design:
  1. Generate trajectories with FINELY-SPACED continuous energy values
  2. Learn P (action) for each using HST+MLP
  3. Histogram the learned P values
  4. Check: Do peaks appear at discrete values? Or continuous spread?

If Glinsky is right:
  - P should cluster at discrete values even when E is continuous
  - Clustering should relate to Bohr-Sommerfeld: J_n = (n + 1/2) * h_eff

If conventional:
  - P should vary continuously with E
  - No intrinsic discretization
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from hst import hst_forward_pywt, extract_features
from hamiltonian_systems import PendulumOscillator, SimpleHarmonicOscillator, simulate_hamiltonian


class ActionMLP(nn.Module):
    """MLP that learns action P from HST features."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def theoretical_action_sho(E, omega=1.0):
    """
    Theoretical action for SHO: I = E/ω

    Bohr-Sommerfeld: I_n = (n + 1/2)ℏ
    For classical system, ℏ_eff is determined by the system scale.
    """
    return E / omega


def theoretical_action_pendulum(E):
    """
    Theoretical action for pendulum (libration, E < 1).

    I = (8/π) * [E(k) - (1-k²)K(k)]
    where k² = (1+E)/2, K and E are complete elliptic integrals.
    """
    from scipy.special import ellipk, ellipe

    if E >= 1 or E <= -1:
        return np.nan

    k2 = (1 + E) / 2
    if k2 <= 0 or k2 >= 1:
        return np.nan

    k = np.sqrt(k2)
    K = ellipk(k2)
    E_ellip = ellipe(k2)

    # Action for pendulum libration
    I = (8/np.pi) * (E_ellip - (1 - k2) * K)
    return I


def test_sho_discretization():
    """
    Test 1: Simple Harmonic Oscillator

    SHO has linear I(E) = E/ω relationship.
    If HST discretizes, we should see clustering in learned P
    even though true I varies continuously.
    """
    print("=" * 60)
    print("TEST 1: SHO Action Discretization")
    print("=" * 60)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Generate MANY trajectories with finely-spaced energies
    n_trajectories = 100
    energies = np.linspace(0.1, 5.0, n_trajectories)  # Fine spacing

    print(f"\nGenerating {n_trajectories} trajectories with E ∈ [0.1, 5.0]")

    # Generate data
    all_features = []
    all_energies = []
    all_actions_true = []

    window_size = 512

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=0.01)

        # Extract features from middle window
        start = len(z) // 2 - window_size // 2
        if start < 0:
            start = 0
        feat = extract_features(z[start:start+window_size])

        all_features.append(feat)
        all_energies.append(E_actual)
        all_actions_true.append(theoretical_action_sho(E_actual))

    X = np.array(all_features)
    E_arr = np.array(all_energies)
    I_true = np.array(all_actions_true)

    # Normalize
    E_mean, E_std = E_arr.mean(), E_arr.std()

    # Train MLP
    print("\nTraining MLP to learn action from HST features...")
    model = ActionMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor((E_arr - E_mean) / E_std, dtype=torch.float32).unsqueeze(1)

    for epoch in range(2000):
        P_pred = model(X_t)
        loss = nn.MSELoss()(P_pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    # Get learned P values
    model.eval()
    with torch.no_grad():
        P_learned = model(X_t).numpy().flatten() * E_std + E_mean

    # Analyze distribution
    print("\n--- Discretization Analysis ---")

    # Check if P is continuous (linear with E) or discretized (clustered)
    r_P_E, _ = pearsonr(P_learned, E_arr)
    print(f"r(P_learned, E): {r_P_E:.4f}")

    # Histogram analysis
    n_bins = 20
    hist, bin_edges = np.histogram(P_learned, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Look for peaks (discretization)
    peaks, properties = find_peaks(hist, height=n_trajectories/n_bins * 0.5)

    print(f"\nHistogram peaks found: {len(peaks)}")
    if len(peaks) > 0:
        print(f"Peak locations (P values): {bin_centers[peaks]}")
        print(f"Peak heights: {hist[peaks]}")

    # Measure clustering coefficient
    # If discretized: high variance in histogram, few dominant bins
    # If continuous: uniform-ish histogram
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
    max_entropy = np.log(n_bins)  # Uniform distribution
    clustering_coeff = 1 - entropy / max_entropy

    print(f"\nClustering coefficient: {clustering_coeff:.4f}")
    print("  (0 = uniform/continuous, 1 = highly clustered/discrete)")

    # Check residuals from linear fit
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(E_arr, P_learned)
    P_linear = slope * E_arr + intercept
    residuals = P_learned - P_linear

    print(f"\nResiduals from linear fit:")
    print(f"  Mean: {np.mean(residuals):.4f}")
    print(f"  Std: {np.std(residuals):.4f}")
    print(f"  Max: {np.max(np.abs(residuals)):.4f}")

    # Verdict
    print("\n" + "=" * 60)
    if clustering_coeff > 0.3 and len(peaks) >= 3:
        print("\033[92mEVIDENCE FOR DISCRETIZATION!\033[0m")
        print(f"  Found {len(peaks)} peaks with clustering = {clustering_coeff:.2f}")
        status = 'DISCRETIZED'
    elif r_P_E > 0.99 and clustering_coeff < 0.15:
        print("\033[93mCONTINUOUS: P varies smoothly with E\033[0m")
        print("  No evidence of intrinsic discretization")
        status = 'CONTINUOUS'
    else:
        print("\033[93mMIXED: Some structure but not clear discretization\033[0m")
        status = 'MIXED'

    return status, P_learned, E_arr, clustering_coeff


def test_pendulum_discretization():
    """
    Test 2: Pendulum (nonlinear ω(E))

    More interesting because ω varies with E.
    Bohr-Sommerfeld predicts discrete action levels.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Pendulum Action Discretization")
    print("=" * 60)

    pendulum = PendulumOscillator()

    # Generate trajectories with finely-spaced energies (libration only)
    n_trajectories = 80
    energies = np.linspace(-0.9, 0.8, n_trajectories)

    print(f"\nGenerating {n_trajectories} trajectories with E ∈ [-0.9, 0.8]")

    all_features = []
    all_energies = []
    all_actions_true = []

    window_size = 512

    valid_count = 0
    for E in energies:
        if E >= 0.95 or E <= -0.95:
            continue

        try:
            q0, p0 = pendulum.initial_condition_for_energy(E)
            t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=80, dt=0.01)

            start = len(z) // 2 - window_size // 2
            if start < 0:
                start = 0
            feat = extract_features(z[start:start+window_size])

            I_true = theoretical_action_pendulum(E_actual)
            if np.isnan(I_true):
                continue

            all_features.append(feat)
            all_energies.append(E_actual)
            all_actions_true.append(I_true)
            valid_count += 1
        except:
            continue

    print(f"Valid trajectories: {valid_count}")

    X = np.array(all_features)
    E_arr = np.array(all_energies)
    I_true = np.array(all_actions_true)

    E_mean, E_std = E_arr.mean(), E_arr.std()

    # Train MLP
    print("\nTraining MLP...")
    model = ActionMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor((I_true - I_true.mean()) / (I_true.std() + 1e-8), dtype=torch.float32).unsqueeze(1)

    for epoch in range(2000):
        P_pred = model(X_t)
        loss = nn.MSELoss()(P_pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        P_learned = model(X_t).numpy().flatten() * I_true.std() + I_true.mean()

    # Analysis
    print("\n--- Discretization Analysis ---")

    r_P_I, _ = pearsonr(P_learned, I_true)
    print(f"r(P_learned, I_true): {r_P_I:.4f}")

    # Histogram
    n_bins = 15
    hist, bin_edges = np.histogram(P_learned, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(hist, height=len(P_learned)/n_bins * 0.5)

    print(f"\nHistogram peaks: {len(peaks)}")
    if len(peaks) > 0:
        print(f"Peak P values: {bin_centers[peaks]}")

    # Clustering coefficient
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
    clustering_coeff = 1 - entropy / np.log(n_bins)

    print(f"Clustering coefficient: {clustering_coeff:.4f}")

    # Compare with Bohr-Sommerfeld levels
    # For pendulum, I_n should be evenly spaced if Bohr-Sommerfeld applies
    print("\n--- Bohr-Sommerfeld Comparison ---")

    I_sorted = np.sort(I_true)
    delta_I = np.diff(I_sorted)
    print(f"True action range: [{I_true.min():.3f}, {I_true.max():.3f}]")
    print(f"Mean ΔI spacing: {np.mean(delta_I):.4f}")
    print(f"Std ΔI spacing: {np.std(delta_I):.4f}")

    # If Bohr-Sommerfeld, spacing should be ~ constant
    spacing_uniformity = np.std(delta_I) / (np.mean(delta_I) + 1e-10)
    print(f"Spacing uniformity (std/mean): {spacing_uniformity:.4f}")
    print("  (Low = uniform spacing like Bohr-Sommerfeld)")

    print("\n" + "=" * 60)
    if clustering_coeff > 0.25:
        print("\033[92mEVIDENCE FOR DISCRETIZATION!\033[0m")
        status = 'DISCRETIZED'
    else:
        print("\033[93mCONTINUOUS: No clear discretization\033[0m")
        status = 'CONTINUOUS'

    return status, P_learned, I_true, clustering_coeff


def test_hst_scale_discretization():
    """
    Test 3: Does HST's dyadic structure induce discretization?

    HST uses discrete wavelet scales (J levels).
    This could naturally create discrete "bands" in feature space.
    """
    print("\n" + "=" * 60)
    print("TEST 3: HST Scale Structure Analysis")
    print("=" * 60)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Generate trajectories at very fine energy spacing
    energies = np.linspace(0.5, 2.0, 50)

    all_features = []
    window_size = 512

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=0.01)

        start = len(z) // 2 - window_size // 2
        feat = extract_features(z[start:start+window_size])
        all_features.append(feat)

    X = np.array(all_features)

    # Simple PCA using SVD (no sklearn)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = U * S  # Project onto principal components

    variance_explained = (S**2) / np.sum(S**2)
    print(f"PCA variance explained: {variance_explained[:3]}")

    # Check if PC1 is continuous or stepped
    pc1 = X_pca[:, 0]

    # Derivative analysis
    dpc1 = np.diff(pc1)

    # If discretized: large jumps at certain points
    # If continuous: smooth derivative

    jump_threshold = np.std(dpc1) * 2
    jumps = np.where(np.abs(dpc1) > jump_threshold)[0]

    print(f"\nPC1 derivative analysis:")
    print(f"  Mean |dPC1|: {np.mean(np.abs(dpc1)):.4f}")
    print(f"  Max |dPC1|: {np.max(np.abs(dpc1)):.4f}")
    print(f"  Large jumps (>{jump_threshold:.4f}): {len(jumps)}")

    if len(jumps) > 3:
        print(f"  Jump locations (E indices): {jumps}")
        print(f"  Jump E values: {energies[jumps]}")

    # Check correlation with energy
    r_pc1_E, _ = pearsonr(pc1, energies)
    print(f"\nr(PC1, E): {r_pc1_E:.4f}")

    if len(jumps) > 3 and len(jumps) < len(energies) / 3:
        print("\n\033[92mHST features show discrete jumps!\033[0m")
        status = 'DISCRETE_SCALES'
    else:
        print("\n\033[93mHST features vary continuously\033[0m")
        status = 'CONTINUOUS'

    return status, X_pca, energies


def test_quantization_prediction():
    """
    Test 4: If action is quantized, can we predict which level?

    Train classifier (not regressor) to predict discrete action "bin".
    If HST naturally discretizes, classification should be easy.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Action Level Classification")
    print("=" * 60)

    sho = SimpleHarmonicOscillator(omega0=1.0)

    # Define discrete action levels (Bohr-Sommerfeld style)
    n_levels = 5
    level_centers = np.arange(n_levels) + 0.5  # I_n = n + 1/2

    # Generate data for each level
    all_features = []
    all_labels = []

    window_size = 512
    samples_per_level = 20

    np.random.seed(42)
    for level_idx, I_target in enumerate(level_centers):
        E_target = I_target * 1.0  # E = I * ω for SHO

        # Generate multiple trajectories near this energy
        for _ in range(samples_per_level):
            # Add small noise to energy
            E = E_target + np.random.normal(0, 0.05)
            if E <= 0:
                E = 0.1

            q0, p0 = sho.initial_condition_for_energy(E)
            t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=50, dt=0.01)

            # Random window position
            start = np.random.randint(0, len(z) - window_size)
            feat = extract_features(z[start:start+window_size])

            all_features.append(feat)
            all_labels.append(level_idx)

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"Data: {len(X)} samples across {n_levels} action levels")

    # Simple neural network classifier (no sklearn)
    class Classifier(nn.Module):
        def __init__(self, input_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, n_classes)
            )

        def forward(self, x):
            return self.net(x)

    # Train/test split
    n_test = len(X) // 5
    indices = np.random.permutation(len(X))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train
    model = Classifier(X.shape[1], n_levels)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(1000):
        logits = model(X_t)
        loss = criterion(logits, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test accuracy
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).numpy()
        accuracy = np.mean(preds == y_test)

    print(f"\nClassification accuracy: {accuracy:.2%}")
    print(f"  (Random guess would be {1/n_levels:.2%})")

    # If HST naturally discretizes, classification should be nearly perfect
    if accuracy > 0.9:
        print("\n\033[92mHIGH ACCURACY: HST features separate action levels well!\033[0m")
        status = 'SEPARABLE'
    elif accuracy > 0.6:
        print("\n\033[93mMODERATE: Some separation but not clean\033[0m")
        status = 'PARTIAL'
    else:
        print("\n\033[91mLOW: Action levels not separable from HST features\033[0m")
        status = 'INSEPARABLE'

    return status, accuracy


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GLINSKY'S 'QUANTIZATION FROM DETERMINISM' TEST")
    print("Does HST+MLP produce discretized action values?")
    print("=" * 70)

    # Test 1: SHO
    status1, P1, E1, clust1 = test_sho_discretization()

    # Test 2: Pendulum
    status2, P2, I2, clust2 = test_pendulum_discretization()

    # Test 3: HST scale structure
    status3, pca3, E3 = test_hst_scale_discretization()

    # Test 4: Classification
    status4, acc4 = test_quantization_prediction()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Does HST Produce Quantized Action?")
    print("=" * 70)
    print(f"Test 1 (SHO continuity):      {status1} (clustering={clust1:.3f})")
    print(f"Test 2 (Pendulum continuity): {status2} (clustering={clust2:.3f})")
    print(f"Test 3 (HST scale jumps):     {status3}")
    print(f"Test 4 (Level classification): {status4} (acc={acc4:.2%})")

    print("\n--- Interpretation ---")
    discretization_evidence = sum([
        status1 == 'DISCRETIZED',
        status2 == 'DISCRETIZED',
        status3 == 'DISCRETE_SCALES',
        status4 == 'SEPARABLE'
    ])

    if discretization_evidence >= 3:
        print("\033[92mSTRONG EVIDENCE: HST architecture induces discretization!\033[0m")
        print("This supports Glinsky's 'quantization from topology' claim.")
    elif discretization_evidence >= 1:
        print("\033[93mWEAK EVIDENCE: Some discretization effects observed\033[0m")
        print("May be due to finite wavelet scales, not fundamental quantization.")
    else:
        print("\033[91mNO EVIDENCE: Action varies continuously\033[0m")
        print("HST does not intrinsically produce quantized outputs.")
