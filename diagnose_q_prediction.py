"""
Diagnostic: Why is Q (phase) prediction poor in coarse-graining test?

Earlier test showed r(Q, θ) = 0.9999, so the information IS in HST coefficients.
Where is it being lost?

Diagnostic 1: PCA truncation - is variance being lost?
Diagnostic 2: MLP capacity - is the network too small?
Diagnostic 3: Joint vs separate training - does P compete with Q?
Diagnostic 4: Fine vs coarse features - which levels contain Q?
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


def extract_hst_coefficients(z, J=4):
    """Extract raw HST coefficients (not summarized)."""
    coeffs = hst_forward_pywt(z.real, J=J, wavelet_name='db8')
    return coeffs


def extract_features_by_level(z, J=4):
    """Extract features separately for each HST level."""
    coeffs = hst_forward_pywt(z.real, J=J, wavelet_name='db8')

    level_features = []
    for j, cD in enumerate(coeffs['cD']):
        feat = [np.mean(np.abs(cD)), np.std(np.abs(cD)), np.mean(np.abs(cD)**2)]
        level_features.append(np.array(feat))

    ca = coeffs['cA_final']
    final_feat = np.array([np.mean(np.abs(ca)), np.std(np.abs(ca)), np.mean(np.abs(ca)**2)])
    level_features.append(final_feat)

    return level_features, coeffs


def generate_sho_data(n_trajectories=30, n_windows_per_traj=10):
    """Generate SHO data with known P and Q."""
    sho = SimpleHarmonicOscillator(omega0=1.0)

    energies = np.linspace(0.5, 3.0, n_trajectories)

    all_data = []
    window_size = 512

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=100, dt=0.01)

        P_true = E_actual / 1.0  # Action = E/ω for SHO

        for _ in range(n_windows_per_traj):
            start = np.random.randint(0, len(z) - window_size)
            window = z[start:start+window_size]

            # Phase at window center
            center = start + window_size // 2
            Q_true = np.arctan2(p[center], q[center])

            all_data.append({
                'window': window,
                'P': P_true,
                'Q': Q_true,
                'E': E_actual
            })

    return all_data


def diagnostic_1_pca_truncation():
    """Check how much variance is in each PC."""
    print("=" * 60)
    print("DIAGNOSTIC 1: PCA Truncation")
    print("=" * 60)

    data = generate_sho_data(n_trajectories=50, n_windows_per_traj=5)

    # Collect all features
    all_features = []
    for d in data:
        level_feat, _ = extract_features_by_level(d['window'])
        all_features.append(np.concatenate(level_feat))

    X = np.array(all_features)
    print(f"Feature matrix shape: {X.shape}")

    # SVD for variance analysis
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    variance_ratio = (S**2) / np.sum(S**2)
    cumulative = np.cumsum(variance_ratio)

    print(f"\nVariance explained by each PC:")
    for i in range(min(10, len(variance_ratio))):
        print(f"  PC{i+1}: {variance_ratio[i]:.4f} (cumulative: {cumulative[i]:.4f})")

    # How many PCs for 99%?
    n_99 = np.searchsorted(cumulative, 0.99) + 1
    n_95 = np.searchsorted(cumulative, 0.95) + 1
    print(f"\nPCs needed for 95%: {n_95}")
    print(f"PCs needed for 99%: {n_99}")

    # Check correlation of each PC with P and Q
    P_arr = np.array([d['P'] for d in data])
    Q_sin = np.array([np.sin(d['Q']) for d in data])
    Q_cos = np.array([np.cos(d['Q']) for d in data])

    X_pca = U * S

    print(f"\nCorrelation of each PC with P and Q:")
    print(f"{'PC':>5} {'r(PC,P)':>10} {'r(PC,sin(Q))':>15} {'r(PC,cos(Q))':>15}")
    print("-" * 50)

    for i in range(min(8, X_pca.shape[1])):
        r_P, _ = pearsonr(X_pca[:, i], P_arr)
        r_sin, _ = pearsonr(X_pca[:, i], Q_sin)
        r_cos, _ = pearsonr(X_pca[:, i], Q_cos)
        print(f"{i+1:>5} {r_P:>10.3f} {r_sin:>15.3f} {r_cos:>15.3f}")

    return variance_ratio, cumulative


def diagnostic_2_mlp_capacity():
    """Train MLPs of increasing size."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2: MLP Capacity Sweep")
    print("=" * 60)

    data = generate_sho_data(n_trajectories=50, n_windows_per_traj=5)

    # Prepare data
    all_features = []
    P_arr = []
    Q_sin = []
    Q_cos = []

    for d in data:
        level_feat, _ = extract_features_by_level(d['window'])
        all_features.append(np.concatenate(level_feat))
        P_arr.append(d['P'])
        Q_sin.append(np.sin(d['Q']))
        Q_cos.append(np.cos(d['Q']))

    X = np.array(all_features)
    P_arr = np.array(P_arr)
    Q_sin = np.array(Q_sin)
    Q_cos = np.array(Q_cos)

    # Normalize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    P_mean, P_std = P_arr.mean(), P_arr.std()
    P_norm = (P_arr - P_mean) / P_std

    # Train/test split
    n = len(X)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    architectures = [
        [32],
        [64, 32],
        [128, 64, 32],
        [256, 128, 64, 32],
        [512, 256, 128, 64]
    ]

    print(f"\n{'Architecture':>25} {'r(P)':>10} {'r(sin(Q))':>12} {'r(cos(Q))':>12}")
    print("-" * 65)

    results = []

    for arch in architectures:
        # Build MLP
        layers = []
        in_dim = X.shape[1]
        for h in arch:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 3))  # Output: P, sin(Q), cos(Q)

        model = nn.Sequential(*layers)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
        y_train = torch.tensor(np.column_stack([P_norm[train_idx], Q_sin[train_idx], Q_cos[train_idx]]), dtype=torch.float32)

        for epoch in range(2000):
            pred = model(X_train)
            loss = nn.MSELoss()(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            X_test = torch.tensor(X_norm[test_idx], dtype=torch.float32)
            pred = model(X_test).numpy()

        pred_P = pred[:, 0] * P_std + P_mean
        pred_sin = pred[:, 1]
        pred_cos = pred[:, 2]

        r_P, _ = pearsonr(pred_P, P_arr[test_idx])
        r_sin, _ = pearsonr(pred_sin, Q_sin[test_idx])
        r_cos, _ = pearsonr(pred_cos, Q_cos[test_idx])

        arch_str = str(arch)
        print(f"{arch_str:>25} {r_P:>10.3f} {r_sin:>12.3f} {r_cos:>12.3f}")

        results.append({
            'arch': arch,
            'r_P': r_P,
            'r_sin': r_sin,
            'r_cos': r_cos
        })

    return results


def diagnostic_3_separate_heads():
    """Train separate MLPs for P and Q."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3: Separate P and Q Training")
    print("=" * 60)

    data = generate_sho_data(n_trajectories=50, n_windows_per_traj=5)

    all_features = []
    P_arr = []
    Q_sin = []
    Q_cos = []

    for d in data:
        level_feat, _ = extract_features_by_level(d['window'])
        all_features.append(np.concatenate(level_feat))
        P_arr.append(d['P'])
        Q_sin.append(np.sin(d['Q']))
        Q_cos.append(np.cos(d['Q']))

    X = np.array(all_features)
    P_arr = np.array(P_arr)
    Q_sin = np.array(Q_sin)
    Q_cos = np.array(Q_cos)

    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    P_mean, P_std = P_arr.mean(), P_arr.std()
    P_norm = (P_arr - P_mean) / P_std

    n = len(X)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    def train_mlp(X_train, y_train, X_test, y_test, name, epochs=3000):
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, y_train.shape[1] if len(y_train.shape) > 1 else 1)
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        if len(y_t.shape) == 1:
            y_t = y_t.unsqueeze(1)

        for epoch in range(epochs):
            pred = model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            pred = model(X_test_t).numpy()

        return pred

    # Train P-only MLP
    print("\nTraining P-only MLP...")
    pred_P = train_mlp(X_norm[train_idx], P_norm[train_idx], X_norm[test_idx], P_norm[test_idx], "P")
    pred_P = pred_P.flatten() * P_std + P_mean
    r_P, _ = pearsonr(pred_P, P_arr[test_idx])
    print(f"  r(P_pred, P_true) = {r_P:.4f}")

    # Train Q-only MLP (sin and cos)
    print("\nTraining Q-only MLP...")
    y_Q_train = np.column_stack([Q_sin[train_idx], Q_cos[train_idx]])
    y_Q_test = np.column_stack([Q_sin[test_idx], Q_cos[test_idx]])
    pred_Q = train_mlp(X_norm[train_idx], y_Q_train, X_norm[test_idx], y_Q_test, "Q")

    r_sin, _ = pearsonr(pred_Q[:, 0], Q_sin[test_idx])
    r_cos, _ = pearsonr(pred_Q[:, 1], Q_cos[test_idx])
    print(f"  r(sin(Q)_pred, sin(Q)_true) = {r_sin:.4f}")
    print(f"  r(cos(Q)_pred, cos(Q)_true) = {r_cos:.4f}")

    # Reconstruct Q from sin/cos
    Q_pred = np.arctan2(pred_Q[:, 0], pred_Q[:, 1])
    Q_true = np.arctan2(Q_sin[test_idx], Q_cos[test_idx])

    # Phase correlation (handle wrap-around)
    phase_diff = np.angle(np.exp(1j * (Q_pred - Q_true)))
    mean_error = np.mean(np.abs(phase_diff))

    print(f"  Mean phase error = {mean_error:.4f} rad")

    return r_P, r_sin, r_cos


def diagnostic_4_level_by_level():
    """Check which HST levels contain P and Q information."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4: Level-by-Level Analysis")
    print("=" * 60)

    data = generate_sho_data(n_trajectories=50, n_windows_per_traj=5)

    # Collect features by level
    n_levels = 5  # J=4 gives 4 detail levels + 1 approx
    level_features = [[] for _ in range(n_levels)]
    P_arr = []
    Q_sin = []
    Q_cos = []

    for d in data:
        feat_by_level, _ = extract_features_by_level(d['window'], J=4)
        for j, feat in enumerate(feat_by_level):
            level_features[j].append(feat)
        P_arr.append(d['P'])
        Q_sin.append(np.sin(d['Q']))
        Q_cos.append(np.cos(d['Q']))

    P_arr = np.array(P_arr)
    Q_sin = np.array(Q_sin)
    Q_cos = np.array(Q_cos)

    dt = 0.01
    period = 2 * np.pi

    print(f"\nPeriod T = {period:.2f}")
    print(f"\n{'Level':>8} {'Timescale':>12} {'Timescale/T':>12} {'r(feat,P)':>12} {'r(feat,sin(Q))':>15}")
    print("-" * 65)

    for j in range(n_levels):
        X_level = np.array(level_features[j])
        timescale = dt * (2 ** (j + 1)) if j < n_levels - 1 else dt * (2 ** n_levels)

        # Use first feature (mean |c|) as representative
        feat = X_level[:, 0]

        r_P, _ = pearsonr(feat, P_arr)
        r_sin, _ = pearsonr(feat, Q_sin)

        level_name = f"cD[{j}]" if j < n_levels - 1 else "cA_final"
        print(f"{level_name:>8} {timescale:>12.3f} {timescale/period:>12.3f} {r_P:>12.3f} {r_sin:>15.3f}")

    # Now train MLP on each level separately
    print("\n--- MLP trained on each level separately ---")
    print(f"{'Level':>8} {'r(P)':>10} {'r(sin(Q))':>12}")
    print("-" * 35)

    n = len(P_arr)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    P_mean, P_std = P_arr.mean(), P_arr.std()
    P_norm = (P_arr - P_mean) / P_std

    for j in range(n_levels):
        X_level = np.array(level_features[j])
        X_mean, X_std = X_level.mean(axis=0), X_level.std(axis=0) + 1e-8
        X_norm = (X_level - X_mean) / X_std

        # Simple MLP
        model = nn.Sequential(
            nn.Linear(X_level.shape[1], 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2)  # P and sin(Q)
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        X_t = torch.tensor(X_norm[train_idx], dtype=torch.float32)
        y_t = torch.tensor(np.column_stack([P_norm[train_idx], Q_sin[train_idx]]), dtype=torch.float32)

        for _ in range(2000):
            pred = model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_norm[test_idx], dtype=torch.float32)
            pred = model(X_test_t).numpy()

        r_P, _ = pearsonr(pred[:, 0], P_norm[test_idx])
        r_sin, _ = pearsonr(pred[:, 1], Q_sin[test_idx])

        level_name = f"cD[{j}]" if j < n_levels - 1 else "cA_final"
        print(f"{level_name:>8} {r_P:>10.3f} {r_sin:>12.3f}")


def diagnostic_5_raw_coefficients():
    """Use RAW HST coefficients instead of summarized features."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 5: Raw HST Coefficients (No Summarization)")
    print("=" * 60)

    data = generate_sho_data(n_trajectories=30, n_windows_per_traj=5)

    # Collect raw coefficients (flattened)
    all_raw = []
    P_arr = []
    Q_sin = []
    Q_cos = []

    for d in data:
        coeffs = hst_forward_pywt(d['window'].real, J=3, wavelet_name='db8')

        # Concatenate all detail coefficients
        raw = []
        for cD in coeffs['cD']:
            raw.extend(np.abs(cD).tolist())
        raw.extend(np.abs(coeffs['cA_final']).tolist())

        all_raw.append(raw)
        P_arr.append(d['P'])
        Q_sin.append(np.sin(d['Q']))
        Q_cos.append(np.cos(d['Q']))

    X = np.array(all_raw)
    P_arr = np.array(P_arr)
    Q_sin = np.array(Q_sin)
    Q_cos = np.array(Q_cos)

    print(f"Raw feature dimension: {X.shape[1]}")

    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    P_mean, P_std = P_arr.mean(), P_arr.std()
    P_norm = (P_arr - P_mean) / P_std

    n = len(X)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]

    # Train larger MLP on raw coefficients
    model = nn.Sequential(
        nn.Linear(X.shape[1], 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 3)  # P, sin(Q), cos(Q)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    y_t = torch.tensor(np.column_stack([P_norm[train_idx], Q_sin[train_idx], Q_cos[train_idx]]), dtype=torch.float32)

    print("\nTraining on raw coefficients...")
    for epoch in range(3000):
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_norm[test_idx], dtype=torch.float32)
        pred = model(X_test_t).numpy()

    r_P, _ = pearsonr(pred[:, 0], P_norm[test_idx])
    r_sin, _ = pearsonr(pred[:, 1], Q_sin[test_idx])
    r_cos, _ = pearsonr(pred[:, 2], Q_cos[test_idx])

    print(f"\nResults with raw coefficients:")
    print(f"  r(P) = {r_P:.4f}")
    print(f"  r(sin(Q)) = {r_sin:.4f}")
    print(f"  r(cos(Q)) = {r_cos:.4f}")

    return r_P, r_sin, r_cos


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Why is Q (phase) prediction poor?")
    print("=" * 70)

    np.random.seed(42)

    # Diagnostic 1
    var_ratio, cumulative = diagnostic_1_pca_truncation()

    # Diagnostic 2
    capacity_results = diagnostic_2_mlp_capacity()

    # Diagnostic 3
    r_P_sep, r_sin_sep, r_cos_sep = diagnostic_3_separate_heads()

    # Diagnostic 4
    diagnostic_4_level_by_level()

    # Diagnostic 5
    r_P_raw, r_sin_raw, r_cos_raw = diagnostic_5_raw_coefficients()

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print("\n1. PCA: Check which PCs correlate with P vs Q")

    print("\n2. MLP Capacity:")
    best_Q = max(capacity_results, key=lambda x: x['r_sin'])
    print(f"   Best architecture for Q: {best_Q['arch']}")
    print(f"   Best r(sin(Q)) = {best_Q['r_sin']:.3f}")

    print("\n3. Separate Training:")
    print(f"   P-only: r = {r_P_sep:.3f}")
    print(f"   Q-only: r(sin) = {r_sin_sep:.3f}, r(cos) = {r_cos_sep:.3f}")

    print("\n4. Level-by-level: See which levels contain Q info")

    print("\n5. Raw Coefficients:")
    print(f"   r(P) = {r_P_raw:.3f}, r(sin(Q)) = {r_sin_raw:.3f}")

    print("\n--- DIAGNOSIS ---")
    if r_sin_raw > 0.7:
        print("Q information IS present in raw coefficients")
        print("Loss is in summarization (mean/std/mean^2)")
    elif best_Q['r_sin'] > 0.5:
        print("Larger MLP helps - capacity was limiting")
    else:
        print("Q information may be fundamentally limited in HST structure")
