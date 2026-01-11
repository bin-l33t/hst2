"""
Fix for HJB Conservation Loss

The bug: Original loss only checked P(t+1) ≈ P(t) for consecutive pairs,
allowing slow drift.

The fix: Minimize variance of P across ALL points on each trajectory.

This is the key insight from the web UI Claude!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import SimpleHarmonicOscillator, simulate_hamiltonian
from hst import hst_forward_pywt


class HJB_ConservationMLP(nn.Module):
    """MLP with proper conservation loss."""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))  # P and omega
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out[:, 0:1], out[:, 1:2]  # P, omega


def extract_hst_features(z, J=3, wavelet='db8'):
    """Extract HST feature vector."""
    coeffs = hst_forward_pywt(z.real, J=J, wavelet_name=wavelet)

    features = []
    for c in coeffs['cD']:
        features.extend([np.mean(np.abs(c)), np.std(np.abs(c)), np.mean(np.abs(c)**2)])

    ca = coeffs['cA_final']
    features.extend([np.mean(np.abs(ca)), np.std(np.abs(ca)), np.mean(np.abs(ca)**2)])

    return np.array(features)


def generate_windowed_trajectory_data(n_trajectories=30, T=100, dt=0.01,
                                       window_size=512, stride=128):
    """
    Generate data with MULTIPLE WINDOWS per trajectory.

    This is key: we need multiple samples from the same trajectory
    to enforce conservation.

    Returns:
        features_by_traj: list of (n_windows, feature_dim) arrays
        I_true: array of true action per trajectory
    """
    sho = SimpleHarmonicOscillator(omega0=1.0)
    energies = np.linspace(0.5, 5.0, n_trajectories)

    features_by_traj = []
    I_true = []

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=T, dt=dt)

        # Extract features from multiple windows
        windows_features = []
        for start in range(0, len(z) - window_size, stride):
            window = z[start:start+window_size]
            feat = extract_hst_features(window)
            windows_features.append(feat)

        if len(windows_features) > 3:  # Need enough windows
            features_by_traj.append(np.array(windows_features))
            I_true.append(E_actual / sho.omega0)

    return features_by_traj, np.array(I_true)


def train_with_conservation_loss(features_by_traj, I_true, n_epochs=2000, lr=1e-3):
    """
    Train with PROPER conservation loss.

    Loss = regression_loss + conservation_loss

    Where conservation_loss = mean(Var(P) within each trajectory)
    """
    input_dim = features_by_traj[0].shape[1]
    model = HJB_ConservationMLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Normalize targets
    I_mean, I_std = I_true.mean(), I_true.std()
    I_norm = (I_true - I_mean) / (I_std + 1e-8)

    print("Training with CONSERVATION loss...")
    print("(Variance penalty across all windows in each trajectory)")

    for epoch in range(n_epochs):
        total_loss = 0
        total_regression = 0
        total_conservation = 0

        for traj_idx, features in enumerate(features_by_traj):
            X = torch.tensor(features, dtype=torch.float32)
            target_I = I_norm[traj_idx]

            P_pred, omega_pred = model(X)

            # Loss 1: Mean P should match true action
            mean_P = P_pred.mean()
            regression_loss = (mean_P - target_I)**2

            # Loss 2: CONSERVATION - Variance of P should be ZERO
            # This is the KEY FIX!
            conservation_loss = P_pred.var()

            # Combined loss (weight conservation heavily!)
            loss = regression_loss + 10.0 * conservation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_regression += regression_loss.item()
            total_conservation += conservation_loss.item()

        if epoch % 200 == 0:
            n = len(features_by_traj)
            print(f"Epoch {epoch:4d}: loss={total_loss/n:.6f} "
                  f"(reg={total_regression/n:.6f}, cons={total_conservation/n:.6f})")

    return model, I_mean, I_std


def evaluate_conservation(model, features_by_traj, I_true, I_mean, I_std):
    """Evaluate both regression accuracy AND conservation."""
    model.eval()

    P_means = []
    P_cvs = []

    with torch.no_grad():
        for features in features_by_traj:
            X = torch.tensor(features, dtype=torch.float32)
            P_pred, _ = model(X)
            P_np = P_pred.numpy().flatten()

            # Denormalize
            P_denorm = P_np * I_std + I_mean

            P_means.append(np.mean(P_denorm))

            # CV = std/|mean| (coefficient of variation)
            cv = np.std(P_denorm) / (np.abs(np.mean(P_denorm)) + 1e-10)
            P_cvs.append(cv)

    P_means = np.array(P_means)
    P_cvs = np.array(P_cvs)

    # Regression accuracy
    r, _ = pearsonr(P_means, I_true)

    # Conservation
    mean_cv = np.mean(P_cvs)
    fraction_good = np.mean(P_cvs < 0.10)

    return r, mean_cv, fraction_good, P_cvs


def main():
    print("="*60)
    print("FIXED HJB-MLP with Conservation Loss")
    print("="*60)

    # Generate data
    print("\nGenerating windowed trajectory data...")
    features_by_traj, I_true = generate_windowed_trajectory_data(
        n_trajectories=40, T=100, window_size=512, stride=128
    )

    print(f"Generated {len(features_by_traj)} trajectories")
    print(f"Windows per trajectory: {[len(f) for f in features_by_traj[:5]]}...")

    # Split train/test
    n_train = 30
    train_features = features_by_traj[:n_train]
    train_I = I_true[:n_train]
    test_features = features_by_traj[n_train:]
    test_I = I_true[n_train:]

    # Train
    model, I_mean, I_std = train_with_conservation_loss(
        train_features, train_I, n_epochs=3000
    )

    # Evaluate on training set
    print("\n" + "="*60)
    print("TRAINING SET RESULTS")
    print("="*60)
    r_train, cv_train, frac_train, cvs_train = evaluate_conservation(
        model, train_features, train_I, I_mean, I_std
    )
    print(f"Regression: r(P_mean, I_true) = {r_train:.4f}")
    print(f"Conservation: mean CV = {cv_train:.4f}")
    print(f"Fraction with CV < 0.10: {frac_train:.1%}")

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    r_test, cv_test, frac_test, cvs_test = evaluate_conservation(
        model, test_features, test_I, I_mean, I_std
    )
    print(f"Regression: r(P_mean, I_true) = {r_test:.4f}")
    print(f"Conservation: mean CV = {cv_test:.4f}")
    print(f"Fraction with CV < 0.10: {frac_test:.1%}")
    print(f"Individual CVs: {cvs_test}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if r_test > 0.95 and cv_test < 0.10:
        print("\033[92mPASS: Both regression AND conservation work!\033[0m")
        print("This validates Glinsky's geodesic claim.")
    elif r_test > 0.95:
        print("\033[93mMARGINAL: Regression works but conservation still fails\033[0m")
    else:
        print("\033[91mFAIL: Neither regression nor conservation work well\033[0m")

    return model, r_test, cv_test


if __name__ == "__main__":
    model, r, cv = main()
