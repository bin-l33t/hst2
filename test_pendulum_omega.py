"""
Pendulum Test: Can MLP Learn ω(I) for Nonlinear System?

The pendulum has highly nonlinear ω(E):
- At E → -1 (small oscillations): ω → ω₀ = 1
- At E → 1 (separatrix): ω → 0 (period diverges)

Theoretical: T(E) = 4K(k) where k² = (1+E)/2

This is the KEY test for non-trivial action-angle learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from scipy.special import ellipk
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian
from hst import hst_forward_pywt


def theoretical_omega(E):
    """Theoretical ω(E) for pendulum in libration."""
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 >= 1 or k2 <= 0:
        return np.nan
    try:
        K = ellipk(k2)
        T = 4 * K
        return 2 * np.pi / T
    except:
        return np.nan


def extract_features(z):
    """Extract HST features."""
    coeffs = hst_forward_pywt(z.real, J=3, wavelet_name='db8')
    features = []
    for c in coeffs['cD']:
        features.extend([np.mean(np.abs(c)), np.std(np.abs(c)), np.mean(np.abs(c)**2)])
    ca = coeffs['cA_final']
    features.extend([np.mean(np.abs(ca)), np.std(np.abs(ca)), np.mean(np.abs(ca)**2)])
    return np.array(features)


def measure_omega(t, q):
    """Measure ω from zero crossings."""
    zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    if len(zero_crossings) >= 2:
        periods = np.diff(t[zero_crossings])
        return 2 * np.pi / np.mean(periods)
    return np.nan


class ActionOmegaMLP(nn.Module):
    """MLP that learns P (action) and ω(P) jointly."""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)  # P (action)
        )
        self.omega_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1)  # ω(P)
        )

    def forward(self, x):
        P = self.encoder(x)
        omega = self.omega_net(P)
        return P, omega


def generate_pendulum_data(energies, T=100, dt=0.01, window_size=512, stride=128):
    """Generate pendulum data with multiple windows per trajectory."""
    pendulum = PendulumOscillator()
    data = []

    for E in energies:
        if E >= 0.95 or E <= -0.95:  # Avoid separatrix and bottom
            continue

        q0, p0 = pendulum.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T, dt=dt)

        omega_true = theoretical_omega(E_actual)
        omega_meas = measure_omega(t, q)

        if np.isnan(omega_true) or np.isnan(omega_meas):
            continue

        # Extract features from multiple windows
        windows = []
        for start in range(0, len(z) - window_size, stride):
            feat = extract_features(z[start:start+window_size])
            windows.append(feat)

        if len(windows) >= 3:
            data.append({
                'features': np.array(windows),
                'E': E_actual,
                'omega_true': omega_true,
                'omega_meas': omega_meas
            })

    return data


def train_pendulum_mlp(train_data, n_epochs=2000):
    """Train MLP with conservation loss on pendulum data."""
    input_dim = train_data[0]['features'].shape[1]
    model = ActionOmegaMLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Normalize targets
    E_arr = np.array([d['E'] for d in train_data])
    omega_arr = np.array([d['omega_true'] for d in train_data])
    E_mean, E_std = E_arr.mean(), E_arr.std()
    omega_mean, omega_std = omega_arr.mean(), omega_arr.std()

    print(f"Training on {len(train_data)} trajectories...")
    print(f"E range: [{E_arr.min():.2f}, {E_arr.max():.2f}]")
    print(f"ω range: [{omega_arr.min():.3f}, {omega_arr.max():.3f}]")

    for epoch in range(n_epochs):
        total_loss = 0

        for d in train_data:
            X = torch.tensor(d['features'], dtype=torch.float32)
            target_E = (d['E'] - E_mean) / (E_std + 1e-8)
            target_omega = (d['omega_true'] - omega_mean) / (omega_std + 1e-8)

            P, omega = model(X)

            # Loss 1: Mean P should correlate with E
            reg_loss_P = (P.mean() - target_E)**2

            # Loss 2: CONSERVATION - P variance should be zero
            cons_loss = P.var()

            # Loss 3: Mean omega should match theoretical
            reg_loss_omega = (omega.mean() - target_omega)**2

            # Combined loss
            loss = reg_loss_P + 10.0 * cons_loss + reg_loss_omega

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 400 == 0:
            print(f"Epoch {epoch}: loss = {total_loss/len(train_data):.6f}")

    return model, E_mean, E_std, omega_mean, omega_std


def evaluate_pendulum(model, data, E_mean, E_std, omega_mean, omega_std, label=""):
    """Evaluate on dataset."""
    model.eval()

    P_means = []
    P_cvs = []
    omega_preds = []
    E_true = []
    omega_true = []

    with torch.no_grad():
        for d in data:
            X = torch.tensor(d['features'], dtype=torch.float32)
            P, omega = model(X)

            P_np = P.numpy().flatten()
            P_denorm = P_np * E_std + E_mean

            omega_np = omega.numpy().flatten()
            omega_denorm = omega_np * omega_std + omega_mean

            P_means.append(np.mean(P_denorm))
            P_cvs.append(np.std(P_denorm) / (np.abs(np.mean(P_denorm)) + 1e-10))
            omega_preds.append(np.mean(omega_denorm))
            E_true.append(d['E'])
            omega_true.append(d['omega_true'])

    P_means = np.array(P_means)
    omega_preds = np.array(omega_preds)
    E_true = np.array(E_true)
    omega_true = np.array(omega_true)

    r_P_E, _ = pearsonr(P_means, E_true)
    r_omega, _ = pearsonr(omega_preds, omega_true)
    mean_cv = np.mean(P_cvs)

    print(f"\n{label} Results:")
    print(f"  r(P, E): {r_P_E:.4f}")
    print(f"  r(ω_pred, ω_true): {r_omega:.4f}")
    print(f"  Mean CV (conservation): {mean_cv:.4f}")

    return r_P_E, r_omega, mean_cv


def test_pendulum_omega():
    """Main test: Can MLP learn ω(E) for pendulum?"""
    print("="*60)
    print("TEST: Pendulum ω(E) Learning")
    print("="*60)

    # Generate training data (libration regime, away from separatrix)
    train_energies = np.linspace(-0.8, 0.6, 20)
    train_data = generate_pendulum_data(train_energies, T=150)

    if len(train_data) < 5:
        print("Not enough valid trajectories!")
        return

    # Train
    model, E_mean, E_std, omega_mean, omega_std = train_pendulum_mlp(train_data)

    # Evaluate on training data
    r_P_train, r_omega_train, cv_train = evaluate_pendulum(
        model, train_data, E_mean, E_std, omega_mean, omega_std, "TRAIN"
    )

    # Generate TEST data at different energies
    test_energies = np.linspace(-0.6, 0.8, 15)  # Includes some overlap, some new
    test_data = generate_pendulum_data(test_energies, T=150)

    r_P_test, r_omega_test, cv_test = evaluate_pendulum(
        model, test_data, E_mean, E_std, omega_mean, omega_std, "TEST"
    )

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if r_P_test > 0.90 and r_omega_test > 0.85 and cv_test < 0.15:
        print("\033[92mPASS: MLP learns pendulum ω(E) with conservation!\033[0m")
        status = 'PASS'
    elif r_P_test > 0.80 and r_omega_test > 0.70:
        print("\033[93mMARGINAL: Partial success\033[0m")
        status = 'MARGINAL'
    else:
        print("\033[91mFAIL\033[0m")
        status = 'FAIL'

    return status, r_P_test, r_omega_test, cv_test


def test_generalization():
    """Test: Train on E ∈ [-0.8, 0.3], test on E ∈ [0.5, 0.9]."""
    print("\n" + "="*60)
    print("TEST: Generalization to Unseen Energies")
    print("="*60)

    # Training: low to medium energy
    train_energies = np.linspace(-0.8, 0.3, 15)
    train_data = generate_pendulum_data(train_energies, T=150)

    if len(train_data) < 5:
        print("Not enough training data!")
        return

    print(f"Training energies: [{train_energies.min():.2f}, {train_energies.max():.2f}]")

    model, E_mean, E_std, omega_mean, omega_std = train_pendulum_mlp(train_data, n_epochs=2500)

    # Test: higher energy (closer to separatrix - harder!)
    test_energies = np.linspace(0.5, 0.85, 10)
    test_data = generate_pendulum_data(test_energies, T=200)

    if len(test_data) < 3:
        print("Not enough test data!")
        return

    print(f"\nTest energies: [{test_energies.min():.2f}, {test_energies.max():.2f}]")
    print("(Note: This is EXTRAPOLATION near separatrix - very challenging!)")

    r_P_test, r_omega_test, cv_test = evaluate_pendulum(
        model, test_data, E_mean, E_std, omega_mean, omega_std, "GENERALIZATION"
    )

    print("\n" + "="*60)
    if r_P_test > 0.80 and r_omega_test > 0.70:
        print("\033[92mPASS: Generalizes to unseen energies!\033[0m")
    elif r_P_test > 0.60:
        print("\033[93mMARGINAL: Some generalization\033[0m")
    else:
        print("\033[91mFAIL: Poor generalization\033[0m")

    return r_P_test, r_omega_test


if __name__ == "__main__":
    # Test 1: Basic pendulum ω learning
    result = test_pendulum_omega()

    # Test 2: Generalization
    print("\n")
    test_generalization()
