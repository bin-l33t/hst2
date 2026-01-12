"""
Test: Local Accuracy Near vs Far from Separatrix

Better test of Glinsky's cusp theory:
- Train on FULL energy range (including near-separatrix)
- Evaluate LOCAL accuracy in different energy bins
- If cusp theory is correct: MLP should have advantage near E=1

This tests whether the 2.5% error in linear methods is concentrated at cusps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from scipy.special import ellipk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from hst import extract_features
from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian


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


class LinearRegressor:
    """Simple linear regression."""
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_b = np.column_stack([np.ones(len(X)), X])
        reg = 1e-6 * np.eye(X_b.shape[1])
        self.weights = np.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)

    def predict(self, X):
        X_b = np.column_stack([np.ones(len(X)), X])
        return X_b @ self.weights


class SimpleMLP(nn.Module):
    """MLP with ReLU."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def generate_data(energies, window_size=512, windows_per_energy=20):
    """Generate pendulum data at specified energies."""
    pendulum = PendulumOscillator()
    data = []

    for E in energies:
        if E >= 0.98 or E <= -0.98:
            continue

        q0, p0 = pendulum.initial_condition_for_energy(E)
        # Longer simulation for high energy (slow period)
        T_sim = 300 if E > 0.8 else 200
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=0.01)

        omega_true = theoretical_omega(E_actual)
        if np.isnan(omega_true):
            continue

        for _ in range(windows_per_energy):
            start = np.random.randint(0, len(z) - window_size)
            window = z[start:start+window_size]
            feat = extract_features(window)

            data.append({
                'features': feat,
                'E': E_actual,
                'omega': omega_true
            })

    return data


def main():
    print("=" * 70)
    print("TEST: Local Accuracy - Does MLP Help at Cusps?")
    print("=" * 70)

    np.random.seed(42)

    # Generate data across FULL range
    E_all = np.linspace(-0.8, 0.95, 50)  # Full range including near-separatrix

    print(f"\nEnergy range: E ∈ [{E_all.min():.2f}, {E_all.max():.2f}]")
    print("Generating data across full range...")

    data = generate_data(E_all, windows_per_energy=25)
    print(f"Total samples: {len(data)}")

    # Prepare arrays
    X = np.array([d['features'] for d in data])
    E_arr = np.array([d['E'] for d in data])
    omega_arr = np.array([d['omega'] for d in data])

    # Train/test split (stratified by energy)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    E_train, E_test = E_arr[train_idx], E_arr[test_idx]
    omega_train, omega_test = omega_arr[train_idx], omega_arr[test_idx]

    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    E_mean, E_std = E_train.mean(), E_train.std()
    E_train_norm = (E_train - E_mean) / E_std
    E_test_norm = (E_test - E_mean) / E_std

    omega_mean, omega_std = omega_train.mean(), omega_train.std()
    omega_train_norm = (omega_train - omega_mean) / omega_std
    omega_test_norm = (omega_test - omega_mean) / omega_std

    # =====================================================
    # Train both models on FULL range
    # =====================================================
    print("\nTraining models on full energy range...")

    # Linear
    linear_E = LinearRegressor()
    linear_E.fit(X_train_norm, E_train_norm)

    linear_omega = LinearRegressor()
    linear_omega.fit(X_train_norm, omega_train_norm)

    # MLP
    model_E = SimpleMLP(X_train.shape[1], hidden_dims=[128, 64, 32])
    optimizer_E = optim.Adam(model_E.parameters(), lr=1e-3)
    X_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_t_E = torch.tensor(E_train_norm.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(4000):
        pred = model_E(X_t)
        loss = nn.MSELoss()(pred, y_t_E)
        optimizer_E.zero_grad()
        loss.backward()
        optimizer_E.step()
        if epoch % 1000 == 0:
            print(f"  MLP E epoch {epoch}: loss = {loss.item():.6f}")

    model_omega = SimpleMLP(X_train.shape[1], hidden_dims=[128, 64, 32])
    optimizer_omega = optim.Adam(model_omega.parameters(), lr=1e-3)
    y_t_omega = torch.tensor(omega_train_norm.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(4000):
        pred = model_omega(X_t)
        loss = nn.MSELoss()(pred, y_t_omega)
        optimizer_omega.zero_grad()
        loss.backward()
        optimizer_omega.step()

    model_E.eval()
    model_omega.eval()

    # =====================================================
    # Evaluate by energy bins
    # =====================================================
    print("\n" + "=" * 70)
    print("LOCAL ACCURACY BY ENERGY BIN")
    print("=" * 70)

    # Define bins
    bins = [(-0.8, -0.4), (-0.4, 0.0), (0.0, 0.4), (0.4, 0.7), (0.7, 0.85), (0.85, 0.95)]
    bin_labels = ['E∈[-0.8,-0.4]', 'E∈[-0.4,0.0]', 'E∈[0.0,0.4]',
                  'E∈[0.4,0.7]', 'E∈[0.7,0.85]', 'E∈[0.85,0.95]']

    results = []

    print(f"\n{'Energy Bin':<18} {'n':<6} {'Lin r(E)':<12} {'MLP r(E)':<12} {'MLP adv':<10} {'Lin r(ω)':<12} {'MLP r(ω)':<12}")
    print("-" * 95)

    for (e_lo, e_hi), label in zip(bins, bin_labels):
        mask = (E_test >= e_lo) & (E_test < e_hi)
        n_bin = mask.sum()

        if n_bin < 10:
            print(f"{label:<18} {n_bin:<6} (insufficient data)")
            continue

        # Linear predictions
        pred_lin_E = linear_E.predict(X_test_norm[mask])
        pred_lin_omega = linear_omega.predict(X_test_norm[mask])

        # MLP predictions
        with torch.no_grad():
            X_bin = torch.tensor(X_test_norm[mask], dtype=torch.float32)
            pred_mlp_E = model_E(X_bin).numpy().flatten()
            pred_mlp_omega = model_omega(X_bin).numpy().flatten()

        # Correlations
        r_lin_E, _ = pearsonr(pred_lin_E, E_test_norm[mask])
        r_mlp_E, _ = pearsonr(pred_mlp_E, E_test_norm[mask])
        r_lin_omega, _ = pearsonr(pred_lin_omega, omega_test_norm[mask])
        r_mlp_omega, _ = pearsonr(pred_mlp_omega, omega_test_norm[mask])

        mlp_advantage = r_mlp_E - r_lin_E

        results.append({
            'bin': label,
            'e_mid': (e_lo + e_hi) / 2,
            'n': n_bin,
            'r_lin_E': r_lin_E,
            'r_mlp_E': r_mlp_E,
            'r_lin_omega': r_lin_omega,
            'r_mlp_omega': r_mlp_omega,
            'mlp_advantage': mlp_advantage
        })

        print(f"{label:<18} {n_bin:<6} {r_lin_E:<12.4f} {r_mlp_E:<12.4f} {mlp_advantage:>+9.4f} {r_lin_omega:<12.4f} {r_mlp_omega:<12.4f}")

    # =====================================================
    # Overall test set performance
    # =====================================================
    print("\n" + "-" * 70)
    print("OVERALL TEST SET:")

    pred_lin_E_all = linear_E.predict(X_test_norm)
    pred_lin_omega_all = linear_omega.predict(X_test_norm)

    with torch.no_grad():
        X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
        pred_mlp_E_all = model_E(X_test_t).numpy().flatten()
        pred_mlp_omega_all = model_omega(X_test_t).numpy().flatten()

    r_lin_E_all, _ = pearsonr(pred_lin_E_all, E_test_norm)
    r_mlp_E_all, _ = pearsonr(pred_mlp_E_all, E_test_norm)
    r_lin_omega_all, _ = pearsonr(pred_lin_omega_all, omega_test_norm)
    r_mlp_omega_all, _ = pearsonr(pred_mlp_omega_all, omega_test_norm)

    print(f"  Linear:  r(E) = {r_lin_E_all:.4f}, r(ω) = {r_lin_omega_all:.4f}")
    print(f"  MLP:     r(E) = {r_mlp_E_all:.4f}, r(ω) = {r_mlp_omega_all:.4f}")

    # =====================================================
    # Visualization
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: r(E) by energy bin
    ax1 = axes[0, 0]
    e_mids = [r['e_mid'] for r in results]
    r_lin_vals = [r['r_lin_E'] for r in results]
    r_mlp_vals = [r['r_mlp_E'] for r in results]

    ax1.plot(e_mids, r_lin_vals, 'o-', label='Linear', markersize=10, linewidth=2)
    ax1.plot(e_mids, r_mlp_vals, 's-', label='MLP (ReLU)', markersize=10, linewidth=2)
    ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Separatrix')
    ax1.axvspan(0.7, 1.0, alpha=0.15, color='red')
    ax1.set_xlabel('Energy E (bin midpoint)')
    ax1.set_ylabel('Local r(E)')
    ax1.set_title('Energy Prediction Accuracy by Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Plot 2: MLP advantage by energy
    ax2 = axes[0, 1]
    advantages = [r['mlp_advantage'] for r in results]
    colors = ['green' if a > 0.01 else 'gray' if a > -0.01 else 'red' for a in advantages]
    ax2.bar(range(len(results)), advantages, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([r['bin'] for r in results], rotation=45, ha='right')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('MLP advantage (Δr)')
    ax2.set_title('MLP - Linear Advantage by Energy Region')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Predictions vs true (scatter)
    ax3 = axes[1, 0]
    E_test_denorm = E_test_norm * E_std + E_mean
    pred_lin_denorm = pred_lin_E_all * E_std + E_mean
    pred_mlp_denorm = pred_mlp_E_all * E_std + E_mean

    # Color by distance from separatrix
    colors_scatter = E_test
    ax3.scatter(E_test, pred_lin_denorm, c=colors_scatter, cmap='RdYlGn_r', alpha=0.5,
                label='Linear', marker='o', s=20)
    ax3.plot([E_test.min(), E_test.max()], [E_test.min(), E_test.max()], 'k--', label='Perfect')
    ax3.set_xlabel('True Energy E')
    ax3.set_ylabel('Predicted Energy')
    ax3.set_title(f'Linear Predictions (r={r_lin_E_all:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.scatter(E_test, pred_mlp_denorm, c=colors_scatter, cmap='RdYlGn_r', alpha=0.5,
                label='MLP', marker='s', s=20)
    ax4.plot([E_test.min(), E_test.max()], [E_test.min(), E_test.max()], 'k--', label='Perfect')
    ax4.set_xlabel('True Energy E')
    ax4.set_ylabel('Predicted Energy')
    ax4.set_title(f'MLP Predictions (r={r_mlp_E_all:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/ubuntu/rectifier/cusp_local_accuracy.png', dpi=150)
    print("\nSaved: cusp_local_accuracy.png")

    # =====================================================
    # Verdict
    # =====================================================
    print("\n" + "=" * 70)
    print("VERDICT: Glinsky Cusp Theory Test")
    print("=" * 70)

    # Check if MLP advantage increases near separatrix
    near_sep_results = [r for r in results if r['e_mid'] > 0.6]
    far_results = [r for r in results if r['e_mid'] <= 0.6]

    if near_sep_results and far_results:
        avg_adv_near = np.mean([r['mlp_advantage'] for r in near_sep_results])
        avg_adv_far = np.mean([r['mlp_advantage'] for r in far_results])

        print(f"\nAverage MLP advantage:")
        print(f"  Far from separatrix (E ≤ 0.6):  {avg_adv_far:+.4f}")
        print(f"  Near separatrix (E > 0.6):      {avg_adv_near:+.4f}")

        if avg_adv_near > avg_adv_far + 0.02:
            print(f"\n\033[92mCUSP THEORY SUPPORTED!\033[0m")
            print("  MLP shows increasing advantage near singularity.")
        elif avg_adv_near > 0 and avg_adv_far > 0:
            print(f"\n\033[93mMLP HELPS EVERYWHERE\033[0m")
            print("  MLP advantage is not specific to cusps.")
        else:
            print(f"\n\033[91mNO CUSP-SPECIFIC ADVANTAGE\033[0m")
            print("  Linear and MLP perform similarly.")

    return results


if __name__ == "__main__":
    results = main()
