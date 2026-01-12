"""
Test: Linear vs MLP Near Separatrix

Verifies Glinsky's claim that:
- Analytic Hamiltonian is "maximally flat" except at cusps
- Linear methods work well in smooth regions
- MLP with ReLU needed to capture cusps at singularities

For pendulum: separatrix at E = 1 where ω → 0
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
    """Simple linear regression for baseline."""
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add bias column
        X_b = np.column_stack([np.ones(len(X)), X])
        # Solve normal equations with regularization
        reg = 1e-6 * np.eye(X_b.shape[1])
        self.weights = np.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)

    def predict(self, X):
        X_b = np.column_stack([np.ones(len(X)), X])
        return X_b @ self.weights


class SimpleMLP(nn.Module):
    """MLP with ReLU - can capture cusps."""
    def __init__(self, input_dim, hidden_dims=[64, 32]):
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


def generate_data(energies, window_size=512, windows_per_energy=10):
    """Generate pendulum data at specified energies."""
    pendulum = PendulumOscillator()
    data = []

    for E in energies:
        if E >= 0.99 or E <= -0.99:
            continue

        q0, p0 = pendulum.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=200, dt=0.01)

        omega_true = theoretical_omega(E_actual)
        if np.isnan(omega_true):
            continue

        # Multiple windows per trajectory
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


def train_mlp(X_train, y_train, hidden_dims=[64, 32], epochs=2000, lr=1e-3):
    """Train MLP on normalized data."""
    model = SimpleMLP(X_train.shape[1], hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(epochs):
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def evaluate(model, X, y, is_torch=False):
    """Evaluate model and return predictions."""
    if is_torch:
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            pred = model(X_t).numpy().flatten()
    else:
        pred = model.predict(X)

    r, _ = pearsonr(pred, y)
    mse = np.mean((pred - y)**2)
    return pred, r, mse


def main():
    print("=" * 70)
    print("TEST: Linear vs MLP Near Separatrix (Glinsky Cusp Theory)")
    print("=" * 70)

    np.random.seed(42)

    # Define energy regimes
    E_smooth = np.linspace(-0.8, 0.3, 20)      # Far from separatrix
    E_near_sep = np.linspace(0.7, 0.95, 15)    # Near separatrix (E=1)
    E_all = np.concatenate([E_smooth, E_near_sep])

    print(f"\nEnergy regimes:")
    print(f"  Smooth (training):     E ∈ [{E_smooth.min():.1f}, {E_smooth.max():.1f}]")
    print(f"  Near separatrix (test): E ∈ [{E_near_sep.min():.2f}, {E_near_sep.max():.2f}]")
    print(f"  Separatrix at E = 1 (ω → 0)")

    # Generate data
    print("\nGenerating data...")
    data_smooth = generate_data(E_smooth, windows_per_energy=15)
    data_near = generate_data(E_near_sep, windows_per_energy=15)

    print(f"  Smooth regime samples: {len(data_smooth)}")
    print(f"  Near-separatrix samples: {len(data_near)}")

    # Prepare arrays
    X_smooth = np.array([d['features'] for d in data_smooth])
    E_smooth_arr = np.array([d['E'] for d in data_smooth])
    omega_smooth = np.array([d['omega'] for d in data_smooth])

    X_near = np.array([d['features'] for d in data_near])
    E_near_arr = np.array([d['E'] for d in data_near])
    omega_near = np.array([d['omega'] for d in data_near])

    # Normalize features
    X_mean = X_smooth.mean(axis=0)
    X_std = X_smooth.std(axis=0) + 1e-8
    X_smooth_norm = (X_smooth - X_mean) / X_std
    X_near_norm = (X_near - X_mean) / X_std

    # Normalize targets
    E_mean, E_std = E_smooth_arr.mean(), E_smooth_arr.std()
    E_smooth_norm = (E_smooth_arr - E_mean) / E_std
    E_near_norm = (E_near_arr - E_mean) / E_std

    omega_mean, omega_std = omega_smooth.mean(), omega_smooth.std()
    omega_smooth_norm = (omega_smooth - omega_mean) / omega_std
    omega_near_norm = (omega_near - omega_mean) / omega_std

    # =========================================
    # TEST 1: Predicting Energy (P proxy)
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 1: Predicting Energy E (proxy for action P)")
    print("=" * 70)

    # Train Linear
    linear_E = LinearRegressor()
    linear_E.fit(X_smooth_norm, E_smooth_norm)

    # Train MLP
    mlp_E = train_mlp(X_smooth_norm, E_smooth_norm, hidden_dims=[64, 32], epochs=3000)

    # Evaluate on smooth regime
    _, r_lin_smooth_E, mse_lin_smooth_E = evaluate(linear_E, X_smooth_norm, E_smooth_norm)
    _, r_mlp_smooth_E, mse_mlp_smooth_E = evaluate(mlp_E, X_smooth_norm, E_smooth_norm, is_torch=True)

    # Evaluate on near-separatrix
    pred_lin_near_E, r_lin_near_E, mse_lin_near_E = evaluate(linear_E, X_near_norm, E_near_norm)
    pred_mlp_near_E, r_mlp_near_E, mse_mlp_near_E = evaluate(mlp_E, X_near_norm, E_near_norm, is_torch=True)

    print(f"\n{'Method':<15} {'Smooth r(E)':<15} {'Near-sep r(E)':<15} {'Degradation':<15}")
    print("-" * 60)
    print(f"{'Linear':<15} {r_lin_smooth_E:<15.4f} {r_lin_near_E:<15.4f} {(r_lin_smooth_E - r_lin_near_E):<15.4f}")
    print(f"{'MLP (ReLU)':<15} {r_mlp_smooth_E:<15.4f} {r_mlp_near_E:<15.4f} {(r_mlp_smooth_E - r_mlp_near_E):<15.4f}")

    # =========================================
    # TEST 2: Predicting Frequency ω
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 2: Predicting Frequency ω (diverges at separatrix)")
    print("=" * 70)

    # Train Linear
    linear_omega = LinearRegressor()
    linear_omega.fit(X_smooth_norm, omega_smooth_norm)

    # Train MLP
    mlp_omega = train_mlp(X_smooth_norm, omega_smooth_norm, hidden_dims=[64, 32], epochs=3000)

    # Evaluate on smooth regime
    _, r_lin_smooth_w, _ = evaluate(linear_omega, X_smooth_norm, omega_smooth_norm)
    _, r_mlp_smooth_w, _ = evaluate(mlp_omega, X_smooth_norm, omega_smooth_norm, is_torch=True)

    # Evaluate on near-separatrix
    pred_lin_near_w, r_lin_near_w, _ = evaluate(linear_omega, X_near_norm, omega_near_norm)
    pred_mlp_near_w, r_mlp_near_w, _ = evaluate(mlp_omega, X_near_norm, omega_near_norm, is_torch=True)

    print(f"\n{'Method':<15} {'Smooth r(ω)':<15} {'Near-sep r(ω)':<15} {'Degradation':<15}")
    print("-" * 60)
    print(f"{'Linear':<15} {r_lin_smooth_w:<15.4f} {r_lin_near_w:<15.4f} {(r_lin_smooth_w - r_lin_near_w):<15.4f}")
    print(f"{'MLP (ReLU)':<15} {r_mlp_smooth_w:<15.4f} {r_mlp_near_w:<15.4f} {(r_mlp_smooth_w - r_mlp_near_w):<15.4f}")

    # =========================================
    # TEST 3: Residual Analysis
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 3: Residual Analysis (where do linear predictions fail?)")
    print("=" * 70)

    # Denormalize predictions
    pred_lin_E_denorm = pred_lin_near_E * E_std + E_mean
    pred_mlp_E_denorm = pred_mlp_near_E * E_std + E_mean

    # Compute residuals
    residuals_lin = np.abs(pred_lin_E_denorm - E_near_arr)
    residuals_mlp = np.abs(pred_mlp_E_denorm - E_near_arr)

    # Bin by energy to see pattern
    n_bins = 5
    bins = np.linspace(E_near_arr.min(), E_near_arr.max(), n_bins + 1)

    print(f"\n{'Energy bin':<20} {'Linear |resid|':<18} {'MLP |resid|':<18} {'MLP advantage':<15}")
    print("-" * 75)

    for i in range(n_bins):
        mask = (E_near_arr >= bins[i]) & (E_near_arr < bins[i+1])
        if mask.sum() > 0:
            mean_lin = residuals_lin[mask].mean()
            mean_mlp = residuals_mlp[mask].mean()
            advantage = (mean_lin - mean_mlp) / mean_lin * 100 if mean_lin > 0 else 0
            print(f"E ∈ [{bins[i]:.2f}, {bins[i+1]:.2f}]    {mean_lin:<18.4f} {mean_mlp:<18.4f} {advantage:>+10.1f}%")

    # =========================================
    # Visualization
    # =========================================
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: ω vs E (theory)
    ax1 = axes[0, 0]
    E_plot = np.linspace(-0.95, 0.99, 200)
    omega_plot = [theoretical_omega(e) for e in E_plot]
    ax1.plot(E_plot, omega_plot, 'b-', linewidth=2, label='Theoretical ω(E)')
    ax1.axvline(x=1.0, color='r', linestyle='--', label='Separatrix (E=1)')
    ax1.axvspan(0.7, 1.0, alpha=0.2, color='red', label='Near-separatrix regime')
    ax1.axvspan(-0.8, 0.3, alpha=0.2, color='green', label='Smooth regime')
    ax1.set_xlabel('Energy E')
    ax1.set_ylabel('Frequency ω')
    ax1.set_title('Pendulum: ω → 0 at Separatrix (Cusp!)')
    ax1.legend()
    ax1.set_xlim(-1, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predictions vs true E near separatrix
    ax2 = axes[0, 1]
    ax2.scatter(E_near_arr, pred_lin_E_denorm, alpha=0.6, label=f'Linear (r={r_lin_near_E:.3f})', marker='o')
    ax2.scatter(E_near_arr, pred_mlp_E_denorm, alpha=0.6, label=f'MLP (r={r_mlp_near_E:.3f})', marker='^')
    ax2.plot([E_near_arr.min(), E_near_arr.max()], [E_near_arr.min(), E_near_arr.max()], 'k--', label='Perfect')
    ax2.set_xlabel('True Energy E')
    ax2.set_ylabel('Predicted Energy')
    ax2.set_title('Near-Separatrix: E Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals vs E
    ax3 = axes[1, 0]
    ax3.scatter(E_near_arr, residuals_lin, alpha=0.6, label='Linear', marker='o')
    ax3.scatter(E_near_arr, residuals_mlp, alpha=0.6, label='MLP', marker='^')
    ax3.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('True Energy E')
    ax3.set_ylabel('|Prediction Error|')
    ax3.set_title('Residuals Near Separatrix')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison bar chart
    ax4 = axes[1, 1]
    methods = ['Linear', 'MLP']
    smooth_r = [r_lin_smooth_E, r_mlp_smooth_E]
    near_r = [r_lin_near_E, r_mlp_near_E]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax4.bar(x - width/2, smooth_r, width, label='Smooth regime', color='green', alpha=0.7)
    bars2 = ax4.bar(x + width/2, near_r, width, label='Near separatrix', color='red', alpha=0.7)

    ax4.set_ylabel('Correlation r(E)')
    ax4.set_title('Linear vs MLP: Smooth vs Cusp Regime')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/ubuntu/rectifier/linear_vs_mlp_cusp.png', dpi=150)
    print("Saved: linear_vs_mlp_cusp.png")

    # =========================================
    # Verdict
    # =========================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    mlp_advantage_E = r_mlp_near_E - r_lin_near_E
    mlp_advantage_omega = r_mlp_near_w - r_lin_near_w

    if mlp_advantage_E > 0.1 or mlp_advantage_omega > 0.1:
        print("\n\033[92mGLINSKY CUSP THEORY CONFIRMED!\033[0m")
        print(f"  MLP advantage for E near separatrix: {mlp_advantage_E:+.4f}")
        print(f"  MLP advantage for ω near separatrix: {mlp_advantage_omega:+.4f}")
        print("\n  Linear methods fail near singularities (cusps).")
        print("  MLP with ReLU captures the cusp structure.")
    elif mlp_advantage_E > 0.02 or mlp_advantage_omega > 0.02:
        print("\n\033[93mPARTIAL SUPPORT for cusp theory\033[0m")
        print(f"  MLP shows modest advantage: {mlp_advantage_E:+.4f} (E), {mlp_advantage_omega:+.4f} (ω)")
    else:
        print("\n\033[91mNO CLEAR MLP ADVANTAGE\033[0m")
        print("  Both methods perform similarly near separatrix.")
        print("  Either: (1) our phase-aware features already capture cusps,")
        print("          (2) need more data near E=1, or")
        print("          (3) cusp effect is subtle for this system.")

    # Summary statistics
    print("\n" + "-" * 70)
    print("Summary:")
    print(f"  Smooth regime:      Linear r={r_lin_smooth_E:.4f}, MLP r={r_mlp_smooth_E:.4f}")
    print(f"  Near separatrix:    Linear r={r_lin_near_E:.4f}, MLP r={r_mlp_near_E:.4f}")
    print(f"  Linear degradation: {r_lin_smooth_E - r_lin_near_E:.4f}")
    print(f"  MLP degradation:    {r_mlp_smooth_E - r_mlp_near_E:.4f}")

    return {
        'linear_smooth': (r_lin_smooth_E, r_lin_smooth_w),
        'linear_near': (r_lin_near_E, r_lin_near_w),
        'mlp_smooth': (r_mlp_smooth_E, r_mlp_smooth_w),
        'mlp_near': (r_mlp_near_E, r_mlp_near_w),
    }


if __name__ == "__main__":
    results = main()
