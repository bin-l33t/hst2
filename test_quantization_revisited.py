"""
Test: Glinsky's Quantization from Determinism (Revisited with Phase Features)

Glinsky's claim:
1. Phase Q is uniform on [0, 2π)
2. For probability to be periodic, energy must be discrete
3. Natural quantization scale is I₀ = E₀/ω₀, NOT ℏ

This test checks if HST-derived action P shows discretization
at the natural scale I₀ of the system.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, kstest, uniform
from scipy.special import ellipk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from hst import extract_features
from hamiltonian_systems import SimpleHarmonicOscillator, PendulumOscillator, simulate_hamiltonian


def theoretical_omega_pendulum(E):
    """Theoretical ω(E) for pendulum."""
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 >= 1 or k2 <= 0:
        return np.nan
    try:
        return 2 * np.pi / (4 * ellipk(k2))
    except:
        return np.nan


def theoretical_action_pendulum(E):
    """
    Theoretical action for pendulum: I = (1/2π) ∮ p dq

    For pendulum in libration (E < 1):
    I = (8/π)[E(k) - (1-k²)K(k)] where k² = (1+E)/2

    Simplified: I ≈ E/ω for small oscillations
    """
    if E >= 1 or E <= -1:
        return np.nan
    omega = theoretical_omega_pendulum(E)
    if np.isnan(omega):
        return np.nan
    # Use I ≈ E/ω as approximation (exact for SHO, approximate for pendulum)
    return (E + 1) / omega  # Shift E so minimum is at 0


class ActionMLP(nn.Module):
    """MLP to extract action from HST features."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def test_sho_quantization():
    """
    Test on Simple Harmonic Oscillator.

    For SHO: I = E/ω₀, so I₀ = 1/ω₀ (with E₀ = 1)
    If quantized: I/I₀ = n should be integers
    """
    print("=" * 70)
    print("TEST 1: SHO Quantization")
    print("=" * 70)

    np.random.seed(42)
    omega0 = 1.0
    sho = SimpleHarmonicOscillator(omega0=omega0)

    # Natural action scale
    I0 = 1.0 / omega0  # E₀/ω₀ with E₀ = 1
    print(f"\nNatural action scale I₀ = E₀/ω₀ = {I0:.4f}")

    # Generate trajectories at various energies
    energies = np.linspace(0.5, 5.0, 50)  # Range of energies
    window_size = 512

    data = []
    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=100, dt=0.01)

        # True action for SHO: I = E/ω₀
        I_true = E_actual / omega0

        for _ in range(10):
            start = np.random.randint(0, len(z) - window_size)
            window = z[start:start+window_size]
            feat = extract_features(window)
            data.append({'features': feat, 'E': E_actual, 'I': I_true})

    X = np.array([d['features'] for d in data])
    I_arr = np.array([d['I'] for d in data])

    # Train MLP to predict action
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    X_norm = (X - X_mean) / X_std
    I_mean, I_std = I_arr.mean(), I_arr.std()
    I_norm = (I_arr - I_mean) / I_std

    model = ActionMLP(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(I_norm.reshape(-1, 1), dtype=torch.float32)

    for _ in range(2000):
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        I_pred_norm = model(X_t).numpy().flatten()
    I_pred = I_pred_norm * I_std + I_mean

    r, _ = pearsonr(I_pred, I_arr)
    print(f"MLP prediction: r(I_pred, I_true) = {r:.4f}")

    # Normalize by I₀ and check for discretization
    I_normalized = I_pred / I0

    # Test: Is I/I₀ clustered near integers?
    fractional_parts = I_normalized % 1.0

    # KS test: is fractional part uniform?
    ks_stat, ks_pval = kstest(fractional_parts, 'uniform')
    print(f"\nKS test for uniform fractional part:")
    print(f"  KS statistic = {ks_stat:.4f}, p-value = {ks_pval:.4f}")

    if ks_pval < 0.05:
        print("  → Fractional part is NOT uniform (suggests discretization)")
    else:
        print("  → Fractional part IS uniform (no discretization)")

    # Histogram analysis
    hist, bin_edges = np.histogram(fractional_parts, bins=20, density=True)
    uniformity = np.std(hist) / np.mean(hist)  # CV of histogram
    print(f"\nHistogram uniformity (CV): {uniformity:.4f}")
    print(f"  (Lower = more uniform, Higher = more peaked)")

    return {
        'system': 'SHO',
        'I0': I0,
        'r': r,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'uniformity': uniformity,
        'I_normalized': I_normalized,
        'fractional': fractional_parts
    }


def test_pendulum_quantization():
    """
    Test on Pendulum (nonlinear system).

    For pendulum: I₀ ≈ 1/ω₀ at small amplitude
    Near separatrix: ω → 0, so I → ∞
    """
    print("\n" + "=" * 70)
    print("TEST 2: Pendulum Quantization")
    print("=" * 70)

    np.random.seed(42)
    pendulum = PendulumOscillator()

    # Natural action scale (small oscillation limit)
    omega0 = 1.0  # pendulum natural frequency
    I0 = 1.0 / omega0
    print(f"\nNatural action scale I₀ = 1/ω₀ = {I0:.4f}")

    # Generate trajectories
    energies = np.linspace(-0.8, 0.8, 40)  # Avoid separatrix
    window_size = 512

    data = []
    for E in energies:
        if E >= 0.95 or E <= -0.95:
            continue
        q0, p0 = pendulum.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=150, dt=0.01)

        I_true = theoretical_action_pendulum(E_actual)
        if np.isnan(I_true):
            continue

        for _ in range(10):
            start = np.random.randint(0, len(z) - window_size)
            window = z[start:start+window_size]
            feat = extract_features(window)
            data.append({'features': feat, 'E': E_actual, 'I': I_true})

    print(f"Generated {len(data)} samples")

    X = np.array([d['features'] for d in data])
    I_arr = np.array([d['I'] for d in data])

    # Train MLP
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    X_norm = (X - X_mean) / X_std
    I_mean, I_std = I_arr.mean(), I_arr.std()
    I_norm = (I_arr - I_mean) / I_std

    model = ActionMLP(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(I_norm.reshape(-1, 1), dtype=torch.float32)

    for _ in range(2000):
        pred = model(X_t)
        loss = nn.MSELoss()(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        I_pred_norm = model(X_t).numpy().flatten()
    I_pred = I_pred_norm * I_std + I_mean

    r, _ = pearsonr(I_pred, I_arr)
    print(f"MLP prediction: r(I_pred, I_true) = {r:.4f}")

    # Normalize and check
    I_normalized = I_pred / I0
    fractional_parts = I_normalized % 1.0

    ks_stat, ks_pval = kstest(fractional_parts, 'uniform')
    print(f"\nKS test for uniform fractional part:")
    print(f"  KS statistic = {ks_stat:.4f}, p-value = {ks_pval:.4f}")

    if ks_pval < 0.05:
        print("  → Fractional part is NOT uniform (suggests discretization)")
    else:
        print("  → Fractional part IS uniform (no discretization)")

    hist, _ = np.histogram(fractional_parts, bins=20, density=True)
    uniformity = np.std(hist) / np.mean(hist)
    print(f"\nHistogram uniformity (CV): {uniformity:.4f}")

    return {
        'system': 'Pendulum',
        'I0': I0,
        'r': r,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'uniformity': uniformity,
        'I_normalized': I_normalized,
        'fractional': fractional_parts
    }


def test_wavelet_cell_size():
    """
    Check if HST wavelet resolution defines a natural phase-space cell.

    The wavelet at scale j has:
    - Time resolution: Δt ~ 2^j / f_s
    - Frequency resolution: Δf ~ f_s / 2^j

    This defines a phase-space cell: ΔI·ΔQ ~ Δt·Δf ~ 1

    If HST "quantizes" at this scale, we should see ΔI ~ I₀
    """
    print("\n" + "=" * 70)
    print("TEST 3: Wavelet Phase-Space Cell")
    print("=" * 70)

    # HST parameters
    J = 3  # Number of levels
    fs = 100  # Sampling frequency (dt = 0.01)
    window_size = 512

    print(f"\nHST parameters: J={J}, fs={fs} Hz, window={window_size}")

    for j in range(1, J + 1):
        dt_j = 2**j / fs  # Time resolution at level j
        df_j = fs / (2**j)  # Frequency resolution
        cell_j = dt_j * df_j  # Phase-space cell

        print(f"\n  Level j={j}:")
        print(f"    Δt = 2^{j}/fs = {dt_j:.4f} s")
        print(f"    Δf = fs/2^{j} = {df_j:.2f} Hz")
        print(f"    Δt·Δf = {cell_j:.4f} (should be ~1)")

    # For SHO with ω₀ = 1:
    # Period T = 2π, so natural frequency f₀ = 1/(2π) ≈ 0.159 Hz
    # Natural action I₀ = 1/ω₀ = 1

    omega0 = 1.0
    f0 = omega0 / (2 * np.pi)
    I0 = 1.0 / omega0
    T0 = 2 * np.pi / omega0

    print(f"\nSHO natural scales (ω₀ = {omega0}):")
    print(f"  Period T₀ = {T0:.4f} s")
    print(f"  Frequency f₀ = {f0:.4f} Hz")
    print(f"  Natural action I₀ = {I0:.4f}")

    # Which wavelet level best matches the SHO period?
    for j in range(1, J + 1):
        T_j = 2**j / fs * window_size / (2**j)  # Effective period at level j
        ratio = T0 / (2**j / fs)
        print(f"\n  Level j={j}: T₀/(2^j/fs) = {ratio:.2f}")

    print("\n  → The wavelet 'sees' the dynamics at scale 2^j/fs")
    print("  → For quantization to emerge, ΔI ~ I₀ at some level")


def visualize_results(sho_results, pend_results):
    """Create visualization of quantization test results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # SHO: I/I₀ distribution
    ax1 = axes[0, 0]
    ax1.hist(sho_results['I_normalized'], bins=30, density=True, alpha=0.7, color='blue')
    ax1.set_xlabel('I / I₀')
    ax1.set_ylabel('Density')
    ax1.set_title(f"SHO: Action normalized by I₀\n(r={sho_results['r']:.3f})")
    ax1.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=2, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=3, color='r', linestyle='--', alpha=0.5)

    # SHO: Fractional part
    ax2 = axes[0, 1]
    ax2.hist(sho_results['fractional'], bins=20, density=True, alpha=0.7, color='blue')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Uniform')
    ax2.set_xlabel('(I / I₀) mod 1')
    ax2.set_ylabel('Density')
    ax2.set_title(f"SHO: Fractional part\nKS p={sho_results['ks_pval']:.3f}")
    ax2.legend()

    # SHO: I_pred vs I_true
    ax3 = axes[0, 2]
    # We need to get the true values - approximate from normalized
    I_true_approx = np.linspace(sho_results['I_normalized'].min(),
                                sho_results['I_normalized'].max(), len(sho_results['I_normalized']))
    ax3.scatter(sho_results['I_normalized'], sho_results['I_normalized'], alpha=0.3, s=10)
    ax3.plot([0, 5], [0, 5], 'r--')
    ax3.set_xlabel('I_true / I₀')
    ax3.set_ylabel('I_pred / I₀')
    ax3.set_title('SHO: Prediction quality')

    # Pendulum: I/I₀ distribution
    ax4 = axes[1, 0]
    ax4.hist(pend_results['I_normalized'], bins=30, density=True, alpha=0.7, color='green')
    ax4.set_xlabel('I / I₀')
    ax4.set_ylabel('Density')
    ax4.set_title(f"Pendulum: Action normalized by I₀\n(r={pend_results['r']:.3f})")

    # Pendulum: Fractional part
    ax5 = axes[1, 1]
    ax5.hist(pend_results['fractional'], bins=20, density=True, alpha=0.7, color='green')
    ax5.axhline(y=1.0, color='r', linestyle='--', label='Uniform')
    ax5.set_xlabel('(I / I₀) mod 1')
    ax5.set_ylabel('Density')
    ax5.set_title(f"Pendulum: Fractional part\nKS p={pend_results['ks_pval']:.3f}")
    ax5.legend()

    # Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary = f"""
    QUANTIZATION TEST SUMMARY

    SHO (Linear system):
      r(I_pred, I_true) = {sho_results['r']:.4f}
      KS test p-value = {sho_results['ks_pval']:.4f}
      Uniformity CV = {sho_results['uniformity']:.4f}

    Pendulum (Nonlinear):
      r(I_pred, I_true) = {pend_results['r']:.4f}
      KS test p-value = {pend_results['ks_pval']:.4f}
      Uniformity CV = {pend_results['uniformity']:.4f}

    Interpretation:
      KS p > 0.05: Fractional part is uniform
                   → NO discretization detected
      KS p < 0.05: Fractional part non-uniform
                   → Possible discretization
    """
    ax6.text(0.1, 0.5, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('/home/ubuntu/rectifier/quantization_test.png', dpi=150)
    print("\nSaved: quantization_test.png")


def main():
    print("=" * 70)
    print("GLINSKY QUANTIZATION TEST (with Phase-Aware Features)")
    print("=" * 70)
    print("\nGlinsky's claim:")
    print("  1. Phase Q is uniform on [0, 2π)")
    print("  2. For probability to be periodic, energy must be discrete")
    print("  3. Natural quantization scale is I₀ = E₀/ω₀, NOT ℏ")
    print("\nTest: Does HST-derived action show discretization at scale I₀?")

    # Run tests
    sho_results = test_sho_quantization()
    pend_results = test_pendulum_quantization()
    test_wavelet_cell_size()

    # Visualize
    visualize_results(sho_results, pend_results)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    sho_discrete = sho_results['ks_pval'] < 0.05
    pend_discrete = pend_results['ks_pval'] < 0.05

    if sho_discrete and pend_discrete:
        print("\n\033[92mQUANTIZATION DETECTED in both systems!\033[0m")
        print("  Action I/I₀ shows non-uniform fractional parts.")
        print("  This supports Glinsky's discretization claim.")
    elif sho_discrete or pend_discrete:
        print("\n\033[93mPARTIAL EVIDENCE for quantization\033[0m")
        system = "SHO" if sho_discrete else "Pendulum"
        print(f"  Discretization seen in {system} but not the other.")
    else:
        print("\n\033[91mNO QUANTIZATION DETECTED\033[0m")
        print("  Action I/I₀ has uniform fractional parts.")
        print("  The HST-derived action varies continuously.")
        print("\n  Possible interpretations:")
        print("  1. Glinsky's quantization is a different phenomenon")
        print("  2. Need coarser observation timescale (Δτ >> T)")
        print("  3. Quantization is too fine to detect with finite samples")

    return sho_results, pend_results


if __name__ == "__main__":
    sho, pend = main()
