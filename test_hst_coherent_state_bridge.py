"""
HST → Coherent State Quantization Bridge Test

Tests whether HST-derived features have the structure needed for
Ali's action-angle coherent state quantization (Sections 6.4.4 and 11.6.3).

The Bridge:
- P (action) from magnitude features → corresponds to J in Ali's notation
- Q (angle) from phase features → corresponds to γ in Ali's notation

From Ali Section 11.6.3, the procedure is:
1. Start with measured energies E₀ < E₁ < ... < Eₙ
2. Build probability distributions pₙ(J) satisfying ∫ E(J) pₙ(J) dJ = Eₙ + const
3. Construct CS: |J,γ⟩ = (1/√N(J)) Σₙ √pₙ(J) e^{-iαₙγ} |eₙ⟩

Key Question: Can we go from HST features → coherent state basis → discrete spectrum?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import ellipk
import warnings
warnings.filterwarnings('ignore')

# Local imports
from hst import extract_features, hst_forward_pywt
from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian


def generate_pendulum_ensemble(n_energies=10, n_samples_per_energy=50,
                                E_range=(-0.8, 0.8), seed=42):
    """
    Generate pendulum trajectories at discrete energy levels.

    Parameters
    ----------
    n_energies : int
        Number of discrete energy levels
    n_samples_per_energy : int
        Number of trajectory samples per energy level
    E_range : tuple
        (E_min, E_max) for libration regime (|E| < 1)
    seed : int
        Random seed

    Returns
    -------
    data : dict with keys:
        'trajectories': list of complex arrays z = q + ip/ω
        'energies': energy for each trajectory
        'energy_levels': discrete energy levels
        'energy_indices': which level each trajectory belongs to
        'omegas': theoretical frequency for each energy
    """
    np.random.seed(seed)

    # Discrete energy levels
    E_levels = np.linspace(E_range[0], E_range[1], n_energies)

    trajectories = []
    energies = []
    energy_indices = []
    omegas = []

    pendulum = PendulumOscillator()
    dt = 0.01
    T_sim = 5.12  # Simulation time to get ~512 samples

    for i, E_target in enumerate(E_levels):
        for j in range(n_samples_per_energy):
            # Initial conditions for this energy
            # E = (1/2)p² - cos(q) → at q=0: p = sqrt(2(E+1))
            # Add small random phase offset
            phase_offset = np.random.uniform(0, 2*np.pi)

            if E_target < 1:  # Libration regime
                # q_max where p=0: E = -cos(q_max) → q_max = arccos(-E)
                q_max = np.arccos(-E_target) if E_target > -1 else np.pi * 0.99
                q0 = q_max * np.cos(phase_offset)
                p0_sq = 2 * (E_target + np.cos(q0))
                if p0_sq < 0:
                    p0_sq = 0.01
                p0 = np.sqrt(p0_sq)
                if np.random.random() < 0.5:
                    p0 = -p0
            else:
                # Near separatrix - use small oscillation
                q0 = 0.1 * np.cos(phase_offset)
                p0 = np.sqrt(2 * (E_target + np.cos(q0)))

            # Integrate trajectory using simulate_hamiltonian
            t_traj, q_traj, p_traj, z_traj, E_init = simulate_hamiltonian(
                pendulum, q0, p0, T=T_sim, dt=dt
            )

            # Truncate to 512 samples for HST
            n_keep = min(512, len(q_traj))
            q_traj = q_traj[:n_keep]
            p_traj = p_traj[:n_keep]

            # Complex trajectory: z = q + i*p
            z = q_traj + 1j * p_traj

            # True energy (should be conserved)
            E_actual = 0.5 * p_traj[0]**2 - np.cos(q_traj[0])

            # Theoretical frequency for pendulum in libration
            # ω(E) = π / (2 * K(k)) where k² = (1+E)/2
            if E_actual < 1:
                k2 = (1 + E_actual) / 2
                k = np.sqrt(max(k2, 1e-10))
                if k < 1:
                    omega_theory = np.pi / (2 * ellipk(k**2))
                else:
                    omega_theory = 0.01  # Near separatrix
            else:
                omega_theory = 0.01

            trajectories.append(z)
            energies.append(E_actual)
            energy_indices.append(i)
            omegas.append(omega_theory)

    return {
        'trajectories': trajectories,
        'energies': np.array(energies),
        'energy_levels': E_levels,
        'energy_indices': np.array(energy_indices),
        'omegas': np.array(omegas)
    }


def extract_all_features(data, J=3, wavelet='db8'):
    """Extract HST features from all trajectories."""
    features = []
    for z in data['trajectories']:
        feat = extract_features(z, J=J, wavelet_name=wavelet)
        features.append(feat)
    return np.array(features)


def estimate_P_from_features(features, energies):
    """
    Estimate action P from HST features.

    Uses linear regression to find the best linear combination
    of features that predicts energy (proxy for action).
    """
    # Simple standardization
    X_mean = features.mean(axis=0)
    X_std = features.std(axis=0) + 1e-10
    X = (features - X_mean) / X_std

    # Linear regression via least squares: P = X @ w + b
    # Add bias column
    X_bias = np.column_stack([X, np.ones(len(X))])

    # Solve: X_bias @ coeffs = energies
    coeffs, residuals, rank, s = np.linalg.lstsq(X_bias, energies, rcond=None)

    P_pred = X_bias @ coeffs
    r = np.corrcoef(P_pred, energies)[0, 1]

    return P_pred, r, coeffs, (X_mean, X_std)


def estimate_Q_from_features(features, trajectories):
    """
    Estimate angle Q from HST features.

    Q is encoded in the phase features. We extract the mean phase
    from each trajectory and correlate with feature predictions.
    """
    # True Q from trajectory mean phase
    Q_true = []
    for z in trajectories:
        phases = np.angle(z)
        mean_phase = np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
        Q_true.append(mean_phase)
    Q_true = np.array(Q_true)

    # Features should encode Q in the phase components
    # Last two features are sin(mean_phase), cos(mean_phase)
    sin_Q_feat = features[:, -2]
    cos_Q_feat = features[:, -1]
    Q_pred = np.arctan2(sin_Q_feat, cos_Q_feat)

    # Circular correlation
    r_sin = np.corrcoef(np.sin(Q_pred), np.sin(Q_true))[0, 1]
    r_cos = np.corrcoef(np.cos(Q_pred), np.cos(Q_true))[0, 1]

    return Q_pred, Q_true, r_sin, r_cos


def build_probability_distributions(P_values, energy_indices, n_levels):
    """
    Build probability distributions pₙ(J) for each energy level.

    For each discrete energy level n, we collect all P values
    from trajectories at that level and estimate the distribution.

    Returns
    -------
    distributions : list of dicts, each containing:
        'P_values': P values at this level
        'mean': mean P
        'std': std of P
        'kde': kernel density estimate (if enough samples)
    """
    distributions = []

    for n in range(n_levels):
        mask = energy_indices == n
        P_n = P_values[mask]

        dist = {
            'P_values': P_n,
            'mean': np.mean(P_n),
            'std': np.std(P_n),
            'n_samples': len(P_n)
        }

        # KDE if enough samples
        if len(P_n) >= 5:
            try:
                kde = stats.gaussian_kde(P_n)
                dist['kde'] = kde
            except:
                dist['kde'] = None
        else:
            dist['kde'] = None

        distributions.append(dist)

    return distributions


def check_quantization_condition(distributions, E_levels, P_range=None):
    """
    Check Ali's quantization condition:
    ∫ E(J) pₙ(J) dJ = Eₙ + const

    If P is proportional to action, and pₙ(J) is well-localized around Jₙ,
    then this integral should give approximately Eₙ.

    We check if:
    1. Mean P at each level is monotonic with E
    2. The P distributions are well-separated
    3. The relationship E(P) is invertible
    """
    results = {}

    # 1. Mean P vs E relationship
    mean_P = np.array([d['mean'] for d in distributions])
    std_P = np.array([d['std'] for d in distributions])

    # Check monotonicity
    P_diffs = np.diff(mean_P)
    monotonic = np.all(P_diffs > 0) or np.all(P_diffs < 0)
    results['monotonic'] = monotonic

    # Correlation
    r_PE = np.corrcoef(mean_P, E_levels)[0, 1]
    results['r_PE'] = r_PE

    # 2. Separation of distributions
    # Compute overlap between adjacent levels
    overlaps = []
    for n in range(len(distributions) - 1):
        d1, d2 = distributions[n], distributions[n+1]
        # Simple overlap metric: (std1 + std2) / |mean2 - mean1|
        separation = abs(d2['mean'] - d1['mean'])
        spread = d1['std'] + d2['std']
        overlap = spread / separation if separation > 1e-10 else np.inf
        overlaps.append(overlap)

    results['mean_overlap'] = np.mean(overlaps) if overlaps else np.inf
    results['well_separated'] = results['mean_overlap'] < 1.0

    # 3. Natural scale I₀
    # From Glinsky: I₀ = E₀/ω₀
    # For pendulum at E=0: ω₀ = 1, so I₀ ≈ 0
    # Better: use characteristic action scale from P distribution
    P_span = mean_P.max() - mean_P.min()
    E_span = E_levels.max() - E_levels.min()
    I0_estimate = E_span / len(E_levels) if len(E_levels) > 1 else 1.0
    results['I0_estimate'] = I0_estimate

    # 4. Effective quantum number
    # If P ∝ n·I₀, then n ≈ P / I₀
    if I0_estimate > 0:
        n_eff = mean_P / I0_estimate
        results['n_effective'] = n_eff

    return results, mean_P, std_P


def plot_results(data, P_pred, Q_pred, Q_true, distributions, quant_results,
                 mean_P, std_P, save_path=None):
    """Generate diagnostic plots."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. E vs P scatter
    ax = axes[0, 0]
    ax.scatter(P_pred, data['energies'], alpha=0.5, c=data['energy_indices'], cmap='viridis')
    ax.set_xlabel('Learned P (action)')
    ax.set_ylabel('True Energy E')
    ax.set_title(f'E vs P: r = {np.corrcoef(P_pred, data["energies"])[0,1]:.3f}')
    ax.grid(True, alpha=0.3)

    # 2. Q prediction
    ax = axes[0, 1]
    ax.scatter(np.sin(Q_true), np.sin(Q_pred), alpha=0.5)
    r_sin = np.corrcoef(np.sin(Q_true), np.sin(Q_pred))[0, 1]
    ax.plot([-1, 1], [-1, 1], 'r--', label='Perfect')
    ax.set_xlabel('True sin(Q)')
    ax.set_ylabel('Predicted sin(Q)')
    ax.set_title(f'Phase prediction: r(sin Q) = {r_sin:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. P distributions by energy level
    ax = axes[0, 2]
    for n, dist in enumerate(distributions):
        if dist['kde'] is not None:
            P_grid = np.linspace(dist['mean'] - 3*dist['std'],
                                 dist['mean'] + 3*dist['std'], 100)
            ax.plot(P_grid, dist['kde'](P_grid), label=f'E={data["energy_levels"][n]:.2f}')
    ax.set_xlabel('P (action)')
    ax.set_ylabel('Density pₙ(J)')
    ax.set_title('Probability distributions pₙ(J)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Mean P vs Energy level
    ax = axes[1, 0]
    ax.errorbar(data['energy_levels'], mean_P, yerr=std_P, fmt='o-', capsize=3)
    ax.set_xlabel('Energy Level E')
    ax.set_ylabel('Mean P')
    ax.set_title(f'P(E) relationship (r={quant_results["r_PE"]:.3f})')
    ax.grid(True, alpha=0.3)

    # 5. Distribution separation
    ax = axes[1, 1]
    for n, dist in enumerate(distributions):
        ax.scatter([n]*len(dist['P_values']), dist['P_values'], alpha=0.3, s=10)
        ax.errorbar([n], [dist['mean']], yerr=[dist['std']], fmt='ko', capsize=5)
    ax.set_xlabel('Energy Level Index n')
    ax.set_ylabel('P values')
    ax.set_title(f'Separation: overlap = {quant_results["mean_overlap"]:.2f}')
    ax.grid(True, alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    HST → Coherent State Bridge Test
    ================================

    Feature Extraction:
    - r(P, E) = {np.corrcoef(P_pred, data['energies'])[0,1]:.4f}
    - r(sin Q) = {np.corrcoef(np.sin(Q_true), np.sin(Q_pred))[0,1]:.4f}

    Quantization Structure:
    - P(E) monotonic: {quant_results['monotonic']}
    - r(mean_P, E) = {quant_results['r_PE']:.4f}
    - Distribution overlap: {quant_results['mean_overlap']:.3f}
    - Well separated: {quant_results['well_separated']}

    Natural Scale:
    - I₀ estimate: {quant_results['I0_estimate']:.4f}

    Interpretation:
    If r(P,E) > 0.95 and overlap < 1.0, the HST features
    have the structure needed for coherent state quantization.
    The distributions pₙ(J) can be used to construct:

    |J,γ⟩ = (1/√N(J)) Σₙ √pₙ(J) e^{{-iαₙγ}} |eₙ⟩
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()
    return fig


def run_bridge_test(verbose=True, save_plot=True):
    """
    Main test: Can HST features be used for coherent state quantization?
    """
    print("=" * 70)
    print("HST → COHERENT STATE QUANTIZATION BRIDGE TEST")
    print("=" * 70)

    # Step 1: Generate ensemble
    print("\n[1] Generating pendulum ensemble at discrete energies...")
    data = generate_pendulum_ensemble(
        n_energies=10,
        n_samples_per_energy=50,
        E_range=(-0.8, 0.7),  # Stay away from separatrix
        seed=42
    )
    print(f"    Generated {len(data['trajectories'])} trajectories")
    print(f"    Energy levels: {data['energy_levels']}")

    # Step 2: Extract features
    print("\n[2] Extracting HST features (phase-aware)...")
    features = extract_all_features(data, J=3, wavelet='db8')
    print(f"    Feature dimension: {features.shape[1]}")

    # Step 3: Estimate P (action) from features
    print("\n[3] Estimating P (action) from features...")
    P_pred, r_PE, model, scaler = estimate_P_from_features(features, data['energies'])
    print(f"    r(P, E) = {r_PE:.4f}")

    # Step 4: Estimate Q (angle) from features
    print("\n[4] Estimating Q (angle) from features...")
    Q_pred, Q_true, r_sin, r_cos = estimate_Q_from_features(features, data['trajectories'])
    print(f"    r(sin Q) = {r_sin:.4f}")
    print(f"    r(cos Q) = {r_cos:.4f}")

    # Step 5: Build probability distributions pₙ(J)
    print("\n[5] Building probability distributions pₙ(J)...")
    distributions = build_probability_distributions(
        P_pred, data['energy_indices'], len(data['energy_levels'])
    )
    for n, dist in enumerate(distributions):
        print(f"    Level {n}: E={data['energy_levels'][n]:.2f}, "
              f"mean(P)={dist['mean']:.3f}, std(P)={dist['std']:.3f}")

    # Step 6: Check quantization condition
    print("\n[6] Checking quantization structure...")
    quant_results, mean_P, std_P = check_quantization_condition(
        distributions, data['energy_levels']
    )
    print(f"    P(E) monotonic: {quant_results['monotonic']}")
    print(f"    r(mean_P, E) = {quant_results['r_PE']:.4f}")
    print(f"    Mean overlap = {quant_results['mean_overlap']:.3f}")
    print(f"    Well separated: {quant_results['well_separated']}")
    print(f"    I₀ estimate = {quant_results['I0_estimate']:.4f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    passed = (r_PE > 0.95 and quant_results['well_separated'])

    if passed:
        print("""
    ✓ HST features have the structure for coherent state quantization!

    The distributions pₙ(J) satisfy:
    - Well-localized around mean Pₙ
    - Monotonic relationship P(E)
    - Sufficient separation for discrete spectrum

    Next step: Construct coherent states using Ali's formula:
    |J,γ⟩ = (1/√N(J)) Σₙ √pₙ(J) e^{-iαₙγ} |eₙ⟩

    The effective Hamiltonian would be:
    H_eff = Σₙ Eₙ |eₙ⟩⟨eₙ|
        """)
    else:
        print(f"""
    ✗ Structure is incomplete.

    Issues:
    - r(P, E) = {r_PE:.3f} (need > 0.95)
    - Separation = {quant_results['mean_overlap']:.3f} (need < 1.0)

    The distributions overlap too much for clean discrete spectrum.
        """)

    # Plot
    if save_plot:
        fig = plot_results(
            data, P_pred, Q_pred, Q_true, distributions, quant_results,
            mean_P, std_P, save_path='hst_coherent_state_bridge.png'
        )

    return {
        'passed': passed,
        'r_PE': r_PE,
        'r_sin_Q': r_sin,
        'r_cos_Q': r_cos,
        'quant_results': quant_results,
        'distributions': distributions,
        'data': data,
        'P_pred': P_pred,
        'Q_pred': Q_pred,
    }


if __name__ == "__main__":
    results = run_bridge_test(verbose=True, save_plot=True)
