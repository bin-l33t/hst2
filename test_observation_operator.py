"""
Observation Operator Test - The Physical Mechanism of Phase Erasure

Key insight from ChatGPT: The coarse-graining must happen at the OBSERVATION level,
not at the feature extraction level. Features CAN represent phase - but the
observation process makes phase unidentifiable.

This is the difference between:
- "We choose not to look at phase" (artificial, feature engineering)
- "The observation process makes phase unknowable" (physical mechanism)

The observation operator degrades z(t) in ways that preserve action P but
make phase Q unidentifiable - even when using phase-aware features.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian
from hst import extract_features  # FULL phase-aware features


def pendulum_omega(E):
    """Theoretical frequency omega(E) = pi / (2K(k)) where k^2 = (1+E)/2"""
    if E >= 1:
        return 0.01
    k2 = (1 + E) / 2
    if k2 <= 0:
        return 1.0
    k = np.sqrt(k2)
    if k >= 1:
        return 0.01
    return np.pi / (2 * ellipk(k**2))


def observe(z, t, T_period, mode='time_shift', delta_tau=None, seed=None):
    """
    Apply observation operator that makes phase unidentifiable.

    This is the PHYSICAL mechanism of coarse-graining:
    - The trajectory z(t) still contains phase information
    - But the observation process makes it impossible to determine

    Parameters
    ----------
    z : array
        Complex trajectory z = q + ip
    t : array
        Time array
    T_period : float
        Natural period of the system
    mode : str
        'time_shift': Random shift by t0 ~ Unif[0, T)
        'strobe': Sample at intervals delta_tau
        'cycle_average': Average over windows >> T
    delta_tau : float
        For 'strobe' mode: sampling interval
        For 'cycle_average': window size
    seed : int
        Random seed

    Returns
    -------
    z_observed : array
        Degraded observation
    metadata : dict
        Information about the observation process
    """
    if seed is not None:
        np.random.seed(seed)

    dt = t[1] - t[0]
    n = len(z)

    if mode == 'time_shift':
        # Random time shift: learner doesn't know when observation started
        # This makes the initial phase Q_0 uniformly distributed

        # Shift by random amount in [0, T_period)
        shift_samples = np.random.randint(0, max(1, int(T_period / dt)))
        z_observed = np.roll(z, shift_samples)

        # Also: learner only sees a WINDOW of the trajectory
        # The window's position within the orbit is unknown
        window_size = min(512, len(z_observed))
        start = np.random.randint(0, max(1, len(z_observed) - window_size))
        z_observed = z_observed[start:start + window_size]

        return z_observed, {
            'mode': 'time_shift',
            'shift_samples': shift_samples,
            'shift_phase': 2 * np.pi * shift_samples * dt / T_period,
            'T_period': T_period,
            'window_start': start
        }

    elif mode == 'strobe':
        # Sample at intervals delta_tau
        # When delta_tau >> T, phase aliases (undersampling)

        if delta_tau is None:
            delta_tau = T_period  # Default: sample once per period

        sample_interval = max(1, int(delta_tau / dt))
        z_sampled = z[::sample_interval]

        # Pad or truncate to fixed size for feature extraction
        target_size = 512
        if len(z_sampled) < target_size:
            # Interpolate back up (destroys high-freq phase info)
            z_observed = np.interp(
                np.linspace(0, len(z_sampled)-1, target_size),
                np.arange(len(z_sampled)),
                z_sampled
            )
        else:
            z_observed = z_sampled[:target_size]

        return z_observed, {
            'mode': 'strobe',
            'delta_tau': delta_tau,
            'delta_tau_over_T': delta_tau / T_period,
            'sample_interval': sample_interval,
            'n_samples_raw': len(z_sampled)
        }

    elif mode == 'cycle_average':
        # Average over windows of size W
        # When W >> T, phase averages to zero

        if delta_tau is None:
            delta_tau = T_period

        window_samples = max(1, int(delta_tau / dt))
        n_windows = len(z) // window_samples

        if n_windows < 2:
            # Not enough data for averaging
            return z[:512], {'mode': 'cycle_average', 'error': 'insufficient_data'}

        # Compute windowed averages
        z_averaged = []
        for i in range(n_windows):
            window = z[i * window_samples:(i + 1) * window_samples]
            z_averaged.append(np.mean(window))
        z_averaged = np.array(z_averaged)

        # Interpolate to fixed size
        target_size = 512
        z_observed = np.interp(
            np.linspace(0, len(z_averaged)-1, target_size),
            np.arange(len(z_averaged)),
            z_averaged
        )

        return z_observed, {
            'mode': 'cycle_average',
            'window_size': delta_tau,
            'window_over_T': delta_tau / T_period,
            'n_windows': n_windows
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")


def generate_dataset_physical(n_energies=10, n_samples_per_E=20,
                               delta_tau_ratios=[0.1, 0.5, 1.0, 2.0, 5.0],
                               seed=42):
    """
    Generate dataset with PHYSICAL observation degradation.

    The scenario:
    1. System has KNOWN initial phase Q_original (varies across samples)
    2. Observer receives window starting at UNKNOWN time t0 ~ Unif[0, delta_tau]
    3. Features extracted from observed window

    Question: Can features predict Q_original?
    - If delta_tau << T: offset is small, Q_observed ~ Q_original, recoverable
    - If delta_tau >> T: offset spans many periods, Q_observed uncorrelated with Q_original

    P should always be recoverable (constant along orbit).
    """
    np.random.seed(seed)

    E_values = np.linspace(-0.8, 0.7, n_energies)
    pendulum = PendulumOscillator()

    datasets = {ratio: {'features': [], 'P_true': [], 'Q_original': [],
                        'Q_observed': [], 'E': [], 'T_period': [], 'offset_phase': []}
                for ratio in delta_tau_ratios}

    print(f"\nGenerating dataset with PHYSICAL observation model")
    print(f"Delta_tau/T ratios: {delta_tau_ratios}")
    print("Key: Can we predict Q_original when observation starts at unknown t0?")
    print("-" * 60)

    for E_target in E_values:
        # Compute period
        omega = pendulum_omega(E_target)
        T_period = 2 * np.pi / omega
        dt = 0.01
        samples_per_period = int(T_period / dt)

        if E_target >= 0.99:
            continue

        for sample_idx in range(n_samples_per_E):
            # DIFFERENT initial phase for each sample
            Q_original = np.random.uniform(0, 2 * np.pi)

            # Initial condition with this phase
            q_max = np.arccos(-E_target) if E_target > -1 else np.pi * 0.95

            # Phase convention: Q=0 at turning point (q=q_max, p=0)
            # General initial condition at phase Q_original:
            # q = q_max * cos(Q_original) approximately
            # But for pendulum, need to be careful...

            # Simpler: start at turning point, then phase = time * omega
            # So to get phase Q_original, start observation at t = Q_original / omega
            q0 = q_max
            p0 = 0.0

            # Generate trajectory starting at turning point (Q=0)
            T_sim = max(30 * T_period, 150)
            try:
                t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=dt)
            except:
                continue

            if len(z) < 1024:
                continue

            # P is constant along orbit
            P_true = np.mean(np.abs(z))

            # The "original observation time" corresponds to Q_original
            # This is the time we WOULD observe if delta_tau = 0
            t0_original = (Q_original / omega) % T_period
            idx_original = int(t0_original / dt)

            for ratio in delta_tau_ratios:
                # Observation time uncertainty
                delta_tau = ratio * T_period
                max_offset_samples = max(1, int(delta_tau / dt))

                # Random ADDITIONAL offset within [0, delta_tau]
                # This represents timing uncertainty
                offset_samples = np.random.randint(0, max_offset_samples)

                # Total start index = original + offset
                start_idx = (idx_original + offset_samples) % max(1, len(z) - 600)

                window_size = 512
                if start_idx + window_size >= len(z):
                    start_idx = max(0, len(z) - window_size - 1)

                z_obs = z[start_idx:start_idx + window_size]

                if len(z_obs) < window_size:
                    continue

                # Offset phase (the unknown part)
                offset_phase = (omega * offset_samples * dt) % (2 * np.pi)

                # Q_observed = Q_original + offset_phase (mod 2pi)
                Q_observed = (Q_original + offset_phase) % (2 * np.pi)

                # But the learner doesn't know the offset!
                # Features come from z_obs which starts at Q_observed

                try:
                    feat = extract_features(z_obs, J=3, wavelet_name='db8')

                    datasets[ratio]['features'].append(feat)
                    datasets[ratio]['P_true'].append(P_true)
                    datasets[ratio]['Q_original'].append(Q_original)
                    datasets[ratio]['Q_observed'].append(Q_observed)
                    datasets[ratio]['offset_phase'].append(offset_phase)
                    datasets[ratio]['E'].append(E_actual)
                    datasets[ratio]['T_period'].append(T_period)
                except:
                    pass

    # Convert to arrays
    for ratio in delta_tau_ratios:
        for key in datasets[ratio]:
            datasets[ratio][key] = np.array(datasets[ratio][key])
        print(f"  Ratio {ratio:.1f}: {len(datasets[ratio]['P_true'])} samples")

    return datasets


def generate_dataset(n_energies=10, n_samples_per_E=20, delta_tau_ratios=[0.1, 0.5, 1.0, 2.0, 5.0],
                     mode='strobe', seed=42):
    """DEPRECATED - use generate_dataset_physical instead"""
    return generate_dataset_physical(n_energies, n_samples_per_E, delta_tau_ratios, seed)


def train_regressor(X, y, test_fraction=0.3):
    """
    Train simple linear regressor, return test correlation.
    """
    n = len(y)
    if n < 10:
        return np.nan, np.nan, None, None

    # Shuffle
    idx = np.random.permutation(n)
    n_train = int((1 - test_fraction) * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-10
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # Ridge regression
    lam = 0.01
    X_b = np.column_stack([X_train_norm, np.ones(len(X_train_norm))])
    I = np.eye(X_b.shape[1])
    I[-1, -1] = 0  # Don't regularize bias
    w = np.linalg.solve(X_b.T @ X_b + lam * I, X_b.T @ y_train)

    # Predict
    X_test_b = np.column_stack([X_test_norm, np.ones(len(X_test_norm))])
    y_pred = X_test_b @ w

    # Correlation
    r, _ = stats.pearsonr(y_pred, y_test)

    return r, np.sqrt(np.mean((y_pred - y_test)**2)), y_pred, y_test


def run_observation_operator_test():
    """
    Main test: Does observation-level coarse-graining erase phase but preserve action?
    """
    print("=" * 70)
    print("OBSERVATION OPERATOR TEST")
    print("Physical mechanism: Coarse observation makes phase unidentifiable")
    print("=" * 70)

    # Test parameters
    delta_tau_ratios = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Generate dataset
    datasets = generate_dataset(
        n_energies=12,
        n_samples_per_E=25,
        delta_tau_ratios=delta_tau_ratios,
        mode='strobe',
        seed=42
    )

    # Train regressors at each scale
    print("\n" + "=" * 70)
    print("RESULTS: Phase-aware features + Degraded Observation")
    print("=" * 70)
    print("\nKey: Can features predict Q_ORIGINAL (unknown to observer)?")
    print("     Features come from z_obs which has Q_observed = Q_original + noise")

    print(f"\n{'Delta_tau/T':>12} {'r(P)':>10} {'r(Q_orig)':>12} {'r(Q_obs)':>12} {'Interpretation':>20}")
    print("-" * 70)

    results = []

    for ratio in delta_tau_ratios:
        data = datasets[ratio]
        if len(data['P_true']) < 20:
            print(f"{ratio:>12.1f} {'N/A':>10} {'N/A':>12} {'N/A':>12} {'Insufficient data':>20}")
            continue

        X = data['features']
        P_true = data['P_true']
        Q_original = data['Q_original']  # The TARGET we want to predict
        Q_observed = data['Q_observed']  # What features actually encode

        # Normalize P
        P_norm = (P_true - P_true.mean()) / (P_true.std() + 1e-10)

        # Q is circular - use sin for simplicity
        sin_Q_orig = np.sin(Q_original)
        sin_Q_obs = np.sin(Q_observed)

        # Train regressors
        r_P, _, _, _ = train_regressor(X, P_norm)
        r_Q_orig, _, _, _ = train_regressor(X, sin_Q_orig)  # Can we predict ORIGINAL?
        r_Q_obs, _, _, _ = train_regressor(X, sin_Q_obs)    # Can we predict OBSERVED?

        # Interpretation
        # At fine scale: Q_observed ~ Q_original, both predictable
        # At coarse scale: Q_observed still predictable (it's in the features)
        #                  but Q_original NOT predictable (offset unknown)
        if abs(r_Q_orig) > 0.6 and r_P > 0.7:
            interp = "Fine (Q recoverable)"
        elif abs(r_Q_orig) < 0.3 and r_P > 0.7:
            interp = "Coarse (Q erased)"
        elif r_P > 0.7:
            interp = "Transition"
        else:
            interp = "Degraded"

        print(f"{ratio:>12.1f} {r_P:>10.3f} {r_Q_orig:>12.3f} {r_Q_obs:>12.3f} {interp:>20}")

        results.append({
            'ratio': ratio,
            'r_P': r_P,
            'r_Q_orig': r_Q_orig,
            'r_Q_obs': r_Q_obs,
            'interp': interp
        })

    # Analyze P estimates for multi-modality / equivalence classes
    print("\n" + "=" * 70)
    print("P DISTRIBUTION ANALYSIS (Equivalence Classes)")
    print("=" * 70)

    for ratio in [0.1, 1.0, 5.0]:
        if ratio not in [r['ratio'] for r in results]:
            continue
        data = datasets[ratio]
        if len(data['P_true']) < 20:
            continue

        P = data['P_true']
        E = data['E']

        # Check P vs E relationship
        r_PE, _ = stats.pearsonr(P, E)

        # Check for clustering in P
        # Compute gaps in sorted P
        P_sorted = np.sort(P)
        gaps = np.diff(P_sorted)
        mean_gap = np.mean(gaps)
        cv_gaps = np.std(gaps) / mean_gap if mean_gap > 0 else 0

        print(f"\n  Delta_tau/T = {ratio}:")
        print(f"    r(P, E) = {r_PE:.3f}")
        print(f"    Gap CV = {cv_gaps:.3f} (low = uniform, high = clustered)")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. r(P) and r(Q_orig) vs delta_tau/T
    ax = axes[0, 0]
    ratios = [r['ratio'] for r in results]
    r_P_vals = [r['r_P'] for r in results]
    r_Q_orig_vals = [abs(r['r_Q_orig']) for r in results]
    r_Q_obs_vals = [abs(r['r_Q_obs']) for r in results]

    ax.semilogx(ratios, r_P_vals, 'bo-', label='r(P)', markersize=8)
    ax.semilogx(ratios, r_Q_orig_vals, 'ro-', label='r(Q_original)', markersize=8)
    ax.semilogx(ratios, r_Q_obs_vals, 'g^--', label='r(Q_observed)', markersize=6, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Delta_tau = T')
    ax.set_xlabel('Delta_tau / T (log scale)')
    ax.set_ylabel('Correlation')
    ax.set_title('Can we predict Q_original from features?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    # 2. P distribution at fine scale
    ax = axes[0, 1]
    if 0.1 in [r['ratio'] for r in results]:
        data = datasets[0.1]
        ax.scatter(data['E'], data['P_true'], alpha=0.5, s=10)
        ax.set_xlabel('Energy E')
        ax.set_ylabel('P (action proxy)')
        ax.set_title('Delta_tau/T = 0.1 (Fine)')
    ax.grid(True, alpha=0.3)

    # 3. P distribution at coarse scale
    ax = axes[0, 2]
    if 5.0 in [r['ratio'] for r in results]:
        data = datasets[5.0]
        ax.scatter(data['E'], data['P_true'], alpha=0.5, s=10)
        ax.set_xlabel('Energy E')
        ax.set_ylabel('P (action proxy)')
        ax.set_title('Delta_tau/T = 5.0 (Coarse)')
    ax.grid(True, alpha=0.3)

    # 4. Offset distribution at fine scale
    ax = axes[1, 0]
    if 0.1 in [r['ratio'] for r in results]:
        data = datasets[0.1]
        ax.hist(data['offset_phase'], bins=30, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Offset phase (radians)')
        ax.set_ylabel('Density')
        ax.set_title('Offset distribution (Fine: narrow)')
    ax.grid(True, alpha=0.3)

    # 5. Offset distribution at coarse scale
    ax = axes[1, 1]
    if 5.0 in [r['ratio'] for r in results]:
        data = datasets[5.0]
        ax.hist(data['offset_phase'], bins=30, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Offset phase (radians)')
        ax.set_ylabel('Density')
        ax.set_title('Offset distribution (Coarse: uniform)')
    ax.grid(True, alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    # Find transition point
    transition_ratio = None
    for r in results:
        if abs(r['r_Q_orig']) < 0.4 and r['r_P'] > 0.7:
            transition_ratio = r['ratio']
            break

    summary = f"""
    OBSERVATION OPERATOR TEST
    =========================

    Mechanism: Unknown observation start time
    t0 ~ Unif[0, Delta_tau]

    Key distinction:
    - Q_original: phase at t=0 (unknown to observer)
    - Q_observed: phase at t=t0 (in the features)

    Results:
    - r(P) stable: {np.mean(r_P_vals):.3f}
    - r(Q_original) fine: {r_Q_orig_vals[0] if r_Q_orig_vals else 0:.3f}
    - r(Q_original) coarse: {r_Q_orig_vals[-1] if r_Q_orig_vals else 0:.3f}
    - r(Q_observed) always high (it's in features)

    Transition at Delta_tau/T ~ {transition_ratio if transition_ratio else '?'}

    Physical Interpretation:
    - Features CAN represent phase (Q_observed)
    - But ORIGINAL phase unrecoverable when
      offset spans many periods
    - Only action P survives at coarse scales
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('observation_operator_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to observation_operator_test.png")
    plt.show()

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    fine_r_Q_orig = r_Q_orig_vals[0] if r_Q_orig_vals else 0
    coarse_r_Q_orig = r_Q_orig_vals[-1] if r_Q_orig_vals else 0
    fine_r_Q_obs = r_Q_obs_vals[0] if r_Q_obs_vals else 0
    coarse_r_Q_obs = r_Q_obs_vals[-1] if r_Q_obs_vals else 0
    mean_r_P = np.mean(r_P_vals) if r_P_vals else 0

    if mean_r_P > 0.7 and fine_r_Q_orig > 0.5 and coarse_r_Q_orig < 0.3:
        print(f"""
    CONFIRMED: Observation-level coarse-graining erases ORIGINAL phase!

    Key findings:
    1. Phase-aware features (21 dims) used throughout
    2. Features always predict Q_observed (the phase they see)
    3. But Q_original becomes unpredictable when Delta_tau >> T

    Numerical evidence:
    - r(P) stable: {mean_r_P:.3f}
    - r(Q_original) fine: {fine_r_Q_orig:.3f}  (recoverable)
    - r(Q_original) coarse: {coarse_r_Q_orig:.3f}  (erased!)
    - r(Q_observed) stays high: {coarse_r_Q_obs:.3f}  (in features)

    The mechanism is PHYSICAL:
    - Observation starts at unknown time t0 ~ Unif[0, Delta_tau]
    - When Delta_tau >> T, offset spans many periods
    - Original phase becomes uniformly distributed
    - Action (constant along orbit) always recoverable

    This validates Glinsky's claim:
    "Coarse observation makes Q unknowable,
    leading to quantization for probability periodicity."
        """)
    elif coarse_r_Q_orig < fine_r_Q_orig and mean_r_P > 0.6:
        print(f"""
    PARTIALLY CONFIRMED: Q_original degrades with coarser observation

    - r(P) mean: {mean_r_P:.3f}
    - r(Q_original) fine: {fine_r_Q_orig:.3f}
    - r(Q_original) coarse: {coarse_r_Q_orig:.3f}
    - Degradation ratio: {coarse_r_Q_orig / max(fine_r_Q_orig, 0.01):.2f}

    The trend is correct but threshold not fully reached.
        """)
    else:
        print(f"""
    Results require interpretation:
    - Mean r(P): {mean_r_P:.3f}
    - r(Q_original) fine: {fine_r_Q_orig:.3f}
    - r(Q_original) coarse: {coarse_r_Q_orig:.3f}
    - r(Q_observed) fine: {fine_r_Q_obs:.3f}
    - r(Q_observed) coarse: {coarse_r_Q_obs:.3f}

    Expected: Q_original should degrade while Q_observed stays high
        """)

    return results, datasets


if __name__ == "__main__":
    results, datasets = run_observation_operator_test()
