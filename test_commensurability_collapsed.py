"""
Commensurability Test with COLLAPSED Information

Key insight from ChatGPT: The original test failed because 512-sample windows
contain too much fine structure (amplitude, frequency, shape). The classifier
can easily distinguish states even with the same winding number.

FIX: Collapse observation to O(1) features that only preserve what survives
at the coarse observation scale.

Two observation modes:
1. Strobe-Only: Sample at intervals Δτ (sparse samples, no time series)
2. Cycle-Average: Average over complete cycles (removes within-cycle phase)

Expected result: Same winding number → ~50% accuracy (indistinguishable)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian


def pendulum_omega(E):
    """Theoretical frequency ω(E) = π / (2K(k)) where k² = (1+E)/2"""
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 <= 0 or k2 >= 1:
        return 1.0
    k = np.sqrt(k2)
    return np.pi / (2 * ellipk(k**2))


def winding_number(E, delta_tau):
    """n(E) = floor(Δτ · ω(E) / 2π) = floor(Δτ / T)"""
    omega = pendulum_omega(E)
    if np.isnan(omega) or omega <= 0:
        return 0
    return int(np.floor(delta_tau * omega / (2 * np.pi)))


# ============================================================
# COLLAPSED OBSERVATION MODES
# ============================================================

def observe_strobe_only(z, T_period, delta_tau, n_strobes=5):
    """
    Observe ONLY at intervals Δτ (sparse samples, not full time series).

    This removes:
    - Frequency information (can't estimate ω from sparse samples)
    - Amplitude shape details (just isolated points)
    - Within-period phase information

    Returns: O(1) features (4-8 numbers), NOT 512 samples
    """
    dt = 0.01
    sample_interval = max(1, int(delta_tau / dt))

    # Sample at Δτ intervals - get only a few samples
    z_strobe = z[::sample_interval][:n_strobes]

    if len(z_strobe) < 2:
        return None

    # Features: just summary statistics of sparse samples
    mags = np.abs(z_strobe)
    phases = np.angle(z_strobe)

    features = [
        np.mean(mags),           # Mean magnitude (action proxy)
        np.std(mags),            # Magnitude variation
        np.mean(np.cos(phases)), # Phase circular mean (x)
        np.mean(np.sin(phases)), # Phase circular mean (y)
    ]

    return np.array(features)


def observe_cycle_average(z, T_period, n_cycles=10):
    """
    Average over n_cycles periods. Returns ONE complex number per cycle.

    This removes:
    - Phase information within each cycle
    - Frequency information (just slow drift between cycles)
    - Only action-like information survives

    Returns: O(1) features from cycle averages
    """
    dt = 0.01
    samples_per_cycle = max(1, int(T_period / dt))

    # Average each cycle to single complex value
    cycle_averages = []
    for i in range(n_cycles):
        start = i * samples_per_cycle
        end = start + samples_per_cycle
        if end <= len(z):
            cycle_averages.append(np.mean(z[start:end]))

    if len(cycle_averages) < 2:
        return None

    z_avg = np.array(cycle_averages)

    # Features from cycle averages only
    features = [
        np.mean(np.abs(z_avg)),           # Mean magnitude
        np.std(np.abs(z_avg)),            # Magnitude std across cycles
        np.mean(np.real(z_avg)),          # Mean real part
        np.mean(np.imag(z_avg)),          # Mean imag part
    ]

    return np.array(features)


def observe_winding_only(z, T_period, delta_tau):
    """
    Extract ONLY what depends on winding number n.

    Features that should be ~same for same n:
    - Number of zero crossings in window (∝ n)
    - Mean amplitude (roughly same for similar energies in class)

    Features that differ:
    - Exact frequency (but we don't measure it directly)
    - Exact amplitude (but coarsely similar)
    """
    dt = 0.01
    window_samples = int(delta_tau / dt)

    if window_samples > len(z):
        window_samples = len(z)

    z_window = z[:window_samples]

    # Count zero crossings (proxy for winding number)
    real_part = np.real(z_window)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(real_part))) > 0)

    # Mean amplitude (action proxy, but coarse)
    mean_amp = np.mean(np.abs(z_window))

    features = [
        zero_crossings / window_samples,  # Normalized crossing rate ∝ ω
        mean_amp,                          # Mean amplitude
    ]

    return np.array(features)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_trajectory(E, T_sim=200, dt=0.01):
    """Generate trajectory at energy E."""
    pendulum = PendulumOscillator()

    omega = pendulum_omega(E)
    if np.isnan(omega):
        return None, None

    T_period = 2 * np.pi / omega

    # Initial condition at turning point
    q_max = np.arccos(-E) if E > -1 else np.pi * 0.95
    q0 = q_max
    p0 = 0.0

    try:
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=dt)
    except:
        return None, None

    return z, T_period


def generate_collapsed_features(E, delta_tau, mode='strobe', n_samples=50, seed=None):
    """
    Generate collapsed features for energy E with coarse observation.

    mode: 'strobe', 'cycle', or 'winding'
    """
    if seed is not None:
        np.random.seed(seed)

    z, T_period = generate_trajectory(E)
    if z is None:
        return None

    dt = 0.01
    max_offset = max(1, int(delta_tau / dt))

    features_list = []

    for _ in range(n_samples):
        # Random observation start (coarse timing uncertainty)
        offset = np.random.randint(0, max_offset)
        z_shifted = np.roll(z, offset)

        if mode == 'strobe':
            feat = observe_strobe_only(z_shifted, T_period, delta_tau)
        elif mode == 'cycle':
            feat = observe_cycle_average(z_shifted, T_period)
        elif mode == 'winding':
            feat = observe_winding_only(z_shifted, T_period, delta_tau)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if feat is not None:
            features_list.append(feat)

    if not features_list:
        return None

    return np.array(features_list)


# ============================================================
# CLASSIFICATION
# ============================================================

def train_classifier(X, y, test_fraction=0.3):
    """Train simple linear classifier, return test accuracy."""
    n = len(y)
    if n < 20:
        return np.nan

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

    # Logistic regression via least squares
    X_b = np.column_stack([X_train_norm, np.ones(len(X_train_norm))])
    w = np.linalg.lstsq(X_b, y_train, rcond=None)[0]

    # Predict
    X_test_b = np.column_stack([X_test_norm, np.ones(len(X_test_norm))])
    y_pred = (X_test_b @ w > 0.5).astype(int)

    return np.mean(y_pred == y_test)


def test_pair(E1, E2, delta_tau, mode='strobe', n_samples=50, seed=42):
    """Test distinguishability of two energies with collapsed features."""
    np.random.seed(seed)

    feat1 = generate_collapsed_features(E1, delta_tau, mode=mode, n_samples=n_samples, seed=seed)
    feat2 = generate_collapsed_features(E2, delta_tau, mode=mode, n_samples=n_samples, seed=seed+1000)

    if feat1 is None or feat2 is None:
        return None

    if len(feat1) < 10 or len(feat2) < 10:
        return None

    X = np.vstack([feat1, feat2])
    y = np.array([0] * len(feat1) + [1] * len(feat2))

    return train_classifier(X, y)


# ============================================================
# MAIN TEST
# ============================================================

def find_energy_pairs_same_n(delta_tau, E_range=(-0.85, 0.75)):
    """Find pairs of energies with same winding number but different E."""
    E_values = np.linspace(E_range[0], E_range[1], 200)

    n_to_E = {}
    for E in E_values:
        n = winding_number(E, delta_tau)
        if n not in n_to_E:
            n_to_E[n] = []
        n_to_E[n].append(E)

    return n_to_E


def run_collapsed_commensurability_test():
    """
    Main test with collapsed observation modes.
    """
    print("=" * 70)
    print("COMMENSURABILITY TEST (COLLAPSED FEATURES)")
    print("=" * 70)
    print("\nKey fix: Use O(1) features instead of 512-sample windows")
    print("This removes fine structure that made all pairs distinguishable")

    # Parameters
    delta_tau = 50.0  # Observation timescale

    # Find winding number structure
    print(f"\n[1] Finding winding number classes (Δτ = {delta_tau})...")
    n_to_E = find_energy_pairs_same_n(delta_tau)

    # Find classes with good spread
    testable_classes = [(n, Es) for n, Es in n_to_E.items()
                        if len(Es) >= 5 and max(Es) - min(Es) > 0.1]

    print(f"    Found {len(testable_classes)} testable classes")
    for n, Es in testable_classes[:5]:
        print(f"    n={n}: E ∈ [{min(Es):.3f}, {max(Es):.3f}] ({len(Es)} energies)")

    # Test each observation mode
    modes = ['strobe', 'cycle', 'winding']

    all_results = {}

    for mode in modes:
        print(f"\n[2] Testing mode: {mode.upper()}")
        print("-" * 60)

        within_results = []
        across_results = []

        # Test WITHIN same n
        print("\n  [a] WITHIN same winding number:")
        for n, Es in testable_classes[:3]:
            E1, E2 = Es[0], Es[-1]
            if abs(E1 - E2) < 0.05:
                continue

            acc = test_pair(E1, E2, delta_tau, mode=mode)
            if acc is not None:
                within_results.append({'n': n, 'E1': E1, 'E2': E2, 'accuracy': acc})
                marker = "***" if acc < 0.6 else ""
                print(f"      n={n}: E₁={E1:.3f}, E₂={E2:.3f} → {acc:.1%} {marker}")

        # Test ACROSS different n
        print("\n  [b] ACROSS different winding numbers:")
        for i in range(min(3, len(testable_classes) - 1)):
            n1, Es1 = testable_classes[i]
            n2, Es2 = testable_classes[i + 1]
            E1 = np.median(Es1)
            E2 = np.median(Es2)

            acc = test_pair(E1, E2, delta_tau, mode=mode)
            if acc is not None:
                across_results.append({'n1': n1, 'n2': n2, 'E1': E1, 'E2': E2, 'accuracy': acc})
                print(f"      n₁={n1} vs n₂={n2}: E₁={E1:.3f}, E₂={E2:.3f} → {acc:.1%}")

        mean_within = np.mean([r['accuracy'] for r in within_results]) if within_results else 0.5
        mean_across = np.mean([r['accuracy'] for r in across_results]) if across_results else 0.5

        all_results[mode] = {
            'within': within_results,
            'across': across_results,
            'mean_within': mean_within,
            'mean_across': mean_across,
            'separation': mean_across - mean_within
        }

        print(f"\n  Mode {mode}: within={mean_within:.1%}, across={mean_across:.1%}, sep={mean_across - mean_within:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Mode':<12} {'Within (same n)':<18} {'Across (diff n)':<18} {'Separation':<12} {'Verdict':<15}")
    print("-" * 75)

    for mode, res in all_results.items():
        verdict = "QUANTIZED!" if res['mean_within'] < 0.6 and res['separation'] > 0.15 else "Not yet"
        print(f"{mode:<12} {res['mean_within']:<18.1%} {res['mean_across']:<18.1%} {res['separation']:<12.1%} {verdict:<15}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Comparison of modes
    ax = axes[0, 0]
    x = np.arange(len(modes))
    width = 0.35
    within_vals = [all_results[m]['mean_within'] for m in modes]
    across_vals = [all_results[m]['mean_across'] for m in modes]

    ax.bar(x - width/2, within_vals, width, label='Within (same n)', color='green', alpha=0.7)
    ax.bar(x + width/2, across_vals, width, label='Across (diff n)', color='red', alpha=0.7)
    ax.axhline(0.5, color='gray', linestyle='--', label='Chance (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Collapsed Features: Within vs Across')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 2. Separation by mode
    ax = axes[0, 1]
    separations = [all_results[m]['separation'] for m in modes]
    colors = ['green' if s > 0.15 else 'gray' for s in separations]
    ax.bar(modes, separations, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0.15, color='red', linestyle='--', label='Target separation')
    ax.set_ylabel('Separation (Across - Within)')
    ax.set_title('Separation by Observation Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. E vs n mapping
    ax = axes[1, 0]
    E_plot = np.linspace(-0.85, 0.75, 200)
    n_plot = [winding_number(E, delta_tau) for E in E_plot]
    ax.plot(E_plot, n_plot, 'b-', linewidth=2)
    ax.set_xlabel('Energy E')
    ax.set_ylabel('Winding number n')
    ax.set_title(f'n(E) = floor(Δτ·ω(E)/2π), Δτ={delta_tau}')
    ax.grid(True, alpha=0.3)

    # Mark tested pairs
    for mode, res in all_results.items():
        for r in res['within']:
            if mode == 'strobe':  # Only plot one mode
                ax.scatter([r['E1'], r['E2']], [r['n'], r['n']], c='green', s=50, marker='o')

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis('off')

    best_mode = max(modes, key=lambda m: all_results[m]['separation'])
    best_sep = all_results[best_mode]['separation']
    quantized_modes = [m for m in modes if all_results[m]['mean_within'] < 0.6 and all_results[m]['separation'] > 0.15]

    summary = f"""
    COLLAPSED COMMENSURABILITY TEST
    ================================

    Observation modes tested:
    - strobe: Sample only at Δτ intervals (4 features)
    - cycle: Average each period (4 features)
    - winding: Zero-crossings + mean amp (2 features)

    Key question: Do same-n states become indistinguishable?

    Results:
    - Best separation: {best_mode} ({best_sep:.1%})
    - Within same n: {all_results[best_mode]['mean_within']:.1%}
    - Across diff n: {all_results[best_mode]['mean_across']:.1%}

    Verdict: {"QUANTIZATION CONFIRMED for: " + ", ".join(quantized_modes) if quantized_modes else "Fine structure still visible"}

    Note: If still 100%/100%, the amplitude differences
    between E values (even with same n) remain detectable.
    May need to normalize by amplitude to isolate winding.
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('commensurability_collapsed_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to commensurability_collapsed_test.png")
    plt.show()

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if quantized_modes:
        print(f"""
    EQUIVALENCE CLASSES FOUND with {', '.join(quantized_modes)} observation!

    States with same winding number n become indistinguishable when
    observed with collapsed features that don't preserve fine structure.

    This confirms Glinsky's quantization:
    The observable is n (discrete), not E (continuous).
        """)
    else:
        print(f"""
    Fine structure still visible in all modes.

    Best attempt: {best_mode} with separation {best_sep:.1%}

    Interpretation:
    Even with collapsed features, the amplitude differences between
    E values (even with same n) allow classification. The action P
    remains observable and continuous.

    This suggests quantization is in the REPRESENTATION (when we
    choose to ignore amplitude), not in the dynamics themselves.
        """)

    return all_results


if __name__ == "__main__":
    results = run_collapsed_commensurability_test()
