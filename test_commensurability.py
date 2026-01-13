"""
Commensurability / Winding Number Equivalence Test

Key insight from ChatGPT: The "quantization" isn't about P values being discrete.
It's about EQUIVALENCE CLASSES defined by winding number.

States with same winding number n = floor(Δτ · ω(E) / 2π) should be
INDISTINGUISHABLE under coarse observation, even if their E values differ.

Protocol:
1. Define winding number n(E) = floor(Δτ · ω(E) / 2π)
2. Find energy pairs (E₁, E₂) with n(E₁) = n(E₂) but E₁ ≠ E₂
3. Apply coarse observation
4. Train classifier to distinguish E₁ from E₂
5. If accuracy ≈ 50%: They're in same equivalence class (quantization!)
   If accuracy >> 50%: Fine structure still visible (not quantized)

The observable isn't E - it's n (the winding number).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian
from hst import extract_features


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
    """
    Compute winding number: n(E) = floor(Δτ · ω(E) / 2π)

    This counts how many complete oscillations fit in the observation window.
    """
    omega = pendulum_omega(E)
    if np.isnan(omega) or omega <= 0:
        return 0
    return int(np.floor(delta_tau * omega / (2 * np.pi)))


def find_energy_pairs_same_n(E_range=(-0.9, 0.8), delta_tau=50.0, n_points=100):
    """
    Find pairs of energies with the same winding number but different E.

    Returns dict mapping n -> list of E values in that class.
    """
    E_values = np.linspace(E_range[0], E_range[1], n_points)

    n_to_E = {}
    for E in E_values:
        n = winding_number(E, delta_tau)
        if n not in n_to_E:
            n_to_E[n] = []
        n_to_E[n].append(E)

    return n_to_E


def generate_classification_dataset(E1, E2, n_samples=50, delta_tau_ratio=5.0, seed=42):
    """
    Generate dataset for binary classification between E1 and E2.

    Returns features and labels for classification.
    """
    np.random.seed(seed)
    pendulum = PendulumOscillator()

    features = []
    labels = []
    actual_E = []

    for label, E_target in enumerate([E1, E2]):
        omega = pendulum_omega(E_target)
        if np.isnan(omega):
            continue
        T_period = 2 * np.pi / omega
        delta_tau = delta_tau_ratio * T_period
        dt = 0.01

        # Initial condition
        q_max = np.arccos(-E_target) if E_target > -1 else np.pi * 0.95
        q0 = q_max
        p0 = 0.0

        # Generate long trajectory
        T_sim = max(30 * T_period, 150)
        try:
            t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=dt)
        except:
            continue

        if len(z) < 1024:
            continue

        # Multiple samples with random observation offsets (coarse observation)
        max_offset_samples = max(1, int(delta_tau / dt))
        window_size = 512

        for _ in range(n_samples):
            # Random observation start (coarse-graining)
            offset_samples = np.random.randint(0, max_offset_samples)
            start_idx = offset_samples % max(1, len(z) - window_size - 100)

            if start_idx + window_size >= len(z):
                start_idx = max(0, len(z) - window_size - 1)

            z_obs = z[start_idx:start_idx + window_size]

            if len(z_obs) < window_size:
                continue

            try:
                feat = extract_features(z_obs, J=3, wavelet_name='db8')
                features.append(feat)
                labels.append(label)
                actual_E.append(E_actual)
            except:
                pass

    return np.array(features), np.array(labels), np.array(actual_E)


def train_classifier(X, y, test_fraction=0.3):
    """
    Train simple linear classifier, return test accuracy.
    """
    n = len(y)
    if n < 20:
        return np.nan, None, None

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
    y_pred_prob = X_test_b @ w
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Accuracy
    accuracy = np.mean(y_pred == y_test)

    return accuracy, y_pred, y_test


def run_commensurability_test():
    """
    Main test: Are states with same winding number indistinguishable?
    """
    print("=" * 70)
    print("COMMENSURABILITY / WINDING NUMBER EQUIVALENCE TEST")
    print("=" * 70)
    print("\nKey question: Do states with same n become indistinguishable?")
    print("n(E) = floor(Δτ · ω(E) / 2π) = winding number")

    # Parameters
    delta_tau_fixed = 50.0  # Fixed observation window (in time units)

    # Find winding number structure
    print("\n[1] Mapping energy to winding number...")
    print(f"    Δτ = {delta_tau_fixed}")

    n_to_E = find_energy_pairs_same_n(E_range=(-0.85, 0.75), delta_tau=delta_tau_fixed, n_points=200)

    print(f"\n    Winding number classes found:")
    for n in sorted(n_to_E.keys()):
        E_list = n_to_E[n]
        if len(E_list) >= 2:
            print(f"    n={n}: {len(E_list)} energies in range [{min(E_list):.3f}, {max(E_list):.3f}]")

    # Select pairs for testing
    print("\n[2] Testing distinguishability within and across equivalence classes...")

    results = []

    # Find classes with enough spread
    testable_classes = [(n, E_list) for n, E_list in n_to_E.items()
                        if len(E_list) >= 5 and max(E_list) - min(E_list) > 0.1]

    if len(testable_classes) < 2:
        print("    Not enough energy spread within classes. Adjusting Δτ...")
        delta_tau_fixed = 100.0
        n_to_E = find_energy_pairs_same_n(E_range=(-0.85, 0.75), delta_tau=delta_tau_fixed, n_points=200)
        testable_classes = [(n, E_list) for n, E_list in n_to_E.items()
                            if len(E_list) >= 5 and max(E_list) - min(E_list) > 0.1]

    print(f"\n    Testable classes: {len(testable_classes)}")

    # Test 1: Within same equivalence class (same n)
    print("\n    [2a] WITHIN same equivalence class (same n):")
    print("         Expected: Accuracy ≈ 50% (indistinguishable)")
    print("-" * 60)

    within_class_accuracies = []

    for n, E_list in testable_classes[:3]:  # Test up to 3 classes
        # Pick two energies from same class
        E_list_sorted = sorted(E_list)
        E1 = E_list_sorted[0]
        E2 = E_list_sorted[-1]
        E_diff = abs(E2 - E1)

        if E_diff < 0.05:
            continue

        # Determine Δτ/T for these energies
        omega1 = pendulum_omega(E1)
        omega2 = pendulum_omega(E2)
        T1 = 2 * np.pi / omega1
        T2 = 2 * np.pi / omega2
        ratio1 = delta_tau_fixed / T1
        ratio2 = delta_tau_fixed / T2

        X, y, actual_E = generate_classification_dataset(E1, E2, n_samples=40, delta_tau_ratio=ratio1, seed=42)

        if len(y) < 30:
            continue

        accuracy, y_pred, y_test = train_classifier(X, y)

        within_class_accuracies.append(accuracy)

        print(f"    n={n}: E₁={E1:.3f}, E₂={E2:.3f} (ΔE={E_diff:.3f})")
        print(f"          Δτ/T = [{ratio1:.1f}, {ratio2:.1f}]")
        print(f"          Accuracy = {accuracy:.1%}")

        results.append({
            'type': 'within',
            'n': n,
            'E1': E1,
            'E2': E2,
            'E_diff': E_diff,
            'accuracy': accuracy
        })

    # Test 2: Across different equivalence classes (different n)
    print("\n    [2b] ACROSS different equivalence classes (different n):")
    print("         Expected: Accuracy >> 50% (distinguishable)")
    print("-" * 60)

    across_class_accuracies = []

    if len(testable_classes) >= 2:
        for i in range(min(3, len(testable_classes) - 1)):
            n1, E_list1 = testable_classes[i]
            n2, E_list2 = testable_classes[i + 1]

            E1 = np.median(E_list1)
            E2 = np.median(E_list2)
            E_diff = abs(E2 - E1)

            omega1 = pendulum_omega(E1)
            T1 = 2 * np.pi / omega1
            ratio = delta_tau_fixed / T1

            X, y, actual_E = generate_classification_dataset(E1, E2, n_samples=40, delta_tau_ratio=ratio, seed=42)

            if len(y) < 30:
                continue

            accuracy, y_pred, y_test = train_classifier(X, y)

            across_class_accuracies.append(accuracy)

            print(f"    n₁={n1} vs n₂={n2}: E₁={E1:.3f}, E₂={E2:.3f} (ΔE={E_diff:.3f})")
            print(f"          Accuracy = {accuracy:.1%}")

            results.append({
                'type': 'across',
                'n1': n1,
                'n2': n2,
                'E1': E1,
                'E2': E2,
                'E_diff': E_diff,
                'accuracy': accuracy
            })

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_within = np.mean(within_class_accuracies) if within_class_accuracies else 0.5
    mean_across = np.mean(across_class_accuracies) if across_class_accuracies else 0.5

    print(f"\n  Mean accuracy WITHIN same n:   {mean_within:.1%}")
    print(f"  Mean accuracy ACROSS different n: {mean_across:.1%}")
    print(f"  Separation: {mean_across - mean_within:.1%}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Winding number vs Energy
    ax = axes[0, 0]
    E_plot = np.linspace(-0.85, 0.75, 200)
    n_plot = [winding_number(E, delta_tau_fixed) for E in E_plot]
    ax.plot(E_plot, n_plot, 'b-', linewidth=2)
    ax.set_xlabel('Energy E')
    ax.set_ylabel('Winding number n')
    ax.set_title(f'n(E) = floor(Δτ·ω(E)/2π), Δτ={delta_tau_fixed}')
    ax.grid(True, alpha=0.3)

    # Mark tested pairs
    for r in results:
        if r['type'] == 'within':
            ax.axhline(r['n'], color='green', linestyle='--', alpha=0.3)
            ax.scatter([r['E1'], r['E2']], [r['n'], r['n']], c='green', s=50, zorder=5)

    # 2. Accuracy comparison
    ax = axes[0, 1]
    categories = ['Within\n(same n)', 'Across\n(diff n)']
    means = [mean_within, mean_across]
    colors = ['green', 'red']
    bars = ax.bar(categories, means, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0.5, color='gray', linestyle='--', label='Random (50%)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Can We Distinguish E₁ from E₂?')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.1%}', ha='center', fontsize=12)

    # 3. ω(E) showing why equivalence classes form
    ax = axes[1, 0]
    omega_plot = [pendulum_omega(E) for E in E_plot]
    ax.plot(E_plot, omega_plot, 'r-', linewidth=2)
    ax.set_xlabel('Energy E')
    ax.set_ylabel('Frequency ω(E)')
    ax.set_title('Pendulum frequency (approaches 0 at separatrix)')
    ax.grid(True, alpha=0.3)

    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')

    quantized = mean_within < 0.6 and mean_across > 0.7

    summary = f"""
    COMMENSURABILITY TEST
    =====================

    Observation window: Δτ = {delta_tau_fixed}

    Winding number: n(E) = floor(Δτ·ω(E)/2π)
    This counts oscillations in window.

    Results:
    - Within same n: {mean_within:.1%} accuracy
      (Expected ~50% if indistinguishable)
    - Across diff n: {mean_across:.1%} accuracy
      (Expected high if distinguishable)

    Separation: {mean_across - mean_within:.1%}

    VERDICT: {'QUANTIZATION CONFIRMED' if quantized else 'NOT QUANTIZED'}

    Interpretation:
    {'States with same winding number form' if quantized else 'Fine structure visible even'}
    {'equivalence classes - they become' if quantized else 'within same winding class.'}
    {'indistinguishable under coarse' if quantized else 'May need larger Δτ or'}
    {'observation. The observable is n.' if quantized else 'different test parameters.'}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('commensurability_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to commensurability_test.png")
    plt.show()

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if quantized:
        print(f"""
    EQUIVALENCE CLASSES CONFIRMED!

    - States with same winding number n are indistinguishable
      (accuracy ≈ {mean_within:.1%}, near random)
    - States with different n are distinguishable
      (accuracy ≈ {mean_across:.1%}, well above chance)

    The "quantization" is about EQUIVALENCE CLASSES:
    - The observable isn't E (continuous)
    - The observable is n = floor(Δτ·ω(E)/2π) (discrete)
    - Different E values mapping to same n are equivalent

    This is exactly what Glinsky predicts:
    Under coarse observation, fine structure (within class) is lost,
    only the discrete winding number (class label) remains.
        """)
    else:
        print(f"""
    EQUIVALENCE CLASSES NOT CLEARLY SEPARATED

    - Within same n: {mean_within:.1%} (expected ~50%)
    - Across diff n: {mean_across:.1%} (expected high)
    - Separation: {mean_across - mean_within:.1%}

    Possible issues:
    - Δτ not large enough for full coarse-graining
    - Energy differences within class too large
    - Features still capture fine structure

    The test needs parameter tuning or the quantization
    signature is weaker than expected.
        """)

    return {
        'results': results,
        'mean_within': mean_within,
        'mean_across': mean_across,
        'delta_tau': delta_tau_fixed,
        'n_to_E': n_to_E
    }


if __name__ == "__main__":
    results = run_commensurability_test()
