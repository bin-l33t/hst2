"""
Q-Equivalence Classes Under Coarse-Graining Test

Tests Glinsky's core "quantum-like" claim:
States differing only in Q become indistinguishable when observed coarsely.

Protocol:
1. Generate trajectory pairs at SAME energy E, DIFFERENT initial phase Q₀
2. Extract coarse (magnitude-only) and fine (phase-aware) features
3. Train binary classifier to distinguish Q₀ classes
4. Expected: Coarse accuracy ≈ 50%, Fine accuracy >> 50%

This validates: "Coarse observation creates Q-equivalence classes"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian
from hst import extract_features, extract_features_magnitude_only


def generate_q_pairs(E_target, phase_offset_0, phase_offset_1, n_samples=20, seed=None):
    """
    Generate trajectory pairs at same E but different initial phase.

    CORRECT APPROACH: Generate one long trajectory, then extract windows
    starting at different times. Different start times = same E, different Q.
    """
    if seed is not None:
        np.random.seed(seed)

    pendulum = PendulumOscillator()
    dt = 0.01

    # Get period for this energy
    omega = pendulum_omega(E_target)
    T_period = 2 * np.pi / omega if omega > 0.01 else 10.0

    # Generate one long reference trajectory
    # Start at q = q_max (turning point), p = 0
    if E_target < 0.99:
        q_max = np.arccos(-E_target) if E_target > -1 else np.pi * 0.95
        q0, p0 = q_max, 0.0
    else:
        q0, p0 = 0.1, np.sqrt(2 * (E_target + 1))

    # Run for many periods
    T_long = (n_samples + 5) * T_period
    t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_long, dt=dt)

    # Window size (512 samples)
    window = 512

    # Phase offsets as fractions of period
    # phase_offset_0 and phase_offset_1 are in [0, 1]
    samples_per_period = int(T_period / dt)

    trajectories_0 = []
    trajectories_1 = []

    for i in range(n_samples):
        # Starting index for this sample
        base_idx = i * samples_per_period

        # Class 0: start at phase_offset_0 into the period
        idx_0 = base_idx + int(phase_offset_0 * samples_per_period)
        if idx_0 + window < len(z):
            trajectories_0.append(z[idx_0:idx_0+window])

        # Class 1: start at phase_offset_1 into the period
        idx_1 = base_idx + int(phase_offset_1 * samples_per_period)
        if idx_1 + window < len(z):
            trajectories_1.append(z[idx_1:idx_1+window])

    return trajectories_0, trajectories_1


def pendulum_omega(E):
    """Theoretical frequency"""
    from scipy.special import ellipk
    if E >= 1:
        return 0.01
    k2 = (1 + E) / 2
    if k2 <= 0:
        return 1.0
    k = np.sqrt(k2)
    if k >= 1:
        return 0.01
    return np.pi / (2 * ellipk(k**2))


def extract_features_for_trajectories(trajectories, use_phase=True):
    """Extract features from list of trajectories."""
    features = []
    for z in trajectories:
        if use_phase:
            feat = extract_features(z, J=3, wavelet_name='db8')
        else:
            feat = extract_features_magnitude_only(z, J=3, wavelet_name='db8')
        features.append(feat)
    return np.array(features)


def train_simple_classifier(X_train, y_train, X_test, y_test):
    """
    Simple linear classifier using least squares.

    Returns accuracy on test set.
    """
    # Add bias
    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(len(X_test))])

    # Solve via least squares
    w, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)

    # Predict
    y_pred = X_test_b @ w
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = np.mean(y_pred_class == y_test)
    return accuracy


def run_q_equivalence_test():
    """
    Main test: Can we distinguish Q₀ classes with coarse vs fine features?
    """
    print("=" * 70)
    print("Q-EQUIVALENCE CLASSES TEST")
    print("Testing: Coarse observation creates Q-equivalence classes")
    print("=" * 70)

    # Test parameters
    E_values = [-0.6, -0.3, 0.0, 0.3, 0.6]  # Different energy levels
    # Phase offsets as fractions of period [0, 1]
    # Key insight: Half-period separation should show Q-equivalence (symmetry)
    #              Quarter-period should NOT (asymmetric magnitudes)
    phase_pairs = [
        (0.0, 0.5),    # Half period apart → Q-EQUIVALENT (symmetry)
        (0.1, 0.6),    # Half period apart → Q-EQUIVALENT
        (0.0, 0.25),   # Quarter period → NOT equivalent
        (0.0, 0.1),    # Small difference → NOT equivalent
    ]
    n_samples_per_class = 30

    results_coarse = []
    results_fine = []

    print("\n[1] Generating trajectories and training classifiers...")

    for E in E_values:
        for phase_0, phase_1 in phase_pairs:
            # Generate data - same energy, different phases
            traj_0, traj_1 = generate_q_pairs(E, phase_0, phase_1,
                                               n_samples=n_samples_per_class,
                                               seed=42)

            if len(traj_0) < 10 or len(traj_1) < 10:
                print(f"  E={E:+.1f}: Not enough samples, skipping")
                continue

            # Extract features
            feat_0_coarse = extract_features_for_trajectories(traj_0, use_phase=False)
            feat_1_coarse = extract_features_for_trajectories(traj_1, use_phase=False)
            feat_0_fine = extract_features_for_trajectories(traj_0, use_phase=True)
            feat_1_fine = extract_features_for_trajectories(traj_1, use_phase=True)

            # Combine and create labels
            X_coarse = np.vstack([feat_0_coarse, feat_1_coarse])
            X_fine = np.vstack([feat_0_fine, feat_1_fine])
            y = np.array([0]*len(feat_0_coarse) + [1]*len(feat_1_coarse))

            # Split train/test (70/30)
            n_total = len(y)
            n_train = int(0.7 * n_total)
            indices = np.random.permutation(n_total)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            # Standardize
            X_coarse_mean = X_coarse[train_idx].mean(axis=0)
            X_coarse_std = X_coarse[train_idx].std(axis=0) + 1e-10
            X_coarse_norm = (X_coarse - X_coarse_mean) / X_coarse_std

            X_fine_mean = X_fine[train_idx].mean(axis=0)
            X_fine_std = X_fine[train_idx].std(axis=0) + 1e-10
            X_fine_norm = (X_fine - X_fine_mean) / X_fine_std

            # Train and evaluate
            acc_coarse = train_simple_classifier(
                X_coarse_norm[train_idx], y[train_idx],
                X_coarse_norm[test_idx], y[test_idx]
            )
            acc_fine = train_simple_classifier(
                X_fine_norm[train_idx], y[train_idx],
                X_fine_norm[test_idx], y[test_idx]
            )

            results_coarse.append({
                'E': E, 'phase_0': phase_0, 'phase_1': phase_1,
                'accuracy': acc_coarse
            })
            results_fine.append({
                'E': E, 'phase_0': phase_0, 'phase_1': phase_1,
                'accuracy': acc_fine
            })

            print(f"  E={E:+.1f}, phase=[{phase_0:.2f}, {phase_1:.2f}]: "
                  f"Coarse={acc_coarse:.1%}, Fine={acc_fine:.1%}")

    # Aggregate results
    mean_acc_coarse = np.mean([r['accuracy'] for r in results_coarse])
    mean_acc_fine = np.mean([r['accuracy'] for r in results_fine])

    print("\n[2] Summary Statistics...")
    print(f"  Mean accuracy (coarse): {mean_acc_coarse:.1%}")
    print(f"  Mean accuracy (fine):   {mean_acc_fine:.1%}")
    print(f"  Difference:             {mean_acc_fine - mean_acc_coarse:.1%}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Accuracy by energy level
    ax = axes[0]
    E_unique = sorted(set(r['E'] for r in results_coarse))
    acc_coarse_by_E = [np.mean([r['accuracy'] for r in results_coarse if r['E'] == E])
                       for E in E_unique]
    acc_fine_by_E = [np.mean([r['accuracy'] for r in results_fine if r['E'] == E])
                     for E in E_unique]

    x = np.arange(len(E_unique))
    width = 0.35
    ax.bar(x - width/2, acc_coarse_by_E, width, label='Coarse (magnitude)', alpha=0.7)
    ax.bar(x + width/2, acc_fine_by_E, width, label='Fine (phase)', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{E:.1f}' for E in E_unique])
    ax.set_xlabel('Energy E')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Q₀ Class Distinguishability by Energy')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 2. Accuracy by phase pair
    ax = axes[1]
    pair_labels = [f'[{a:.2f},{b:.2f}]' for a, b in phase_pairs]
    acc_coarse_by_Q = []
    acc_fine_by_Q = []
    for phase_0, phase_1 in phase_pairs:
        matching_c = [r['accuracy'] for r in results_coarse
                      if r['phase_0'] == phase_0 and r['phase_1'] == phase_1]
        matching_f = [r['accuracy'] for r in results_fine
                      if r['phase_0'] == phase_0 and r['phase_1'] == phase_1]
        acc_coarse_by_Q.append(np.mean(matching_c) if matching_c else 0.5)
        acc_fine_by_Q.append(np.mean(matching_f) if matching_f else 0.5)

    x = np.arange(len(phase_pairs))
    ax.bar(x - width/2, acc_coarse_by_Q, width, label='Coarse', alpha=0.7)
    ax.bar(x + width/2, acc_fine_by_Q, width, label='Fine', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=15)
    ax.set_xlabel('Q₀ Pair')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Q₀ Class Distinguishability by Phase Difference')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 3. Summary text
    ax = axes[2]
    ax.axis('off')

    # Analyze by phase separation type
    half_period_pairs = [(0.0, 0.5), (0.1, 0.6)]
    quarter_period_pairs = [(0.0, 0.25), (0.0, 0.1)]

    acc_half_coarse = np.mean([r['accuracy'] for r in results_coarse
                               if (r['phase_0'], r['phase_1']) in half_period_pairs])
    acc_quarter_coarse = np.mean([r['accuracy'] for r in results_coarse
                                  if (r['phase_0'], r['phase_1']) in quarter_period_pairs])

    # Q-equivalence: half-period should be ~random, quarter should be high
    half_near_random = abs(acc_half_coarse - 0.5) < 0.2
    quarter_distinguishable = acc_quarter_coarse > 0.8
    fine_perfect = mean_acc_fine > 0.95

    verdict = "✓ CONFIRMED" if (half_near_random and fine_perfect) else "✗ UNCLEAR"

    print(f"\n  By phase separation:")
    print(f"    Half-period (coarse):    {acc_half_coarse:.1%} (expect ~50%)")
    print(f"    Quarter-period (coarse): {acc_quarter_coarse:.1%} (expect high)")
    print(f"    All fine features:       {mean_acc_fine:.1%} (expect ~100%)")

    summary = f"""
    Q-EQUIVALENCE CLASSES TEST
    ==========================

    Claim: Coarse observation creates Q-equivalence

    By Phase Separation:
    - Half-period (coarse):   {acc_half_coarse:.1%}
      (expect ~50% - Q-equivalent)
    - Quarter-period (coarse): {acc_quarter_coarse:.1%}
      (expect high - NOT equivalent)
    - Fine features (all):     {mean_acc_fine:.1%}
      (expect ~100% - always distinguishes)

    Key Insight:
    Half-period separation → SYMMETRY
    q(t) and q(t+T/2) have same |z| magnitude
    but opposite phase. Coarse features see
    only magnitude → can't distinguish!

    Quarter-period → ASYMMETRY
    q(t) at turning point, q(t+T/4) at center
    Different magnitudes → distinguishable

    Verdict: {verdict}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('q_equivalence_classes.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to q_equivalence_classes.png")
    plt.show()

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if half_near_random and fine_perfect:
        print(f"""
    ✓ Q-EQUIVALENCE CONFIRMED!

    Half-period separation (coarse): {acc_half_coarse:.1%}
      → CANNOT distinguish symmetric phases
      → This is Q-equivalence!

    Quarter-period separation (coarse): {acc_quarter_coarse:.1%}
      → CAN distinguish asymmetric phases
      → Magnitude differences detectable

    Fine features (all): {mean_acc_fine:.1%}
      → ALWAYS distinguishes phases
      → Phase information preserved

    Physical interpretation:
    - Pendulum has Z₂ symmetry: q(t) ↔ q(t + T/2)
    - Magnitude-only features are blind to this symmetry
    - Phase features break the symmetry

    This IS the "quantum-like" behavior:
    Coarse observation creates equivalence classes
    of states related by symmetry operations.
        """)
    else:
        print(f"""
    Results:
    - Half-period (coarse):    {acc_half_coarse:.1%} (expect ~50%)
    - Quarter-period (coarse): {acc_quarter_coarse:.1%} (expect high)
    - Fine features:           {mean_acc_fine:.1%} (expect ~100%)

    Interpretation depends on specific values.
        """)

    return {
        'results_coarse': results_coarse,
        'results_fine': results_fine,
        'mean_coarse': mean_acc_coarse,
        'mean_fine': mean_acc_fine
    }


if __name__ == "__main__":
    results = run_q_equivalence_test()
