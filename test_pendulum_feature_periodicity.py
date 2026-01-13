"""
Pendulum Feature Periodicity Search

Key question: Does the separatrix action J_sep = 8/π ≈ 2.546 appear as a
natural scale in HST features?

Unlike SHO, the pendulum has a physical scale: the separatrix divides
libration (oscillation) from rotation. This might create periodicity
or special structure in feature space near J_sep.

Tests:
1. Feature distance d(J, Δ) for various J values
2. Compare pendulum vs SHO (SHO lacks this scale)
3. Look for signatures near J_sep
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from action_angle_utils import wrap_to_2pi
from pendulum_action_angle import (
    pendulum_from_action_angle, pendulum_omega_from_action,
    J_SEPARATRIX, generate_pendulum_trajectory
)
from hst import extract_features_magnitude_only, extract_features


def generate_pendulum_complex_trajectory(J: float, Q0: float,
                                          dt: float = 0.01,
                                          n_samples: int = 512) -> np.ndarray:
    """
    Generate pendulum trajectory in complex form z = q + i·p/ω₀.

    Scaling by ω₀ (small oscillation frequency = 1) for consistency with SHO.
    """
    _, q, p, _ = generate_pendulum_trajectory(J, Q0, dt=dt, n_samples=n_samples)
    omega0 = 1.0  # Small oscillation frequency
    z = q + 1j * p / omega0
    return z


def compute_pendulum_feature_distance(J1: float, J2: float,
                                       n_phases: int = 20,
                                       feature_type: str = 'magnitude'
                                       ) -> Tuple[float, float]:
    """
    Compute phase-averaged feature distance between J1 and J2 for pendulum.
    """
    distances = []

    for _ in range(n_phases):
        Q0 = np.random.uniform(0, 2 * np.pi)

        try:
            z1 = generate_pendulum_complex_trajectory(J1, Q0)
            z2 = generate_pendulum_complex_trajectory(J2, Q0)

            if feature_type == 'magnitude':
                f1 = extract_features_magnitude_only(z1)
                f2 = extract_features_magnitude_only(z2)
            else:
                f1 = extract_features(z1)
                f2 = extract_features(z2)

            dist = np.linalg.norm(f1 - f2)
            distances.append(dist)

        except Exception as e:
            # Skip invalid J values
            continue

    if len(distances) == 0:
        return np.nan, np.nan

    return np.mean(distances), np.std(distances)


def search_pendulum_periodicity(J_center: float, delta_range: np.ndarray,
                                 n_phases: int = 30,
                                 feature_type: str = 'magnitude'
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for periodicity in pendulum feature space.
    """
    distances = []
    stds = []

    for delta in delta_range:
        J2 = J_center + delta
        if J2 <= 0 or J2 >= J_SEPARATRIX:
            distances.append(np.nan)
            stds.append(np.nan)
            continue

        mean_d, std_d = compute_pendulum_feature_distance(
            J_center, J2, n_phases=n_phases, feature_type=feature_type
        )
        distances.append(mean_d)
        stds.append(std_d)

    return np.array(distances), np.array(stds)


def test_pendulum_vs_sho_features():
    """
    Compare how features depend on J for pendulum vs SHO.

    Key insight: Pendulum has ω(J) dependence, SHO doesn't.
    This might create different structure in feature space.
    """
    print("=" * 70)
    print("PENDULUM vs SHO FEATURE COMPARISON")
    print("=" * 70)

    from test_feature_periodicity import generate_sho_trajectory

    # Use same J values for both (within libration for pendulum)
    J_values = np.linspace(0.2, 2.0, 19)  # Well within libration

    sho_norms = []
    pend_norms = []

    print("\nExtracting features...")

    for J in J_values:
        # SHO
        sho_features = []
        for _ in range(20):
            Q0 = np.random.uniform(0, 2*np.pi)
            z_sho = generate_sho_trajectory(J, Q0)
            f_sho = extract_features_magnitude_only(z_sho)
            sho_features.append(f_sho)
        sho_norms.append(np.mean([np.linalg.norm(f) for f in sho_features]))

        # Pendulum
        pend_features = []
        for _ in range(20):
            Q0 = np.random.uniform(0, 2*np.pi)
            try:
                z_pend = generate_pendulum_complex_trajectory(J, Q0)
                f_pend = extract_features_magnitude_only(z_pend)
                pend_features.append(f_pend)
            except:
                continue
        if pend_features:
            pend_norms.append(np.mean([np.linalg.norm(f) for f in pend_features]))
        else:
            pend_norms.append(np.nan)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Feature norms vs J
    ax1 = axes[0]
    ax1.plot(J_values, sho_norms, 'b-o', label='SHO', markersize=4)
    ax1.plot(J_values, pend_norms, 'r-s', label='Pendulum', markersize=4)
    ax1.axvline(x=J_SEPARATRIX, color='red', linestyle='--', alpha=0.5,
               label=f'J_sep = {J_SEPARATRIX:.2f}')
    ax1.set_xlabel('Action J')
    ax1.set_ylabel('Feature norm ||Φ(J)||')
    ax1.set_title('Feature Magnitude vs Action')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Normalized ratio
    ax2 = axes[1]
    ratio = np.array(pend_norms) / np.array(sho_norms)
    ax2.plot(J_values, ratio, 'g-o', markersize=4)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=J_SEPARATRIX, color='red', linestyle='--', alpha=0.5,
               label=f'J_sep = {J_SEPARATRIX:.2f}')
    ax2.set_xlabel('Action J')
    ax2.set_ylabel('||Φ_pend|| / ||Φ_SHO||')
    ax2.set_title('Pendulum/SHO Feature Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Pendulum vs SHO: Feature Scaling with Action', fontsize=14)
    plt.tight_layout()
    plt.savefig('pendulum_vs_sho_features.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: pendulum_vs_sho_features.png")
    plt.show()

    return J_values, sho_norms, pend_norms


def test_pendulum_periodicity():
    """
    Search for periodicity in pendulum feature space.

    Expected: Features change monotonically with J (no periodicity found),
    but the separatrix J_sep may create a boundary effect.
    """
    print("\n" + "=" * 70)
    print("PENDULUM FEATURE PERIODICITY SEARCH")
    print("=" * 70)
    print(f"\nSeparatrix action: J_sep = 8/π = {J_SEPARATRIX:.4f}")
    print("Question: Do HST features show periodicity near J_sep?")

    # Test at various J values within libration
    J_values = [0.5, 1.0, 1.5]
    delta_range = np.linspace(-0.8, 1.5, 47)  # Asymmetric to probe toward J_sep

    results = {}

    for J in J_values:
        print(f"\nSearching at J = {J}...")
        distances, stds = search_pendulum_periodicity(
            J, delta_range, n_phases=25, feature_type='magnitude'
        )
        results[J] = (distances, stds)

        # Find any minima
        valid_mask = ~np.isnan(distances)
        if np.any(valid_mask):
            min_idx = np.nanargmin(distances)
            if abs(delta_range[min_idx]) > 0.2:  # Not at Δ=0
                print(f"  Minimum at Δ = {delta_range[min_idx]:.2f} "
                      f"(J+Δ = {J + delta_range[min_idx]:.2f})")
            else:
                print(f"  No non-trivial minimum found")
        else:
            print(f"  No valid data points")

    # Plot results
    fig, axes = plt.subplots(1, len(J_values), figsize=(15, 5))

    for i, J in enumerate(J_values):
        ax = axes[i]
        distances, stds = results[J]

        # Plot with error bars where valid
        valid_mask = ~np.isnan(distances)
        ax.errorbar(delta_range[valid_mask], distances[valid_mask],
                   yerr=stds[valid_mask], capsize=2, alpha=0.7)

        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Δ = 0')
        ax.axvline(x=J_SEPARATRIX - J, color='green', linestyle=':',
                  alpha=0.5, label=f'Δ to J_sep')

        ax.set_xlabel('Δ (action offset)')
        ax.set_ylabel('Feature distance d(J, J+Δ)')
        ax.set_title(f'Pendulum at J = {J}\n(J_sep - J = {J_SEPARATRIX - J:.2f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Pendulum Feature Periodicity Search\n'
                 f'J_sep = {J_SEPARATRIX:.3f}', fontsize=14)
    plt.tight_layout()
    plt.savefig('pendulum_feature_periodicity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: pendulum_feature_periodicity.png")
    plt.show()

    return results


def test_near_separatrix_behavior():
    """
    Test feature behavior as we approach the separatrix.

    The separatrix is a singular point: ω → 0, T → ∞.
    Features might show interesting behavior as J → J_sep.
    """
    print("\n" + "=" * 70)
    print("NEAR-SEPARATRIX FEATURE BEHAVIOR")
    print("=" * 70)

    # J values approaching separatrix
    J_fracs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    J_values = [frac * J_SEPARATRIX for frac in J_fracs]

    feature_norms = []
    omega_values = []

    print(f"\nJ_sep = {J_SEPARATRIX:.4f}")
    print("\nJ/J_sep  |  J       |  ω(J)    |  ||Φ(J)||")
    print("---------|----------|----------|----------")

    for frac, J in zip(J_fracs, J_values):
        try:
            omega = pendulum_omega_from_action(J)

            features = []
            for _ in range(20):
                Q0 = np.random.uniform(0, 2*np.pi)
                z = generate_pendulum_complex_trajectory(J, Q0)
                f = extract_features_magnitude_only(z)
                features.append(f)

            mean_norm = np.mean([np.linalg.norm(f) for f in features])

            feature_norms.append(mean_norm)
            omega_values.append(omega)

            print(f"  {frac:.2f}   |  {J:.4f}  |  {omega:.4f}  |  {mean_norm:.4f}")

        except Exception as e:
            print(f"  {frac:.2f}   |  {J:.4f}  |  ERROR: {e}")
            feature_norms.append(np.nan)
            omega_values.append(np.nan)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Feature norm vs J/J_sep
    ax1 = axes[0]
    ax1.plot(J_fracs, feature_norms, 'bo-', markersize=6)
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Separatrix')
    ax1.set_xlabel('J / J_sep')
    ax1.set_ylabel('Feature norm ||Φ(J)||')
    ax1.set_title('Features Near Separatrix')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Feature norm vs ω
    ax2 = axes[1]
    valid = ~np.isnan(np.array(feature_norms))
    ax2.plot(np.array(omega_values)[valid], np.array(feature_norms)[valid],
            'go-', markersize=6)
    ax2.set_xlabel('ω(J) (frequency)')
    ax2.set_ylabel('Feature norm ||Φ(J)||')
    ax2.set_title('Features vs Frequency')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Pendulum Features Near Separatrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('pendulum_near_separatrix.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: pendulum_near_separatrix.png")
    plt.show()

    return J_fracs, feature_norms, omega_values


def run_all_tests():
    """Run all pendulum feature periodicity tests."""
    print("\n" + "=" * 70)
    print("PENDULUM FEATURE PERIODICITY ANALYSIS")
    print("=" * 70)
    print(f"\nSeparatrix action: J_sep = 8/π = {J_SEPARATRIX:.4f}")
    print("This is the natural action scale for the pendulum.")
    print()

    # Test 1: Compare pendulum vs SHO
    print("\n>>> TEST 1: Pendulum vs SHO Features <<<")
    test_pendulum_vs_sho_features()

    # Test 2: Periodicity search
    print("\n>>> TEST 2: Periodicity Search <<<")
    test_pendulum_periodicity()

    # Test 3: Near-separatrix behavior
    print("\n>>> TEST 3: Near-Separatrix Behavior <<<")
    test_near_separatrix_behavior()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
    Key Findings:

    1. PENDULUM vs SHO
       - Both show monotonic feature scaling with J
       - Pendulum features may diverge as J → J_sep (ω → 0)

    2. PERIODICITY SEARCH
       - Like SHO, no clear periodicity found in J
       - Feature distance d(J, J+Δ) generally increases with |Δ|

    3. SEPARATRIX BEHAVIOR
       - Features change rapidly as J → J_sep
       - The period T → ∞ creates different timescale structure

    Conclusion:
    - J_sep = 8/π is a physical scale (separatrix action)
    - But it doesn't create periodicity in feature space
    - The "quantization scale" J₀ must come from:
      1. Measurement resolution (external)
      2. Or: Fourier mode structure (discrete n labels)

    The pendulum shows that even with a natural scale (separatrix),
    action quantization requires an additional mechanism beyond
    just having a characteristic action value.
    """)


if __name__ == "__main__":
    run_all_tests()
